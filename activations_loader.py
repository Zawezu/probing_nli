from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from pathlib import Path

from sick_loader import SICKMergedDataset
from common_constants import ACTIVATIONS_PATH, MODEL_FILEPATHS

device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")


class SpecialCase:
    def __init__(self, language: str, split: str, start_from_batch: int) -> None:
        self.language: str = language
        self.split: str = split
        self.start_from_batch: int = start_from_batch


class ActivationSaver:
    def __init__(self, model_name, tokenizer=None, hf_model=None) -> None:
        self.activations: dict[str, t.Tensor] = {}
        self.model_name: str = model_name

        self.tokenizer = tokenizer
        self.hf_model = hf_model

    def get_activation(self, name):
        def hook(model, input, output) -> None:
            # 'output' is a tuple for some models; we want the first element (the tensor)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()

        return hook

    def _save_batch(
        self,
        acts_by_layer: dict[int, t.Tensor],
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        """write a single batch of activations for every layer."""
        for layer_num, acts in acts_by_layer.items():
            data_to_save: dict[str, t.Tensor | dict[str, int | str]] = {
                "activations": acts,
                "metadata": {
                    "layer": layer_num,
                    "model": self.model_name,
                    "batch_id": batch_id,
                },
            }
            save_path: str = get_activations_filepath(
                self.model_name, language, split, layer_num, batch_id
            )

            # If the folder doesn't exist, create it
            path_obj = Path(save_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            t.save(data_to_save, save_path)
            print(
                f"Saved {len(acts)} samples for layer {layer_num} (batch {batch_id}) to {save_path}"
            )

    def generate_activations(
        self,
        language: str,
        split: str,
        save_to_disk=True,
        amount_to_generate=None,
        batch_size: int = 128,
        start_from_batch: int = 0,
    ) -> None:
        """
        Iterate through `dataloader` in batches, processing all sentences in each batch together.
        `start_from_batch` can be used to skip the first N batches so that you can resume
        after a crash.

        `amount_to_generate` limits the number of batches produced *after* the start
        index.
        """
        if self.tokenizer is None or self.hf_model is None:
            print(
                f"The tokenizer or model were not loaded for the ActivationLoader of {self.model_name}. Loading now."
            )
            self.load_model()

        assert self.tokenizer is not None and self.hf_model is not None

        dataset: SICKMergedDataset = SICKMergedDataset(language, split)
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

        n_layers: int = len(self.hf_model.model.layers)

        # register hooks
        hook_handles = []
        for layer_num in range(n_layers):
            handle = self.hf_model.model.layers[layer_num].register_forward_hook(
                self.get_activation(f"layer_{layer_num}")
            )
            hook_handles.append(handle)

        batch_id: int = start_from_batch

        len_dataloader: int = len(dataloader)
        try:
            for batch_num, (
                sentence_tuple_batch,
                _,
                original_ids_batch,
            ) in tqdm(
                enumerate(dataloader),
                desc="Extracting all layers",
                total=len_dataloader,
            ):
                if batch_num < start_from_batch:  # skip until we reach the resume point
                    continue

                batch_acts_by_layer: dict[int, t.Tensor] = {}

                # Create prompts for all sentences in the batch
                prompts: list[str] = [
                    f"Premise: {sent_a} Hypothesis: {sent_b} Label:"
                    for sent_a, sent_b in zip(
                        sentence_tuple_batch[0], sentence_tuple_batch[1]
                    )
                ]

                # Tokenize entire batch at once
                tokens = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                    device
                )

                with t.no_grad():
                    self.hf_model(**tokens)

                # Extract activations for all samples in the batch
                for layer_num in range(n_layers):
                    acts: t.Tensor = self.activations[f"layer_{layer_num}"][
                        :, -1, :
                    ].cpu()

                    batch_acts_by_layer[layer_num] = acts

                # Clear GPU memory
                self.activations.clear()
                t.cuda.empty_cache()

                # print(f"batch_acts_by_layer[0]:\n{batch_acts_by_layer[0]}")
                # print(f"batch_acts_by_layer[1]:\n{batch_acts_by_layer[1]}")

                # Save batch and reset
                if save_to_disk:
                    self._save_batch(
                        batch_acts_by_layer,
                        language,
                        split,
                        batch_id,
                    )
                batch_id += 1

                if (
                    amount_to_generate
                    and batch_id - start_from_batch >= amount_to_generate
                ):
                    break
        finally:
            for handle in hook_handles:
                handle.remove()

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_FILEPATHS[self.model_name], local_files_only=True
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_FILEPATHS[self.model_name], local_files_only=True
        ).to(device)  # type: ignore
        with open(f"{ACTIVATIONS_PATH}/{self.model_name}/n_layers.txt", "w") as file:
            file.write(str(len(self.hf_model.model.layers)))

    def get_number_of_layers(self) -> int:
        if self.hf_model is not None:
            return len(self.hf_model.model.layers)
        else:
            print("Model not loaded. Getting the number of layers from n_layers.txt")
            with open(
                f"{ACTIVATIONS_PATH}/{self.model_name}/n_layers.txt", "r"
            ) as file:
                return int(file.readline())


class ActivationDataset(Dataset):
    def __init__(
        self,
        language: str,
        split: str,
        layer_num: int,
        probing_task: str,
        model_name: str,
    ) -> None:
        self.language: str = language
        self.split: str = split
        self.layer_num: int = layer_num
        self.probing_task: str = probing_task
        self.model_name: str = model_name
        self.original_dataset: SICKMergedDataset = SICKMergedDataset(language, split)

        self.activations, self.labels = self.load_activations()

    def load_activations(self) -> tuple[Tensor, Tensor]:
        """
        Loads the tensors form the pt files of every batch and puts them into a single unified tensor
        """
        total_activations_list: list[Tensor] = []
        total_labels_list: list[Tensor] = []

        batch_id = 0
        i = 0
        while True:
            try:
                activations_filepath: str = get_activations_filepath(
                    self.model_name, self.language, self.split, self.layer_num, batch_id
                )
                data = t.load(activations_filepath)
            except FileNotFoundError:
                # print("No more batches found")
                break

            # print(f"Loaded activations from {activations_filepath}")

            activations: Tensor = data["activations"]

            new_activations: list[Tensor] = list(activations)

            new_labels: Tensor = t.IntTensor(
                self.original_dataset.get_labels_in_range(
                    i, i + len(new_activations), self.probing_task
                )
            )
            # print(new_labels)
            i += len(new_activations)

            # Convert tensors to lists of individual tensors and extend
            total_activations_list.extend(new_activations)
            total_labels_list.extend(new_labels)

            batch_id += 1

        return t.stack(total_activations_list, dim=0), t.stack(total_labels_list, dim=0)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.activations[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.activations)


def get_activations_filepath(
    model_name: str, language: str, split: str, layer_num: int, batch_id: int
) -> str:
    return f"{ACTIVATIONS_PATH}/{model_name}/{language}/{split}/layer{layer_num}_batch{batch_id}.pt"


def generate_all_activations(
    languages_to_generate: list,
    splits_to_generate: list,
    amount_to_generate: int | None,
    save_to_disk: bool,
    batch_size: int,
    special_cases: list[SpecialCase] = [],
) -> None:
    for language in languages_to_generate:
        for split in splits_to_generate:
            print(f"{"-"*20}\nGenerating language {language}, split {split}\n{"-"*20}")
            start_from_batch: int = 0

            # Handle special cases where we want to start generating from another batch.
            for special_case in special_cases:
                if special_case.language == language and special_case.split == split:
                    start_from_batch: int = special_case.start_from_batch
                    print(
                        f"Handling special case: {special_case}. Starting from batch {special_case.start_from_batch}"
                    )

            activation_loader.generate_activations(
                language,
                split,
                save_to_disk=save_to_disk,
                amount_to_generate=amount_to_generate,
                batch_size=batch_size,
                start_from_batch=start_from_batch,
            )


if __name__ == "__main__":
    # debug = True
    # print(f"Debug mode: {debug}")

    # if debug:
    #     save_to_disk: bool = False
    #     generate: bool = True
    #     amount_to_generate: int | None = 4
    #     batch_size: int = 2
    #     model_name: str = "olmo_model"
    #     languages_to_generate: list[str] = ["en"]
    #     splits_to_generate: list[str] = ["train"]
    # else:
    #     save_to_disk: bool = True
    #     generate: bool = True
    #     amount_to_generate: int | None = None
    #     batch_size: int = 128
    #     model_name: str = "olmo_model"
    #     languages_to_generate: list[str] = LANGUAGES
    #     splits_to_generate: list[str] = SPLITS

    save_to_disk: bool = True
    generate: bool = True
    amount_to_generate: int | None = 4
    batch_size: int = 256
    model_name: str = "olmo_model"
    languages_to_generate: list[str] = ["en"]
    splits_to_generate: list[str] = ["train"]

    print(
        f"save_to_disk={save_to_disk}\ngenerate={generate}\namount_to_generate={amount_to_generate}\nbatch_size={batch_size}\nmodel_name={model_name}\nlanguages_to_generate={languages_to_generate}"
    )

    activation_loader: ActivationSaver = ActivationSaver(model_name)

    # special_cases: list[SpecialCase] = [SpecialCase("es", "train", 26)]
    special_cases: list[SpecialCase] = []

    if generate:
        generate_all_activations(
            languages_to_generate,
            splits_to_generate,
            amount_to_generate,
            save_to_disk,
            batch_size,
            special_cases=special_cases,
        )

    activations_dataset: ActivationDataset = ActivationDataset(
        "en", "train", 0, "standard", model_name
    )

    for i in range(8):
        print(f"{activations_dataset[i]}\n-----------------")
