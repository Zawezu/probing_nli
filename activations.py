from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from pathlib import Path

from sick import SICKMergedDataset
from common_constants import ACTIVATIONS_FOLDER, MODELS_FOLDER

device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")


class SpecialCase:
    def __init__(self, language: str, split: str, start_from_batch: int | None) -> None:
        self.language: str = language
        self.split: str = split
        self.start_from_batch: int | None = start_from_batch

    def __str__(self) -> str:
        return f"SpecialCase(language={self.language}, split={self.split}, start_from_batch={self.start_from_batch})"


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
            # print(
            #     f"Saved {len(acts)} samples for layer {layer_num} (batch {batch_id}) to {save_path}"
            # )

        print(
            f"Saved samples for batch {batch_id} to {get_activations_filepath(self.model_name, language, split, 'n', batch_id)}"
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
                _,
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
                    self.generate_prompt(sent_a, sent_b, language)
                    for sent_a, sent_b in zip(
                        sentence_tuple_batch[0], sentence_tuple_batch[1]
                    )
                ]
                # print(prompts)

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
        model_filepath: str = f"{MODELS_FOLDER}/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_filepath, local_files_only=True
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_filepath, local_files_only=True
        ).to(device)  # type: ignore
        n_layers_txt_filepath = self.get_n_layers_txt_filepath()
        # Create parent directory if it doesn't exist
        path_obj = Path(n_layers_txt_filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(n_layers_txt_filepath, "w") as file:
            file.write(str(len(self.hf_model.model.layers)))

    def get_number_of_layers(self) -> int:
        if self.hf_model is not None:
            return len(self.hf_model.model.layers)
        else:
            n_layers_txt_filepath: str = self.get_n_layers_txt_filepath()
            print(
                f"Model not loaded. Getting the number of layers from {n_layers_txt_filepath}"
            )
            try:
                with open(n_layers_txt_filepath, "r") as file:
                    return int(file.readline())
            except FileNotFoundError:
                print("Could not find . Loading model")
                self.load_model()
                return self.get_number_of_layers()

    def get_n_layers_txt_filepath(self) -> str:
        return f"{ACTIVATIONS_FOLDER}/{self.model_name}/n_layers.txt"

    @staticmethod
    def generate_prompt(sent_a, sent_b, language) -> str:
        match language:
            case "en":
                return f"Premise: {sent_a}. Hypothesis: {sent_b}. Do these sentences entail, contradict, or are neutral to each other?"
            case "es":
                return f"Premisa: {sent_a}. Hipótesis: {sent_b}. ¿Estas frases implican, contradicen o son neutrales entre sí?"
            case _:
                raise KeyError("Invalid language passed")


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

        self.activations, self.labels = self.load_activations_from_merged()

    # Outdated function. The new function loads activations from the merged file. Keep but ignore for now
    # def load_activations(self) -> tuple[Tensor, Tensor]:
    #     """
    #     Loads the tensors form the pt files of every batch and puts them into a single unified tensor
    #     """
    #     total_activations_list: list[Tensor] = []
    #     total_labels_list: list[Tensor] = []

    #     batch_id = 0
    #     i = 0
    #     while True:
    #         activations_filepath: str = get_activations_filepath(
    #             self.model_name, self.language, self.split, self.layer_num, batch_id
    #         )

    #         # Check if file exists
    #         if not Path(activations_filepath).exists():
    #             if batch_id == 0:
    #                 raise FileNotFoundError(
    #                     f"No activation file found at {activations_filepath}. "
    #                     f"Make sure activations have been generated for {self.model_name}, "
    #                     f"language={self.language}, split={self.split}, layer={self.layer_num}"
    #                 )
    #             else:
    #                 # No more batches found
    #                 break

    #         data = t.load(activations_filepath)

    #         # print(f"Loaded activations from {activations_filepath}")

    #         activations: Tensor = data["activations"]

    #         new_activations: list[Tensor] = list(activations)

    #         new_labels: Tensor = t.IntTensor(
    #             self.original_dataset.get_labels_in_range(
    #                 i, i + len(new_activations), self.probing_task
    #             )
    #         )
    #         # print(new_labels)
    #         i += len(new_activations)

    #         # Convert tensors to lists of individual tensors and extend
    #         total_activations_list.extend(new_activations)
    #         total_labels_list.extend(new_labels)

    #         batch_id += 1

    #     return t.stack(total_activations_list, dim=0), t.stack(total_labels_list, dim=0)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.activations[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.activations)

    def load_activations_from_merged(self) -> tuple[Tensor, Tensor]:
        """
        Load activations and labels from a merged activation file.
        """
        original_dataset: SICKMergedDataset = SICKMergedDataset(
            self.language, self.split
        )

        # Load merged activation file
        merged_filepath: str = f"{get_activations_filepath(self.model_name, self.language, self.split, self.layer_num, None)}_merged.pt"

        if not Path(merged_filepath).exists():
            raise FileNotFoundError(
                f"Merged activation file not found at {merged_filepath}. "
                f"Make sure activations have been merged for {self.model_name}, "
                f"language={self.language}, split={self.split}, layer={self.layer_num}"
            )

        data = t.load(merged_filepath, weights_only=True)
        activations: Tensor = data["activations"]

        # Get labels for all samples
        num_samples = len(activations)
        labels: Tensor = t.IntTensor(
            original_dataset.get_labels_in_range(0, num_samples, self.probing_task)
        )

        return activations, labels


def get_activations_filepath(
    model_name: str,
    language: str,
    split: str,
    layer_num: int | str | None,
    batch_id: int | str | None,
) -> str:
    if layer_num is not None and batch_id is not None:
        if isinstance(batch_id, "int"):
            return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}_batch{batch_id}.pt"
        elif isinstance(batch_id, "str"):
            return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}_{batch_id}.pt"
    elif layer_num is not None and batch_id is None:
        return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}"
    elif layer_num is None and batch_id is not None:
        raise KeyError("If layer_numb is None, batch_id cannot be None")
    else:
        return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}"


def delete_individual_file(filepath, ignore_substring, actually_delete) -> None:
    print(filepath.name)
    if ignore_substring and ignore_substring in filepath.name:
        print(f"Ignoring {filepath} because {ignore_substring} is in its filename")
    else:
        if actually_delete:
            filepath.unlink()
        print(f"Deleted {filepath}")


def delete_activations_file(
    model_name: str,
    language: str,
    split: str,
    layer_num: int | str | None = None,
    batch_id: int | None = None,
    ignore_substring: str = "",
    actually_delete: bool = True,
) -> None:
    """
    Delete activation pt file(s).

    If both layer_num and batch_id are provided, deletes the specific file.
    If batch_id is None and layer_num is provided, deletes all batches for that layer.
    If layer_num is None and batch_id is provided, deletes all layers for that batch.
    If both are None, deletes all activations for this model/language/split.
    """
    directory = Path(get_activations_filepath(model_name, language, split, None, None))

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    deleted_count = 0
    if layer_num is not None and batch_id is not None:
        # Delete specific file
        filepath = Path(
            get_activations_filepath(model_name, language, split, layer_num, batch_id)
        )
        delete_individual_file(filepath, ignore_substring, actually_delete)
        deleted_count = 1
    elif layer_num is not None and batch_id is None:
        # Delete all batches for this layer
        pattern = f"layer{layer_num}_batch*.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1
    elif layer_num is None and batch_id is not None:
        # Delete all layers for this batch
        pattern = f"layer*_batch{batch_id}.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1
    else:
        # Delete all activations for this model/language/split
        pattern = "layer*_batch*.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1

    if deleted_count == 0:
        print(f"No files found matching the criteria in {directory}")


def merge_activation_batches(
    model_name: str, language: str, split: str, layer_num: int | str
) -> str:
    # Get the directory containing batch files
    directory = Path(f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}")

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all batch files for this layer, sorted by batch number
    pattern: str = f"layer{layer_num}_batch*.pt"
    batch_files: list[Path] = sorted(directory.glob(pattern))

    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found for {model_name}/{language}/{split}/layer{layer_num}"
        )

    # Load and concatenate all activations
    all_activations: list[Tensor] = []
    for batch_file in batch_files:
        data = t.load(batch_file)
        all_activations.append(data["activations"])

    # Concatenate all activations along the batch dimension
    merged_activations: Tensor = t.cat(all_activations, dim=0)

    # Save merged file
    merged_filepath: str = f"{get_activations_filepath(model_name, language, split, layer_num, None)}_merged.pt"

    data_to_save = {
        "activations": merged_activations,
        "metadata": {
            "layer": layer_num,
            "model": model_name,
            "merged": True,
            "num_batches": len(batch_files),
        },
    }

    t.save(data_to_save, merged_filepath)
    print(
        f"Merged {len(batch_files)} batch files for {model_name}, {language}, {split}, layer{layer_num}"
    )
    print(f"Merged activations shape: {merged_activations.shape}")
    print(f"Saved to {merged_filepath}")

    return merged_filepath


def generate_all_activations(
    activation_loader,
    languages_to_generate: list,
    splits_to_generate: list,
    amount_to_generate: int | None,
    save_to_disk: bool,
    batch_size: int,
    special_cases: list[SpecialCase] = [],
) -> None:
    for language in languages_to_generate:
        for split in splits_to_generate:
            print(f"{'-'*20}\nGenerating language {language}, split {split}\n{'-'*20}")
            start_from_batch: int = 0

            skip_split = False
            # Handle special cases where we want to start generating from another batch or skip a split.
            for special_case in special_cases:
                if special_case.language == language and special_case.split == split:
                    if special_case.start_from_batch is None:
                        skip_split = True
                        print(f"Handling special case: {special_case}. Skipping split")
                    else:
                        start_from_batch: int = special_case.start_from_batch
                        print(
                            f"Handling special case: {special_case}. Starting from batch {special_case.start_from_batch}"
                        )

            if not skip_split:
                activation_loader.generate_activations(
                    language,
                    split,
                    save_to_disk=save_to_disk,
                    amount_to_generate=amount_to_generate,
                    batch_size=batch_size,
                    start_from_batch=start_from_batch,
                )
