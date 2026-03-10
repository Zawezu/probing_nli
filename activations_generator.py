from typing import Literal

from sick_loader import get_dataset_and_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm

from pathlib import Path


MODEL_FOLDER = "models"
MODEL_FILEPATHS: dict[str, str] = {"olmo_model": f"./{MODEL_FOLDER}/olmo_model"}

ACTIVATIONS_PATH = "./data/activations/"

# global device so that methods can refer to it
device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


class ActivationLoader:
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
        labels: list[int],
        control_labels: list[int],
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        """write a single batch of activations/labels for every layer."""
        for layer_num, acts in acts_by_layer.items():
            labels_tensor: t.Tensor = t.tensor(labels)
            control_labels_tensor: t.Tensor = t.tensor(control_labels)
            print(acts)
            print(acts.shape)
            data_to_save: dict[str, t.Tensor | dict[str, int | str]] = {
                "activations": acts,
                "labels": labels_tensor,
                "control_labels": control_labels_tensor,
                "metadata": {
                    "layer": layer_num,
                    "model": self.model_name,
                    "batch_id": batch_id,
                },
            }
            save_path: str = f"{ACTIVATIONS_PATH}/{self.model_name}/{language}/{split}/batch{batch_id}_layer{layer_num}.pt"

            # If the folder doesn't exist, create it
            path_obj = Path(save_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            t.save(data_to_save, save_path)
            print(
                f"Saved {len(acts)} samples for layer {layer_num} (batch {batch_id}) to {save_path}"
            )

    def generate_activations(
        self,
        language,
        split,
        save_to_disk=True,
        amount_to_generate=None,
        batch_size: int = 128,
        start_index: int = 0,
    ) -> None:
        """
        Iterate through `dataloader` in batches, processing all sentences in each batch together.
        `start_index` can be used to skip the first N batches so that you can resume
        after a crash.

        `amount_to_generate` limits the number of examples produced *after* the start
        index.
        """
        if self.tokenizer is None or self.hf_model is None:
            print(
                f"The tokenizer or model were not loaded for the ActivationLoader of {self.model_name}. Loading now."
            )
            self.load_model()

        assert self.tokenizer is not None and self.hf_model is not None

        _, dataloader = get_dataset_and_dataloader(
            language, split, batch_size=batch_size
        )
        n_layers = len(self.hf_model.model.layers)

        # register hooks
        hook_handles = []
        for layer_num in range(n_layers):
            handle = self.hf_model.model.layers[layer_num].register_forward_hook(
                self.get_activation(f"layer_{layer_num}")
            )
            hook_handles.append(handle)

        processed = 0  # number of examples seen after start_index
        batch_id: int = start_index

        len_dataloader = len(dataloader)
        try:
            for batch_num, (
                (sentence_a_batch, sentence_b_batch),
                batch_labels,
                batch_control_labels,
                _,
            ) in tqdm(
                enumerate(dataloader),
                desc="Extracting all layers",
                total=len_dataloader,
            ):
                if batch_num < start_index:  # skip until we reach the resume point
                    continue

                batch_acts_by_layer: dict[int, t.Tensor] = {}

                # Create prompts for all sentences in the batch
                prompts: list[str] = [
                    f"Premise: {sent_a} Hypothesis: {sent_b} Label:"
                    for sent_a, sent_b in zip(sentence_a_batch, sentence_b_batch)
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

                print(f"batch_acts_by_layer[0]:\n{batch_acts_by_layer[0]}")
                print(f"batch_acts_by_layer[1]:\n{batch_acts_by_layer[1]}")

                processed += len(batch_labels)

                # Save batch and reset
                if save_to_disk:
                    self._save_batch(
                        batch_acts_by_layer,
                        batch_labels,
                        batch_control_labels,
                        language,
                        split,
                        batch_id,
                    )
                batch_id += 1

                if amount_to_generate and processed >= amount_to_generate:
                    break
        finally:
            for handle in hook_handles:
                handle.remove()

    def load_activations(
        self,
        language: str,
        split: str,
        layer_num: int,
        control: bool,
        batch_id=0,  # !!! batch_id here is temporary
    ) -> tuple[t.Tensor, t.Tensor]:
        """Load activations and labels for a specific layer."""
        save_path = f"{ACTIVATIONS_PATH}/{self.model_name}/{language}/{split}/batch{batch_id}_layer{layer_num}.pt"
        print(f"Loading activations from {save_path}")
        data = t.load(save_path)

        if control:
            return data["activations"], data["control_labels"]
        else:
            return data["activations"], data["labels"]

    def load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_FILEPATHS[self.model_name], local_files_only=True
        )
        self.hf_model = AutoModelForCausalLM.from_pretrained(
            MODEL_FILEPATHS[self.model_name], local_files_only=True
        ).to(device)
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


if __name__ == "__main__":
    save_to_disk = True
    generate = False
    amount_to_generate = 2
    batch_size = 2

    model_name = "olmo_model"
    # languages_generated = ["en", "es"]
    languages_generated = ["en"]

    activation_loader: ActivationLoader = ActivationLoader(model_name)

    if generate:
        for language in languages_generated:
            # example: start at 256th example, batch size 128
            activation_loader.generate_activations(
                language,
                "train",
                save_to_disk=save_to_disk,
                amount_to_generate=amount_to_generate,
                batch_size=batch_size,
                start_index=0,
            )

            activation_loader.generate_activations(
                language,
                "test",
                save_to_disk=save_to_disk,
                amount_to_generate=amount_to_generate,
                batch_size=batch_size,
                start_index=0,
            )

            activation_loader.generate_activations(
                language,
                "val",
                save_to_disk=save_to_disk,
                amount_to_generate=amount_to_generate,
                batch_size=batch_size,
                start_index=0,
            )

    # Sanity check: just print some activations for the first 3 layers
    split_shown = "train"
    batch_id = 0
    for language in languages_generated:
        for layer in range(3):
            loaded_activations = activation_loader.load_activations(
                language, split_shown, layer, control=False, batch_id=batch_id
            )
            print(
                f"Language: {language}, split shown: {split_shown}. Layer: {layer}. Batch: {batch_id}"
            )
            print(loaded_activations)
            print(loaded_activations[0].size())
