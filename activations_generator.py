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
device: Literal["cuda"] | Literal["cpu"] = "cuda" if t.cuda.is_available() else "cpu"


class ActivationLoader:
    def __init__(self, model_name, tokenizer=None, hf_model=None) -> None:
        self.activations = {}
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
        acts_by_layer: dict,
        labels: list,
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        """write a single batch of activations/labels for every layer."""
        for layer_num, acts in acts_by_layer.items():
            X: t.Tensor = t.stack(acts)
            y: t.Tensor = t.tensor(labels)
            data_to_save = {
                "activations": X,
                "labels": y,
                "metadata": {
                    "layer": layer_num,
                    "model": self.model_name,
                    "batch_id": batch_id,
                },
            }
            save_path: str = f"{ACTIVATIONS_PATH}/{self.model_name}/{language}/{split}/{layer_num}.pt"

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
        Iterate through `dataloader` in batches of `batch_size`, saving each batch.
        `start_index` can be used to skip the first N examples so that you can resume
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

        _, dataloader = get_dataset_and_dataloader(language, split)
        n_layers = len(self.hf_model.model.layers)

        # register hooks
        hook_handles = []
        for layer_number in range(n_layers):
            handle = self.hf_model.model.layers[layer_number].register_forward_hook(
                self.get_activation(f"layer_{layer_number}")
            )
            hook_handles.append(handle)

        processed = 0  # number of examples seen after start_index
        batch_id: int = start_index // batch_size
        batch_acts_by_layer = {i: [] for i in range(n_layers)}
        batch_labels = []

        len_dataloader = len(dataloader)
        try:
            for i, ((sentence_a, sentence_b), label, _) in tqdm(
                enumerate(dataloader),
                desc="Extracting all layers",
                total=len_dataloader,
            ):
                if i < start_index:  # skip until we reach the resume point
                    continue

                prompt: str = f"Premise: {sentence_a} Hypothesis: {sentence_b} Label:"
                tokens = self.tokenizer(prompt, return_tensors="pt").to(device)

                with t.no_grad():
                    self.hf_model(**tokens)

                for layer_number in range(n_layers):
                    act = self.activations[f"layer_{layer_number}"][0, -1, :].cpu()
                    batch_acts_by_layer[layer_number].append(act)

                batch_labels.append(label)
                processed += 1

                # flush a batch when full or when we hit the amount_to_generate limit
                if (processed % batch_size == 0) or (
                    amount_to_generate and processed == amount_to_generate
                ):
                    if save_to_disk:
                        self._save_batch(
                            batch_acts_by_layer, batch_labels, language, split, batch_id
                        )
                    batch_id += 1
                    batch_acts_by_layer = {i: [] for i in range(n_layers)}
                    batch_labels = []

                if amount_to_generate and processed >= amount_to_generate:
                    break
        finally:
            for handle in hook_handles:
                handle.remove()

    def load_activations(self, language: str, split: str, layer_number: int):
        """Load activations and labels for a specific layer."""
        save_path = (
            f"{ACTIVATIONS_PATH}/{self.model_name}/{language}/{split}/{layer_number}.pt"
        )
        data = t.load(save_path)
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
    amount_to_generate = 64
    batch_size = 32

    model_name = "olmo_model"
    language = "es"

    activation_loader = ActivationLoader(model_name)

    for language in ["en", "es"]:
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

        print(activation_loader.load_activations(language, "train", 1))
