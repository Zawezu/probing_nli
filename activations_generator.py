from sick_loader import get_dataset_and_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from tqdm import tqdm

import os

MODEL_FOLDER = "models"
MODEL_FILEPATHS = {"olmo_model": f"./{MODEL_FOLDER}/olmo_model"}

ACTIVATIONS_PATH = "./data/activations/"


class ActivationLoader:
    def __init__(self, model_name, tokenizer=None, hf_model=None) -> None:
        self.activations = {}
        self.model_name = model_name

        self.tokenizer = tokenizer
        self.hf_model = hf_model

    def get_activation(self, name):
        def hook(model, input, output):
            # 'output' is a tuple for some models; we want the first element (the tensor)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()

        return hook

    def generate_activations(
        self, split, save_to_disk=True, amount_to_generate=None
    ) -> None:
        if self.tokenizer is None and self.tokenizer is None:
            print(
                f"The tokenizer and model were not loaded for the ActivationLoader of {self.model_name}. Loading now."
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                MODEL_FILEPATHS[model_name], local_files_only=True
            )
            self.hf_model = AutoModelForCausalLM.from_pretrained(
                MODEL_FILEPATHS[model_name], local_files_only=True
            ).to(device)
        elif self.tokenizer is None or self.tokenizer is None:
            raise Exception(
                "tokenizer was {self.tokenizer}, but hf_model was {self.hf_model}"
            )

        _, dataloader = get_dataset_and_dataloader("en", split)
        """Extract activations from all layers in a single forward pass."""
        all_acts_by_layer = {i: [] for i in range(len(self.hf_model.model.layers))}
        all_labels = []

        # 1. Register hooks on all layers
        hook_handles = []
        for layer_number in range(len(self.hf_model.model.layers)):
            hook_handle = self.hf_model.model.layers[
                layer_number
            ].register_forward_hook(self.get_activation(f"layer_{layer_number}"))
            hook_handles.append(hook_handle)

        # 2. Single extraction loop
        try:
            for (sentence_a, sentence_b), label in tqdm(
                dataloader, desc="Extracting all layers"
            ):
                prompt = f"Premise: {sentence_a} Hypothesis: {sentence_b} Label:"
                tokens = self.tokenizer(prompt, return_tensors="pt").to(device)

                with t.no_grad():
                    self.hf_model(**tokens)

                # Extract last token activations from all layers
                for layer_number in range(len(self.hf_model.model.layers)):
                    act = self.activations[f"layer_{layer_number}"][0, -1, :].cpu()
                    all_acts_by_layer[layer_number].append(act)

                all_labels.append(label)

                if amount_to_generate and len(all_labels) >= amount_to_generate:
                    break
        finally:
            for handle in hook_handles:
                handle.remove()

        # 3. Save activations for each layer
        for layer_number in range(self.get_number_of_layers()):
            X = t.stack(all_acts_by_layer[layer_number])
            y = t.tensor(all_labels)

            if save_to_disk:
                data_to_save = {
                    "activations": X,
                    "labels": y,
                    "metadata": {"layer": layer_number, "model": model_name},
                }
                save_path = f"{ACTIVATIONS_PATH}/{model_name}/{split}/{layer_number}.pt"
                t.save(data_to_save, save_path)
                print(
                    f"Saved {len(all_acts_by_layer[layer_number])} samples to {save_path}"
                )

    def load_activations(self, split: str, layer_number: int):
        """Load activations and labels for a specific layer.

        Args:
            model_name: Name of the model (e.g., "olmo_model")
            layer_number: Layer number to load

        Returns:
            Tuple of (activations, labels) tensors
        """
        save_path = f"{ACTIVATIONS_PATH}/{self.model_name}/{split}/{layer_number}.pt"
        data = t.load(save_path)
        return data["activations"], data["labels"]

    def get_number_of_layers(self) -> int:
        if self.hf_model is not None:
            return len(self.hf_model.model.layers)
        else:
            print("Model not loaded. Getting the number of layers the hard way")

            # Create a list of all the numbers of the layer activation files
            nums = [
                int(f[:-3])
                for f in os.listdir(f"{ACTIVATIONS_PATH}/{self.model_name}/{"train"}")
                if f.endswith(".pt") and f[:-3].isdigit()
            ]

            # Find the maximum number, which corresponds to the highest layer. Add one because layers start counting at 0
            # This finds the number of layers
            return max(nums) + 1


if __name__ == "__main__":
    device = "cuda" if t.cuda.is_available() else "cpu"

    amount_to_generate = 5
    save_to_disk = False

    model_name = "olmo_model"

    olmo_activation_loader = ActivationLoader("olmo_model")

    olmo_activation_loader.generate_activations(
        "train", save_to_disk=save_to_disk, amount_to_generate=amount_to_generate
    )

    olmo_activation_loader.generate_activations(
        "test", save_to_disk=save_to_disk, amount_to_generate=amount_to_generate
    )

    print(olmo_activation_loader.load_activations("train", 1))

    # Currently no validation set on SICK
    # olmo_activation_loader.generate_activations("val", amount_to_generate=amount_to_generate)
