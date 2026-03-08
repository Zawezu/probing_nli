from sick_loader import get_dataset_and_dataloader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch import Tensor
from jaxtyping import Float
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


# # import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, FactoredMatrix

MODEL_FOLDER = "models"
ACTIVATIONS_PATH = f"./data/activations/"

activations = {}

def get_activation(name):
    def hook(model, input, output):
        # 'output' is a tuple for some models; we want the first element (the tensor)
        if isinstance(output, tuple):
            activations[name] = output[0].detach()
        else:
            activations[name] = output.detach()
    return hook

def generate_activations(layer_number, model_name, save_to_disk=True, amount_to_generate=None):
    all_acts = []
    all_labels = []

    # 1. Register Hook
    hook_handle = hf_model.model.layers[layer_number].register_forward_hook(
        get_activation(f"layer_{layer_number}")
    )
    
    # 2. Extraction Loop
    try:
        # Using tqdm for a progress bar
        for (sentence_a, sentence_b), label in tqdm(sick_en_dataloader_train, desc=f"Probing Layer {layer_number}"):
            prompt = f"Premise: {sentence_a} Hypothesis: {sentence_b} Label:"
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
        
            with t.no_grad():
                hf_model(**tokens)

            # Extract last token and move to CPU to save GPU memory
            # raw_acts shape: [1, seq_len, d_model]
            act = activations[f"layer_{layer_number}"][0, -1, :].cpu()
            
            all_acts.append(act)
            all_labels.append(label)

            if amount_to_generate and len(all_acts) >= amount_to_generate:
                break
    finally:
        hook_handle.remove()

    # 3. Format into Tensors
    # X shape: [n_samples, d_model]
    X = t.stack(all_acts)
    # y shape: [n_samples]
    y = t.tensor(all_labels)

    if save_to_disk:
        # 4. Save to disk
        data_to_save = {
            "activations": X,
            "labels": y,
            "metadata": {
                "layer": layer_number,
                "model": model_name
            }
        }

        save_path = f"{ACTIVATIONS_PATH}/{model_name}/{layer_number}"

        t.save(data_to_save, save_path)
        print(f"Saved {len(all_acts)} samples to {save_path}")

    return X, y

def generate_activations_all_layers(model_name, save_to_disk=True, amount_to_generate=None):
    """Extract activations from all layers in a single forward pass."""
    all_acts_by_layer = {i: [] for i in range(len(hf_model.model.layers))}
    all_labels = []
    
    # 1. Register hooks on all layers
    hook_handles = []
    for layer_number in range(len(hf_model.model.layers)):
        hook_handle = hf_model.model.layers[layer_number].register_forward_hook(
            get_activation(f"layer_{layer_number}")
        )
        hook_handles.append(hook_handle)
    
    # 2. Single extraction loop
    try:
        for (sentence_a, sentence_b), label in tqdm(sick_en_dataloader_train, desc=f"Extracting all layers"):
            prompt = f"Premise: {sentence_a} Hypothesis: {sentence_b} Label:"
            tokens = tokenizer(prompt, return_tensors="pt").to(device)
        
            with t.no_grad():
                hf_model(**tokens)

            # Extract last token activations from all layers
            for layer_number in range(len(hf_model.model.layers)):
                act = activations[f"layer_{layer_number}"][0, -1, :].cpu()
                all_acts_by_layer[layer_number].append(act)
            
            all_labels.append(label)

            if amount_to_generate and len(all_labels) >= amount_to_generate:
                break
    finally:
        for handle in hook_handles:
            handle.remove()

    # 3. Save activations for each layer
    for layer_number in range(len(hf_model.model.layers)):
        X = t.stack(all_acts_by_layer[layer_number])
        y = t.tensor(all_labels)

        if save_to_disk:
            data_to_save = {
                "activations": X,
                "labels": y,
                "metadata": {
                    "layer": layer_number,
                    "model": model_name
                }
            }
            save_path = f"{ACTIVATIONS_PATH}/{model_name}/{layer_number}"
            t.save(data_to_save, save_path)
            print(f"Saved {len(all_acts_by_layer[layer_number])} samples to {save_path}")

class LRProbe(t.nn.Module):
    def __init__(self, d_in: int, scaler_mean: Tensor | None = None, scaler_scale: Tensor | None = None):
        super().__init__()
        self.net = t.nn.Sequential(t.nn.Linear(d_in, 3, bias=False), t.nn.Sigmoid())
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, "n d_model"]:
        """Apply StandardScaler normalization if scaler parameters are available."""
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self.net(self._normalize(x)).squeeze(-1)

    def pred(self, x: Float[Tensor, "n d_model"]) -> Float[Tensor, " n"]:
        return self(x).round()

    @property
    def direction(self) -> Float[Tensor, " d_model"]:
        return self.net[0].weight.data[0]

    @staticmethod
    def from_data(
        acts: Float[Tensor, "n d_model"],
        labels: Float[Tensor, " n"],
        C: float = 0.1,
        device: str = "cpu",
    ) -> "LRProbe":
        """
        Train an LR probe using sklearn's LogisticRegression with StandardScaler normalization.

        Args:
            acts: Activation matrix [n_samples, d_model].
            labels: Binary labels (1=true, 0=false).
            C: Inverse regularization strength (lower = stronger regularization).
                Default 0.1 (reg_coeff=10) matches the deception-detection paper's cfg.yaml.
                The repo class default is reg_coeff=1000 (C=0.001), which is stronger.
            device: Device to place the resulting probe on.
        """
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        # Standardize features (zero mean, unit variance) before fitting, as in the paper
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # fit_intercept=False: the paper fits on normalized data so the intercept is redundant
        lr_model = LogisticRegression(C=C, random_state=42, fit_intercept=False, max_iter=1000, multi_class="multinomial")
        lr_model.fit(X_scaled, y)

        # Build probe with scaler parameters baked in
        scaler_mean = t.Tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.Tensor(scaler.scale_, dtype=t.float32)
        probe = LRProbe(acts.shape[-1], scaler_mean=scaler_mean, scaler_scale=scaler_scale).to(device)
        probe.net[0].weight.data[0] = t.Tensor(lr_model.coef_[0], dtype=t.float32).to(device)

        return probe

if __name__ == "__main__":
    sick_en_dataset_train, sick_en_dataloader_train = get_dataset_and_dataloader("en", "train")

    olmo_filepath = f"./{MODEL_FOLDER}/olmo_model"

    device = "cuda" if t.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(olmo_filepath, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(olmo_filepath, local_files_only=True).to(device)

    # print(hf_model.model.layers)

    # for layer_number in range(len(hf_model.model.layers)):
    #     generate_activations(layer_number, "olmo_model", amount_to_generate=10)

    generate_activations_all_layers("olmo_model", amount_to_generate=10)