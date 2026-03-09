# from sick_loader import get_dataset_and_dataloader
from activations_generator import ActivationLoader

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# # import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, FactoredMatrix


class LRProbe(t.nn.Module):
    def __init__(self, d_in, scaler_mean, scaler_scale):
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, 3, bias=False), t.nn.Softmax(dim=-1)
        )
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x):
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x):
        return self.net(self._normalize(x))

    def pred(self, x):
        logits = self.forward(x)
        return t.argmax(logits, dim=-1)

    @staticmethod
    def from_data(acts, labels, C=0.1, device="cpu"):
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(
            C=C,
            random_state=42,
            fit_intercept=False,
            max_iter=1000,
            # multi_class='auto'
        )
        lr_model.fit(X_scaled, y)

        scaler_mean = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale = t.tensor(scaler.scale_, dtype=t.float32)

        probe = LRProbe(acts.shape[-1], scaler_mean, scaler_scale).to(device)

        probe.net[0].weight.data = t.tensor(lr_model.coef_, dtype=t.float32).to(device)

        return probe


if __name__ == "__main__":
    device = "cuda" if t.cuda.is_available() else "cpu"

    language = "en"

    olmo_activation_loader = ActivationLoader("olmo_model")

    for layer_number in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_number}")

        train_acts, train_labels = olmo_activation_loader.load_activations(
            language, "train", layer_number
        )

        lr_probe = LRProbe.from_data(train_acts, train_labels, device="cpu")

        # Train accuracy
        train_preds = lr_probe.pred(train_acts)
        # print(f"Train predictions:\n{train_preds}")
        # print(f"Train labels:\n{train_labels}")
        train_acc = (train_preds == train_labels).float().mean().item()
        print(f"Train accuracy: {train_acc}")

        # Test accuracy
        test_acts, test_labels = olmo_activation_loader.load_activations(
            language, "test", layer_number
        )
        test_preds = lr_probe.pred(test_acts)
        test_acc = (test_preds == test_labels).float().mean().item()
        print(f"Test accuracy: {test_acc}")
