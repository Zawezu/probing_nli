# from sick_loader import get_dataset_and_dataloader
from typing import Literal

from activations_loader import ActivationLoader, ActivationDataset

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

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


class LRProbe(t.nn.Module):
    def __init__(self, d_in, scaler_mean, scaler_scale) -> None:
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

    def forward(self, x) -> t.Tensor:
        return self.net(self._normalize(x))

    def pred(self, x) -> t.Tensor:
        logits: t.Tensor = self.forward(x)
        return t.argmax(logits, dim=-1)


def create_probe_from_data(acts, labels, C=0.1, device="cpu") -> "LRProbe":
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

    scaler_mean: t.Tensor = t.tensor(scaler.mean_, dtype=t.float32)
    scaler_scale: t.Tensor = t.tensor(scaler.scale_, dtype=t.float32)

    probe: LRProbe = LRProbe(acts.shape[-1], scaler_mean, scaler_scale).to(device)

    probe.net[0].weight.data = t.tensor(lr_model.coef_, dtype=t.float32).to(device)

    return probe


def get_accuracy(preds: t.Tensor, labels: t.Tensor) -> float:
    return (preds == labels).float().mean().item()


# if __name__ == "__main__":
#     language = "en"
#     control = True

#     olmo_activation_loader = ActivationLoader("olmo_model")

#     for layer_num in range(olmo_activation_loader.get_number_of_layers()):
#         print(f"Probing at layer {layer_num}")


#         train_acts, train_labels = olmo_activation_loader.load_activations(
#             language, "train", layer_num, control=control
#         )

#         lr_probe = create_probe_from_data(train_acts, train_labels, device="cpu")

#         # Train accuracy
#         train_preds: t.Tensor = lr_probe.pred(train_acts)
#         # print(f"Train predictions:\n{train_preds}")
#         # print(f"Train labels:\n{train_labels}")
#         train_acc: float = get_accuracy(train_preds, train_labels)

#         print(f"Train accuracy: {train_acc}")

#         # Test accuracy
#         test_acts, test_labels = olmo_activation_loader.load_activations(
#             language, "test", layer_num, control=control
#         )
#         test_preds: t.Tensor = lr_probe.pred(test_acts)
#         test_acc: float = get_accuracy(test_preds, test_labels)
#         print(f"Test accuracy: {test_acc}")

if __name__ == "__main__":
    language = "en"
    control = True
    probing_task = "standard"
    model_name = "olmo_model"

    olmo_activation_loader: ActivationLoader = ActivationLoader("olmo_model")

    for layer_num in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_num}")

        activation_dataset_train = ActivationDataset(
            language, "train", layer_num, probing_task, model_name
        )
        train_acts, train_labels = (
            activation_dataset_train.activations,
            activation_dataset_train.labels,
        )

        lr_probe: LRProbe = create_probe_from_data(
            train_acts, train_labels, device="cpu"
        )

        # Train accuracy
        train_preds: t.Tensor = lr_probe.pred(train_acts)
        # print(f"Train predictions:\n{train_preds}")
        # print(f"Train labels:\n{train_labels}")
        train_acc: float = get_accuracy(train_preds, train_labels)

        print(f"Train accuracy: {train_acc}")

        # Test accuracy
        activation_dataset_test = ActivationDataset(
            language, "test", layer_num, probing_task, model_name
        )
        test_acts, test_labels = (
            activation_dataset_test.activations,
            activation_dataset_test.labels,
        )

        test_preds: t.Tensor = lr_probe.pred(test_acts)
        test_acc: float = get_accuracy(test_preds, test_labels)
        print(f"Test accuracy: {test_acc}")
