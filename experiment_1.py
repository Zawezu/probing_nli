# from sick_loader import get_dataset_and_dataloader
from typing import Literal

from activations_loader import ActivationSaver, ActivationDataset
import probes

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t


# # import transformer_lens
# import transformer_lens.utils as utils
# from transformer_lens.hook_points import (
#     HookPoint,
# )  # Hooking utilities
# from transformer_lens import HookedTransformer, FactoredMatrix

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"

mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}


def get_accuracy(preds: t.Tensor, labels: t.Tensor) -> float:
    return (preds == labels).float().mean().item()


def run_full_experiment(
    language: str, probing_task: str, probe_type: str, model_name: str
) -> None:
    olmo_activation_loader: ActivationSaver = ActivationSaver("olmo_model")

    for layer_num in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_num}")

        activation_dataset_train = ActivationDataset(
            language, "train", layer_num, probing_task, model_name
        )
        train_acts, train_labels = (
            activation_dataset_train.activations,
            activation_dataset_train.labels,
        )

        match probe_type:
            case "lr":
                probe = probes.LRProbe.create_from_data(
                    activation_dataset_train, device="cpu"
                )
            case "mlp":
                probe = probes.MLPProbe.create_from_data(
                    activation_dataset_train, 128, mlp_training_parameters, device
                )
            case _:
                raise KeyError(f"Probe {probe_type} does not exist")

        # Train accuracy
        train_preds: t.Tensor = probe.pred(train_acts)
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

        test_preds: t.Tensor = probe.pred(test_acts)
        test_acc: float = get_accuracy(test_preds, test_labels)
        print(f"Test accuracy: {test_acc}")


if __name__ == "__main__":
    language = "en"
    probing_task = "standard"
    probe_type = "lr"
    model_name = "olmo_model"

    run_full_experiment(language, probing_task, probe_type, model_name)
