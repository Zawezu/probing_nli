# from sick_loader import get_dataset_and_dataloader
from typing import Literal

import torch as t

from activations_loader import ActivationSaver, ActivationDataset
import probes
from experiment_common_code import ExperimentResult, get_accuracy, plot_multiple_metrics
from common_constants import LANGUAGES

experiment_number = 1

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"

mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}


def run_full_experiment(
    language: str, probing_task: str, probe_type: str, model_name: str
) -> ExperimentResult:
    print(
        f"Running experiment 1 instance. {language}, {probing_task}, {probe_type}, {model_name}"
    )
    olmo_activation_loader: ActivationSaver = ActivationSaver("olmo_model")

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    for layer_num in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_num}")

        activation_dataset_train = ActivationDataset(
            language, "train", layer_num, probing_task, model_name
        )
        train_acts, train_labels = (
            activation_dataset_train.activations,
            activation_dataset_train.labels,
        )

        # Check if probe already exists
        if probes.probe_exists(
            language, layer_num, probing_task, probe_type, model_name
        ):
            print("Probe already exists. Loading from file...")
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
            probe = probes.load_probe(
                probe,
                language,
                layer_num,
                probing_task,
                probe_type,
                model_name,
                device="cpu",
            )
        else:
            # Create new probe
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
            # Save the probe
            probes.save_probe(
                probe, language, layer_num, probing_task, probe_type, model_name
            )

        # Train accuracy
        train_predictions: t.Tensor = probe.pred(train_acts)
        # print(f"Train predictions:\n{train_preds}")
        # print(f"Train labels:\n{train_labels}")
        train_accuracy: float = get_accuracy(train_predictions, train_labels)

        train_accuracies.append(train_accuracy)
        print(f"Train accuracy: {train_accuracy}")

        # Test accuracy
        activation_dataset_test = ActivationDataset(
            language, "test", layer_num, probing_task, model_name
        )
        test_acts, test_labels = (
            activation_dataset_test.activations,
            activation_dataset_test.labels,
        )

        test_predictions: t.Tensor = probe.pred(test_acts)
        test_accuracy: float = get_accuracy(test_predictions, test_labels)

        test_accuracies.append(test_accuracy)
        print(f"Test accuracy: {test_accuracy}")

    experiment_result = ExperimentResult(
        1, language, probing_task, probe_type, model_name
    )
    experiment_result.train_accuracies = train_accuracies
    experiment_result.test_accuracies = test_accuracies

    print(f"{"-"*30}")
    return experiment_result


def run_experiment_1(
    languages: list[str],
    probing_tasks: list[str],
    probe_type: str,
    model_names: list[str],
    save_results: bool = True,
) -> list[ExperimentResult]:
    experiment_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language in languages:
            for probing_task in probing_tasks:
                experiment_result: ExperimentResult = run_full_experiment(
                    language, probing_task, probe_type, model_name
                )
                experiment_results.append(experiment_result)

                # Save result if requested
                if save_results:
                    filepath: str = experiment_result.save_to_file()
                    print(f"Saved result to {filepath}")

    return experiment_results


if __name__ == "__main__":
    run_experiment: bool = True

    languages: list[str] = LANGUAGES
    probing_tasks: list[str] = ["standard", "control", "disjunct_control"]
    probe_type: str = "lr"
    model_names: list[str] = ["olmo_model"]

    # If run_experiment, run the experiment and save the results as files.
    if run_experiment:
        run_experiment_1(languages, probing_tasks, probe_type, model_names)

    # Load the results from files
    experiment_results_per_language: dict[str, list[ExperimentResult]] = {}

    for model_name in model_names:
        for language in languages:
            experiment_results_per_language[language] = []
            for probing_task in probing_tasks:
                experiment_result: ExperimentResult = ExperimentResult.get_from_file(
                    experiment_number, language, probing_task, probe_type, model_name
                )
                experiment_results_per_language[language].append(experiment_result)

            # Make some plots
            plot_multiple_metrics(
                experiment_results_per_language[language],
                ["test_accuracies", "train_accuracies"],
                f"Test and test accuracies by layer for language {language}",
                "layer",
                "accuracy",
                show=True,
                save=True,
            )
