# from sick_loader import get_dataset_and_dataloader
from typing import Literal

import torch as t

from activations import ActivationSaver, ActivationDataset
from probes import get_probe
from experiment_common_code import ExperimentResult, get_accuracy

experiment_number = 1

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def run_full_experiment(
    language: str,
    probing_task: str,
    probe_type: str,
    model_name: str,
    force_probe_creation: bool,
) -> ExperimentResult:
    print(
        f"Running experiment {experiment_number} instance. {language}, {probing_task}, {probe_type}, {model_name}"
    )
    olmo_activation_loader: ActivationSaver = ActivationSaver("olmo_model")

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    for layer_num in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_num}")

        activation_dataset_train = ActivationDataset(
            language, "train", layer_num, probing_task, model_name
        )

        probe = get_probe(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            activation_dataset_train,
            force_probe_creation,
            device,
        )

        train_acts, train_labels = (
            activation_dataset_train.activations,
            activation_dataset_train.labels,
        )

        # Train accuracy
        train_predictions: t.Tensor = probe.pred(train_acts)  # type: ignore
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

        test_predictions: t.Tensor = probe.pred(test_acts)  # type: ignore
        test_accuracy: float = get_accuracy(test_predictions, test_labels)

        test_accuracies.append(test_accuracy)
        print(f"Test accuracy: {test_accuracy}")

    experiment_result = ExperimentResult(
        experiment_number, language, probing_task, probe_type, model_name
    )
    experiment_result.train_accuracies = train_accuracies
    experiment_result.test_accuracies = test_accuracies

    print("=" * 30)
    return experiment_result


def run_experiment_1(
    languages: list[str],
    probing_tasks: list[str],
    probe_type: str,
    model_names: list[str],
    force_probe_creation: bool,
    save_results: bool = True,
) -> list[ExperimentResult]:
    experiment_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language in languages:
            for probing_task in probing_tasks:
                experiment_result: ExperimentResult = run_full_experiment(
                    language, probing_task, probe_type, model_name, force_probe_creation
                )
                experiment_results.append(experiment_result)

                # Save result if requested
                if save_results:
                    filepath: str = experiment_result.save_to_file()
                    print(f"Saved result to {filepath}")

    return experiment_results
