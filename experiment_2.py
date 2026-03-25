# from sick_loader import get_dataset_and_dataloader
from typing import Literal

import torch as t

from activations_loader import ActivationSaver, ActivationDataset
from probes import get_probe
from experiment_common_code import ExperimentResult, get_accuracy, plot_multiple_metrics
from common_constants import LANGUAGES
from itertools import permutations

experiment_number = 2

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def get_language_merged_string(language_a: str, language_b: str) -> str:
    """
    Merges the two languages into a single language string. This is because ExperimentResult expects a single language.
    """
    return f"{language_a}→{language_b}"


def run_full_experiment(
    language_a: str,
    language_b: str,
    probing_task: str,
    probe_type: str,
    model_name: str,
) -> ExperimentResult:
    print(
        f"Running experiment {experiment_number} instance. Using probes from {language_a} and activations from {language_b}. {probing_task}, {probe_type}, {model_name}"
    )
    olmo_activation_loader: ActivationSaver = ActivationSaver("olmo_model")

    train_accuracies: list[float] = []
    test_accuracies: list[float] = []

    for layer_num in range(olmo_activation_loader.get_number_of_layers()):
        print(f"Probing at layer {layer_num}")

        activation_dataset_train_a = ActivationDataset(
            language_a, "train", layer_num, probing_task, model_name
        )

        activation_dataset_train_b = ActivationDataset(
            language_b, "train", layer_num, probing_task, model_name
        )

        probe = get_probe(
            language_a,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            activation_dataset_train_a,
            device,
        )

        train_acts, train_labels = (
            activation_dataset_train_b.activations,
            activation_dataset_train_b.labels,
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
            language_b, "test", layer_num, probing_task, model_name
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
        experiment_number,
        get_language_merged_string(language_a, language_b),
        probing_task,
        probe_type,
        model_name,
    )
    experiment_result.train_accuracies = train_accuracies
    experiment_result.test_accuracies = test_accuracies

    print(f"{"-"*30}")
    return experiment_result


def run_experiment_2(
    language_pairs: list[tuple[str, str]],
    probing_tasks: list[str],
    probe_type: str,
    model_names: list[str],
    save_results: bool = True,
) -> list[ExperimentResult]:
    experiment_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language_a, language_b in language_pairs:
            for probing_task in probing_tasks:
                experiment_result: ExperimentResult = run_full_experiment(
                    language_a, language_b, probing_task, probe_type, model_name
                )
                experiment_results.append(experiment_result)

                # Save result if requested
                if save_results:
                    filepath: str = experiment_result.save_to_file()
                    print(f"Saved result to {filepath}")

    return experiment_results


if __name__ == "__main__":
    run_experiment: bool = False

    language_pairs: list[tuple[str, ...]] = list(permutations(LANGUAGES, 2))
    # language_pairs = [("es", "en")]
    probing_tasks: list[str] = ["standard", "control", "disjunct_control"]
    probe_type: str = "lr"
    model_names: list[str] = ["olmo_model"]

    print(
        f"language_pairs: {language_pairs}\nprobing_tasks: {probing_tasks}\nprobe_type: {probe_type}\nmodel_names: {model_names}\n{"="*30}"
    )

    # If run_experiment, run the experiment and save the results as files.
    if run_experiment:
        run_experiment_2(language_pairs, probing_tasks, probe_type, model_names)

    # Load the results from files
    experiment_results_per_language: dict[str, list[ExperimentResult]] = {}

    for model_name in model_names:
        for language_a, language_b in language_pairs:
            language_merged_string: str = get_language_merged_string(
                language_a, language_b
            )
            experiment_results_per_language[language_merged_string] = []
            for probing_task in probing_tasks:
                experiment_result: ExperimentResult = ExperimentResult.get_from_file(
                    experiment_number,
                    language_merged_string,
                    probing_task,
                    probe_type,
                    model_name,
                )
                experiment_results_per_language[language_merged_string].append(
                    experiment_result
                )

            # Make some plots
            plot_multiple_metrics(
                experiment_results_per_language[language_merged_string],
                ["test_accuracies", "train_accuracies"],
                f"Test and test accuracies by layer for languages {language_merged_string}",
                "layer",
                "accuracy",
                show=True,
                save=True,
            )
