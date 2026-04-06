# from sick_loader import get_dataset_and_dataloader
from typing import Literal
from torch import Tensor

import torch as t
from sklearn.metrics import confusion_matrix

from activations import ActivationSaver, ActivationDataset
from probes import get_probe
from experiment_common_code import ExperimentResult

experiment_number = 1

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def run_full_experiment(
    language: str,
    probing_task: str,
    probe_type: str,
    model_name: str,
    force_probe_creation: bool,
    num_layers: int | None,
) -> ExperimentResult:
    """
    Performs a full run of experiment 1
    - Gets a probe trained on the train set
    - Generates predictions on the train and test sets
    - Saves all the metrics into a ExperimentResult object
    """
    print(
        f"Running experiment {experiment_number} instance. {language}, {probing_task}, {probe_type}, {model_name}"
    )
    # Create empty ExperimentResult that we will gradually fill with the results
    exp_result = ExperimentResult(
        experiment_number, language, probing_task, probe_type, model_name
    )

    olmo_activation_loader: ActivationSaver = ActivationSaver("olmo_model")

    # If num_layers is specified, run experiment on those layers. Otherwise get the number of layers automatically
    if num_layers:
        layers: list[int] = list(range(num_layers))
    else:
        layers: list[int] = list(range(olmo_activation_loader.get_number_of_layers()))

    # Run a sub-experiment in each layer
    for layer_num in layers:
        print(f"Probing at layer {layer_num}")

        # Train data
        activation_dataset_train: ActivationDataset = ActivationDataset(
            language, "train", layer_num, probing_task, model_name
        )

        # Test data
        activation_dataset_test: ActivationDataset = ActivationDataset(
            language, "test", layer_num, probing_task, model_name
        )

        # Load labels for this layer (not necessary in theory, but done just in case the layer activations somehow got misaligned)
        train_labels: Tensor = activation_dataset_train.labels
        test_labels: Tensor = activation_dataset_test.labels

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

        # Get train predictions for generating the metrics
        train_preds: Tensor = probe.pred(activation_dataset_train.activations)  # type: ignore

        # Save confusion matrix of train predictions
        exp_result.append_metric(
            "train", "cm", confusion_matrix(train_labels, train_preds)
        )  # type: ignore

        # Get test predictions for generating the metrics
        test_preds: Tensor = probe.pred(activation_dataset_test.activations)  # type: ignore

        print(f"First few test labels: {activation_dataset_test.labels[:20]}")
        print(f"First few test preds:  {test_preds}")

        # Save confusion matrix of test predictions
        exp_result.append_metric(
            "test", "cm", confusion_matrix(test_labels, test_preds)
        )  # type: ignore

        # Use the confusion matrix to get the rest of metrics for this layer
        exp_result.add_metrics_from_confusion_matrix()

    return exp_result


def run_experiment_1(
    languages: list[str],
    standard_task: str,
    control_task: str,
    probe_type: str,
    model_names: list[str],
    force_probe_creation: bool,
    save_results: bool = True,
    num_layers: int
    | None = None,  # Attribute to force the model to only generate a number of layers
) -> list[ExperimentResult]:
    exp_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language in languages:
            # Run full experiment on control task
            control_exp_result: ExperimentResult = run_full_experiment(
                language,
                control_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_layers,
            )

            # Run full experiment on standard task
            standard_exp_result: ExperimentResult = run_full_experiment(
                language,
                standard_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_layers,
            )

            # Add the marginal metrics (so the difference between standard and control metrics) to the standard experiment result
            standard_exp_result.add_marginal_metrics(control_exp_result)

            exp_results.extend([control_exp_result, standard_exp_result])

    # Save results if requested
    if save_results:
        for exp_result in exp_results:
            filepath: str = exp_result.save_to_file()
            print(f"Saved result to {filepath}")

    return exp_results
