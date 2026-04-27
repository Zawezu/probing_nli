from typing import Literal
from torch import Tensor

import torch as t
from sklearn.metrics import confusion_matrix

from activations import ActivationDataset
from probes import LRProbe, get_probe, save_probe
from experiment_common_code import ExperimentResult
from utils import LABEL_MAP, get_language_merged_string, get_number_of_layers_from_file

experiment_number = 2

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def run_full_experiment(
    language_pair: tuple[str, str],
    probing_task: str,
    probe_type: str,
    model_name: str,
    force_probe_creation: bool,
    num_refits: int,
    num_layers: int | None,
    iterations_per_refit: int,
    save_refitted_probes: bool = True,
) -> list[ExperimentResult]:
    """
    Performs a full run of experiment 2
    - Gets a probe trained on the train set of language a (the first of the pair)
    - Generates predictions on the train and test sets of language b (the second of the pair)
    - Saves all the metrics into a ExperimentResult object
    """
    language_a, language_b = language_pair

    print(
        f"Running experiment {experiment_number} instance. {get_language_merged_string(language_pair)}, {probing_task}, {probe_type}, {model_name}"
    )

    # Create empty ExperimentResult that we will gradually fill with the results
    exp_results = [
        ExperimentResult(
            experiment_number,
            get_language_merged_string(language_pair),
            probing_task,
            probe_type,
            model_name,
            refit_num * iterations_per_refit,
        )
        for refit_num in range(num_refits)
    ]

    layers: list[int] = list(range(get_number_of_layers_from_file(model_name)))

    for layer_num in layers:
        activation_dataset_train_a: ActivationDataset = ActivationDataset(
            language_a, "train", layer_num, probing_task, model_name
        )

        # Train data (language b)
        activation_dataset_train_b: ActivationDataset = ActivationDataset(
            language_b, "train", layer_num, probing_task, model_name
        )

        # Test data (language b)
        activation_dataset_test_b: ActivationDataset = ActivationDataset(
            language_b, "test", layer_num, probing_task, model_name
        )

        # We get the appropiate probe for this layer
        probe: LRProbe = get_probe(
            language_a,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            activation_dataset_train_a,
            force_probe_creation,
            device,
        )

        # Load labels for this layer (not necessary in theory, but done just in case the layer activations somehow got misaligned)
        train_labels: Tensor = activation_dataset_train_a.labels
        test_labels: Tensor = activation_dataset_test_b.labels

        for refit_num in range(num_refits):
            # print(
            #     f"Testing probe refitted for {refit_num * iterations_per_refit} iterations"
            # )

            # Get the correct experiment result
            exp_result = exp_results[refit_num]

            if refit_num != 0:  # We skip refitting the first time
                # We refit the probe on the training set of the new language for a limited number of iterations
                probe.refit(activation_dataset_train_b, iterations_per_refit)

                if save_refitted_probes:
                    save_probe(
                        probe,
                        get_language_merged_string(language_pair),
                        layer_num,
                        probing_task,
                        probe_type,
                        model_name,
                        refit_num * iterations_per_refit,
                    )

            # Get train predictions (language b) for generating the metrics
            train_preds: Tensor = probe.pred(activation_dataset_train_b.activations)  # type: ignore

            # Save confusion matrix of train predictions
            # Specify labels [0, 1, 2] from LABEL_MAP for consistent 3x3 matrix
            exp_result.append_metric(
                "train",
                "cm",
                confusion_matrix(
                    train_labels, train_preds, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            # Get test predictions (language b) for generating the metrics
            test_preds: Tensor = probe.pred(activation_dataset_test_b.activations)  # type: ignore

            # print(f"First few test labels: {activation_dataset_test_b.labels[:20]}")
            # print(f"First few test preds:  {test_preds}")

            # Save confusion matrix of test predictions
            exp_result.append_metric(
                "test",
                "cm",
                confusion_matrix(
                    test_labels, test_preds, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            # Add indices per confusion matrix cell for both splits
            exp_result.add_idxs_per_cm_cell_metric("train", train_labels, train_preds)
            exp_result.add_idxs_per_cm_cell_metric("test", test_labels, test_preds)

            # Use the confusion matrix to get the rest of metrics for this layer
            exp_result.add_metrics_from_confusion_matrix()

    # After all the the experiments are done for all layers, add the overlapping indices metrics
    for exp_result in exp_results:
        exp_result.add_overlapping_idxs_metric(True)
        exp_result.add_overlapping_idxs_metric(False)

    return exp_results


def run_experiment_2(
    language_pairs: list[tuple[str, str]],
    standard_task: str,
    control_task: str,
    probe_type: str,
    model_names: list[str],
    num_refits: int = 0,
    iterations_per_refit: int = 50,
    force_probe_creation: bool = False,
    save_results: bool = True,
    num_layers: int | None = None,
) -> list[ExperimentResult]:
    exp_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language_pair in language_pairs:
            # Run full experiment on control task
            control_exp_results: list[ExperimentResult] = run_full_experiment(
                language_pair,
                control_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_refits,
                num_layers,
                iterations_per_refit,
            )

            # Run full experiment on standard task
            standard_exp_results: list[ExperimentResult] = run_full_experiment(
                language_pair,
                standard_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_refits,
                num_layers,
                iterations_per_refit,
            )

            for control_exp_result, standard_exp_result in zip(
                control_exp_results, standard_exp_results
            ):
                # Add the marginal metrics (so the difference between standard and control metrics) to the standard experiment result
                standard_exp_result.add_marginal_metrics(control_exp_result)

                exp_results.extend([control_exp_result, standard_exp_result])

    # Save results if requested
    if save_results:
        for exp_result in exp_results:
            filepath: str = exp_result.save_to_file()
            print(f"Saved result to {filepath}")

    return exp_results
