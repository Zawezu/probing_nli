# from sick_loader import get_dataset_and_dataloader
from typing import Literal
from torch import Tensor

import argparse
import torch as t
from sklearn.metrics import confusion_matrix

from activations import ActivationDataset, get_number_of_layers_from_file
from probes import get_probe
from experiment_common_code import ExperimentResult
from utils import LABEL_MAP, LANGUAGES, MODEL_NAMES

experiment_number = 1

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def run_full_experiment_1(
    language: str,
    probing_task: str,
    probe_type: str,
    model_name: str,
    force_probe_creation: bool,
    num_layers: int | None,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> ExperimentResult:
    """
    Perform a full run of experiment 1, iterating over all model layers.

    For each layer:
    - Gets (or trains) a probe on the train set of the given language
    - Generates predictions on the train and test sets
    - Appends confusion matrices and derived metrics to the ExperimentResult

    After all layers, appends cumulative and previous-layer overlapping index metrics.

    Returns:
        An ExperimentResult populated with per-layer metrics for every split.
    """
    print(
        f"Running experiment {experiment_number} instance. {language}, {probing_task}, {probe_type}, {model_name}"
    )
    # Create empty ExperimentResult that we will gradually fill with the results
    exp_result = ExperimentResult(
        experiment_number,
        language,
        probing_task,
        probe_type,
        model_name,
        zeroed_out_activation_dims=zeroed_out_activation_dims,
        zeroed_out_weight_dims=zeroed_out_weight_dims,
        force_original_labels=force_original_labels,
    )

    layers: list[int] = list(range(get_number_of_layers_from_file(model_name)))
    if num_layers is not None:
        layers = layers[:num_layers]

    # Run a sub-experiment in each layer
    for layer_num in layers:
        # print(f"Probing at layer {layer_num}")

        # Train data
        activation_dataset_train: ActivationDataset = ActivationDataset(
            language,
            "train",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
        )

        # Test data
        activation_dataset_test: ActivationDataset = ActivationDataset(
            language,
            "test",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
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
            activation_dataset_train=activation_dataset_train,
            force_probe_creation=force_probe_creation,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
            force_original_labels=force_original_labels,
        )

        # Get train predictions for generating the metrics
        train_preds: Tensor = probe.pred(activation_dataset_train.activations)  # type: ignore

        # Save confusion matrix of train predictions
        # Specify labels [0, 1, 2] from LABEL_MAP for consistent 3x3 matrix
        exp_result.append_metric(
            "train",
            "cm",
            confusion_matrix(
                train_labels, train_preds, labels=list(LABEL_MAP.values())
            ),
        )  # type: ignore

        # Get test predictions for generating the metrics
        test_preds: Tensor = probe.pred(activation_dataset_test.activations)  # type: ignore

        # print(f"First few test labels: {activation_dataset_test.labels[:20]}")
        # print(f"First few test preds:  {test_preds}")

        # Save confusion matrix of test predictions
        exp_result.append_metric(
            "test",
            "cm",
            confusion_matrix(test_labels, test_preds, labels=list(LABEL_MAP.values())),
        )  # type: ignore

        # Add indices per confusion matrix cell for both splits
        exp_result.add_idxs_per_cm_cell_metric("train", train_labels, train_preds)
        exp_result.add_idxs_per_cm_cell_metric("test", test_labels, test_preds)

        # Use the confusion matrix to get the rest of metrics for this layer
        exp_result.add_metrics_from_confusion_matrix()

    exp_result.add_overlapping_idxs_metric(True)
    exp_result.add_overlapping_idxs_metric(False)

    return exp_result


def run_experiment_1(
    languages: list[str],
    standard_task: str,
    control_task: str,
    probe_type: str,
    model_names: list[str],
    force_probe_creation: bool,
    save_results: bool = True,
    num_layers: int | None = None,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> list[ExperimentResult]:
    """
    Run experiment 1 for all combinations of model names and languages.

    For each (model, language) pair, probes are trained for both the standard and control
    tasks. Marginal metrics (standard minus control) are added to the standard result.
    Results are optionally saved to disk.

    Returns:
        List of ExperimentResult objects (alternating control and standard per pair).
    """
    exp_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language in languages:
            # Run full experiment on control task
            control_exp_result: ExperimentResult = run_full_experiment_1(
                language,
                control_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_layers,
                zeroed_out_activation_dims,
                zeroed_out_weight_dims,
                force_original_labels,
            )

            # Run full experiment on standard task
            standard_exp_result: ExperimentResult = run_full_experiment_1(
                language,
                standard_task,
                probe_type,
                model_name,
                force_probe_creation,
                num_layers,
                zeroed_out_activation_dims,
                zeroed_out_weight_dims,
                force_original_labels,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model names", nargs="*", default=MODEL_NAMES)
    parser.add_argument("-l", help="languages", nargs="*", default=LANGUAGES)
    parser.add_argument("-pt", help="probe type", default="lr")
    parser.add_argument(
        "-nl", help="number of layers (None = all)", type=int, default=None
    )
    parser.add_argument(
        "-f",
        help="force probe creation even if a saved probe exists",
        nargs="?",
        default="False",
        const="True",
    )
    parser.add_argument(
        "-sr",
        help="whether to save results",
        nargs="?",
        default="True",
        const="True",
    )
    parser.add_argument(
        "-zad",
        help="number of highest-magnitude activation dims to zero out before training (0 = disabled)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-zwd",
        help="number of highest-magnitude weight dims to zero out per class after loading (0 = disabled)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-fol",
        help="force original (non-Japanese) labels for Japanese data",
        nargs="?",
        default="False",
        const="True",
    )

    args: argparse.Namespace = parser.parse_args()
    print(args)

    model_names: list[str] = args.m
    languages: list[str] = args.l
    probe_type: str = args.pt
    num_layers: int | None = args.nl
    force_probe_creation: bool = args.f.lower() == "true"
    save_results: bool = args.sr.lower() == "true"
    zeroed_out_activation_dims: int = args.zad
    zeroed_out_weight_dims: int = args.zwd
    force_original_labels: bool = args.fol.lower() == "true"

    run_experiment_1(
        languages,
        "standard",
        "control",
        probe_type,
        model_names,
        force_probe_creation,
        num_layers=num_layers,
        save_results=save_results,
        zeroed_out_activation_dims=zeroed_out_activation_dims,
        zeroed_out_weight_dims=zeroed_out_weight_dims,
        force_original_labels=force_original_labels,
    )
