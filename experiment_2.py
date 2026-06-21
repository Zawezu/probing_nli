from typing import Literal
from torch import Tensor

import argparse
import torch as t
from sklearn.metrics import confusion_matrix

from activations import ActivationDataset
from probes import get_probe, save_probe, probe_exists, load_probe
from experiment_common_code import ExperimentResult
from utils import (
    LABEL_MAP,
    LANGUAGES,
    MODEL_NAMES,
    get_language_merged_string,
    get_language_pair_permutations,
    get_number_of_layers_from_file,
)

experiment_number = 2

device: Literal["cuda", "cpu"] = "cuda" if t.cuda.is_available() else "cpu"


def run_full_experiment_2(
    language_pair: tuple[str, str],
    probing_task: str,
    probe_type: str,
    model_name: str,
    force_probe_creation: bool,
    force_refit_probe_creation: bool,
    num_refits: int,
    num_layers: int | None,
    iterations_per_refit: int,
    save_refitted_probes: bool = True,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> list[ExperimentResult]:
    """
    Perform a full run of experiment 2, iterating over all model layers.

    For each layer:
    - Gets (or trains) a probe on the train set of language_a
    - Optionally refits the probe on the train set of language_b for a fixed number of
      iterations (repeated num_refits times, each adding iterations_per_refit steps)
    - Generates predictions on all four splits (train_a, test_a, train_b, test_b) for
      each refit level (including the 0-refit baseline)

    Returns:
        A list of ExperimentResult objects, one per refit level (length = num_refits).
        The first result corresponds to 0 extra iterations (original probe, no refit).
    """
    language_a, language_b = language_pair

    print(
        f"Running experiment {experiment_number} instance. {get_language_merged_string(language_pair)}, {probing_task}, {probe_type}, {model_name}{', original labels forced' if force_original_labels else ''}"
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
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
            force_original_labels=force_original_labels,
        )
        for refit_num in range(num_refits)
    ]

    layers: list[int] = list(range(get_number_of_layers_from_file(model_name)))
    if num_layers is not None:
        layers = layers[:num_layers]

    for layer_num in layers:
        # Train data (language a)
        activation_dataset_train_a: ActivationDataset = ActivationDataset(
            language_a,
            "train",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
        )

        # Train data (language b)
        activation_dataset_train_b: ActivationDataset = ActivationDataset(
            language_b,
            "train",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
        )

        # Test data (language a)
        activation_dataset_test_a: ActivationDataset = ActivationDataset(
            language_a,
            "test",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
        )

        # Test data (language b)
        activation_dataset_test_b: ActivationDataset = ActivationDataset(
            language_b,
            "test",
            layer_num,
            probing_task,
            model_name,
            force_original_labels=force_original_labels,
        )

        # We get the appropiate probe for this layer (pretrained on language a)
        probe = get_probe(
            language_a,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            activation_dataset_train=activation_dataset_train_a,
            force_probe_creation=force_probe_creation,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
            force_original_labels=force_original_labels,
        )

        # Load labels for this layer (not necessary in theory, but done just in case the layer activations somehow got misaligned)
        train_labels_a: Tensor = activation_dataset_train_a.labels
        train_labels_b: Tensor = activation_dataset_train_b.labels
        test_labels_a: Tensor = activation_dataset_test_a.labels
        test_labels_b: Tensor = activation_dataset_test_b.labels

        for refit_num in range(num_refits):
            extra_iters = refit_num * iterations_per_refit
            # print(
            #     f"Testing probe refitted for {refit_num * iterations_per_refit} iterations"
            # )

            # Get the correct experiment result
            exp_result = exp_results[refit_num]

            if refit_num != 0:  # We skip refitting the first time
                # Get the refitted probe from a saved file if force_refit_probe_creation is set to false and that file exists
                language_merged_string = get_language_merged_string(language_pair)
                if (not force_refit_probe_creation) and probe_exists(
                    language_merged_string,
                    layer_num,
                    probing_task,
                    probe_type,
                    model_name,
                    extra_iters,
                    zeroed_out_activation_dims,
                    force_original_labels=force_original_labels,
                ):
                    probe = load_probe(
                        language_merged_string,
                        layer_num,
                        probing_task,
                        probe_type,
                        model_name,
                        refit_num * iterations_per_refit,
                        zeroed_out_activation_dims=zeroed_out_activation_dims,
                        zeroed_out_weight_dims=zeroed_out_weight_dims,
                        force_original_labels=force_original_labels,
                    )
                else:
                    # Otherwise, refit and save the probe
                    # We refit the probe on the training set of language b for a limited number of iterations
                    probe.refit(activation_dataset_train_b, iterations_per_refit)

                    if save_refitted_probes:
                        save_probe(
                            probe,
                            language_merged_string,
                            layer_num,
                            probing_task,
                            probe_type,
                            model_name,
                            extra_iters,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            force_original_labels=force_original_labels,
                        )

            # Get all predictions for generating the metrics
            train_preds_a: Tensor = probe.pred(activation_dataset_train_a.activations)  # type: ignore
            test_preds_a: Tensor = probe.pred(activation_dataset_test_a.activations)  # type: ignore

            train_preds_b: Tensor = probe.pred(activation_dataset_train_b.activations)  # type: ignore
            test_preds_b: Tensor = probe.pred(activation_dataset_test_b.activations)  # type: ignore

            # Save confusion matrix of train predictions
            exp_result.append_metric(
                "train_a",
                "cm",
                confusion_matrix(
                    train_labels_a, train_preds_a, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            # Save confusion matrix of test predictions (on both languages)
            exp_result.append_metric(
                "test_a",
                "cm",
                confusion_matrix(
                    test_labels_a, test_preds_a, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            exp_result.append_metric(
                "train_b",
                "cm",
                confusion_matrix(
                    train_labels_b, train_preds_b, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            exp_result.append_metric(
                "test_b",
                "cm",
                confusion_matrix(
                    test_labels_b, test_preds_b, labels=list(LABEL_MAP.values())
                ),
            )  # type: ignore

            # Add indices per confusion matrix cell for both splits (on both languages)
            exp_result.add_idxs_per_cm_cell_metric(
                "train_a", train_labels_a, train_preds_a
            )
            exp_result.add_idxs_per_cm_cell_metric(
                "test_a", test_labels_a, test_preds_a
            )

            exp_result.add_idxs_per_cm_cell_metric(
                "train_b", train_labels_b, train_preds_b
            )
            exp_result.add_idxs_per_cm_cell_metric(
                "test_b", test_labels_b, test_preds_b
            )

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
    force_refit_probe_creation: bool = True,
    save_results: bool = True,
    num_layers: int | None = None,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> list[ExperimentResult]:
    """
    Run experiment 2 for all combinations of model names and language pairs.

    For each (model, language_pair) pair, runs cross-lingual probing on both the
    standard and control tasks across all refit levels. Marginal metrics (standard
    minus control) are added to each standard result. When force_original_labels is
    True, only pairs involving Japanese are processed. Results are optionally saved.

    Returns:
        List of ExperimentResult objects covering all refit levels, tasks, and pairs,
        in the order [control_refit0, standard_refit0, control_refit1, standard_refit1, ...].
    """
    exp_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language_pair in language_pairs:
            # For the Japanese original label ablation, we only need to run the experiments involving Japanese
            if force_original_labels and "jp" not in language_pair:
                print(f"Skipping {language_pair}")
                continue

            # Run full experiment on control task
            control_exp_results: list[ExperimentResult] = run_full_experiment_2(
                language_pair,
                control_task,
                probe_type,
                model_name,
                force_probe_creation,
                force_refit_probe_creation,
                num_refits,
                num_layers,
                iterations_per_refit,
                zeroed_out_activation_dims=zeroed_out_activation_dims,
                zeroed_out_weight_dims=zeroed_out_weight_dims,
                force_original_labels=force_original_labels,
            )

            # Run full experiment on standard task
            standard_exp_results: list[ExperimentResult] = run_full_experiment_2(
                language_pair,
                standard_task,
                probe_type,
                model_name,
                force_probe_creation,
                force_refit_probe_creation,
                num_refits,
                num_layers,
                iterations_per_refit,
                zeroed_out_activation_dims=zeroed_out_activation_dims,
                zeroed_out_weight_dims=zeroed_out_weight_dims,
                force_original_labels=force_original_labels,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model names", nargs="*", default=MODEL_NAMES)
    parser.add_argument(
        "-l",
        help="languages (language_pairs are derived from these)",
        nargs="*",
        default=LANGUAGES,
    )
    parser.add_argument("-pt", help="probe type", default="lr")
    parser.add_argument(
        "-nl", help="number of layers (None = all)", type=int, default=None
    )
    parser.add_argument("-nr", help="number of refits", type=int, default=2)
    parser.add_argument("-ir", help="iterations per refit", type=int, default=1000)
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
    language_pairs: list[tuple[str, str]] = get_language_pair_permutations(languages)
    probe_type: str = args.pt
    num_layers: int | None = args.nl
    num_refits: int = args.nr
    iterations_per_refit: int = args.ir
    force_probe_creation: bool = args.f.lower() == "true"
    save_results: bool = args.sr.lower() == "true"
    zeroed_out_activation_dims: int = args.zad
    zeroed_out_weight_dims: int = args.zwd
    force_original_labels: bool = args.fol.lower() == "true"

    run_experiment_2(
        language_pairs,
        "standard",
        "control",
        probe_type,
        model_names,
        num_refits=num_refits,
        iterations_per_refit=iterations_per_refit,
        force_probe_creation=force_probe_creation,
        num_layers=num_layers,
        save_results=save_results,
        zeroed_out_activation_dims=zeroed_out_activation_dims,
        zeroed_out_weight_dims=zeroed_out_weight_dims,
        force_original_labels=force_original_labels,
    )
