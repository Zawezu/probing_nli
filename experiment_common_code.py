import math
import pickle
import re
from pathlib import Path
from typing import Any
from itertools import product
from collections import defaultdict
from matplotlib.pylab import ndarray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from torch import Tensor

from utils import (
    EXPERIMENT_RESULTS_FOLDER,
    PLOTS_FOLDER,
    LABEL_MAP,
    LANGUAGE_FULL_NAME_MAP,
    REVERSE_LABEL_MAP,
    get_verbose_version_of_language_string,
)
from sick import calculate_majority_class_baseline_f1

mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}


class ExperimentResult:
    def __init__(
        self,
        experiment_number: int,
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
        extra_iter_num: int = 0,
        zeroed_out_activation_dims: int = 0,
        zeroed_out_weight_dims: int = 0,
        force_original_labels: bool = False,
    ) -> None:
        """
        Initialize an ExperimentResult with experiment metadata and empty metrics storage.

        Sets self.splits based on experiment_number:
        - 1 or 3: ["train", "test"]
        - 2: ["train_a", "test_a", "train_b", "test_b"]
        """
        self.experiment_number: int = experiment_number
        self.language: str = language
        self.probing_task: str = probing_task
        self.probe_type: str = probe_type
        self.model_name: str = model_name
        self.extra_iter_num: int = extra_iter_num
        self.zeroed_out_activation_dims: int = zeroed_out_activation_dims
        self.zeroed_out_weight_dims: int = zeroed_out_weight_dims
        self.force_original_labels: bool = force_original_labels

        match experiment_number:
            case 1 | 3:
                self.splits: list[str] = ["train", "test"]
            case 2:
                self.splits = ["train_a", "test_a", "train_b", "test_b"]
            case _:
                raise ValueError(f"Incorrect experiment_number: {experiment_number}")

        self.metrics: dict[str, dict[str, Any]] = {split: {} for split in self.splits}

    def append_metric(self, split: str, metric: str, value: Any) -> None:
        """Append value to the metric list for the given split, creating the list if absent."""
        # print(f"Appending {value}")
        try:
            self.metrics[split][metric].append(value)
        except KeyError:
            self.metrics[split][metric] = [value]

    def get_metric(
        self,
        split: str,
        metric: str,
        layer_num: int | None = None,
        cls: str | None = None,
    ) -> Any:
        """
        Retrieve metric values for the given split.

        If layer_num is None, returns the full list across layers; otherwise returns the
        value at that index. If cls is provided, indexes into a per-class dict at that layer.
        """
        if layer_num is None:
            if cls is None:
                return self.metrics[split][metric]
            else:
                return [m.get(cls, 0) for m in self.metrics[split][metric]]
        else:
            if cls is None:
                return self.metrics[split][metric][layer_num]
            else:
                return self.metrics[split][metric][layer_num].get(cls, 0)

    def add_metrics_from_confusion_matrix(self, include_unknown=False) -> None:
        """
        Calculate and add overall and per-class metrics based on the most recently
        appended confusion matrix for each split.

        Adds the following metrics to self.metrics[split] (with an "_including_unk"
        suffix when include_unknown=True):
        - accuracy: overall accuracy (float)
        - precision, recall, f1: macro-averaged over supported classes (float)
        - per_class_precision, per_class_recall, per_class_f1: dict keyed by str label value

        Label keys depend on the confusion matrix shape after optional truncation:
        - 3x3 matrix (or 4x4 with include_unknown=False): keys are ["0", "1", "2"]
        - 4x4 matrix with include_unknown=True: keys are ["-1", "0", "1", "2"]

        Args:
            include_unknown: If True, retain the -1 "unknown" row/column in 4x4 matrices
                and append metrics with the "_including_unk" suffix. Defaults to False.
        """
        for split in self.splits:
            # Take the confusion matrix at the latest layer that has been recorded
            cm: ndarray = self.get_metric(split, "cm", -1)  # type: ignore

            if include_unknown:
                assert (
                    cm.shape[0] == 4
                ), f"include_unknown set to {include_unknown} even though there was no unknown in the confusion matrix"
                metric_suffix = "_including_unk"
            else:
                # Cut off the first row and column (unknown values), but only if the unknown row and column exist
                if cm.shape[0] == 4:
                    cm = cm[1:, 1:]
                metric_suffix: str = ""

            # Determine label values based on confusion matrix shape
            num_classes: int = cm.shape[0]
            if num_classes == 4:
                # 4x4 matrix: [-1, 0, 1, 2] for experiment 3
                label_values: list[int] = [-1, 0, 1, 2]
            elif num_classes == 3:
                # 3x3 matrix: [0, 1, 2] for experiments 1 and 2
                label_values: list[int] = [0, 1, 2]
            else:
                raise ValueError(
                    f"Unexpected confusion matrix shape: {cm.shape}. Expected 3x3 or 4x4."
                )

            # Accuracy: sum of diagonal / sum of all elements
            total_sum = np.sum(cm)
            accuracy = float(np.trace(cm) / total_sum if total_sum > 0 else 0.0)
            self.append_metric(split, f"accuracy{metric_suffix}", accuracy)

            # Per-class metrics
            per_class_recall: dict[str, float] = {}
            per_class_precision: dict[str, float] = {}
            per_class_f1: dict[str, float] = {}

            supported_class_ids: list[str] = []

            for class_idx in range(num_classes):
                # Use the actual label value as the key
                label_value = label_values[class_idx]
                class_id = str(label_value)

                # Recall (sensitivity/true positive rate): TP / (TP + FN)
                # TP is diagonal, FN is rest of row
                row_sum = np.sum(cm[class_idx, :])
                recall: float = float(
                    cm[class_idx, class_idx] / row_sum if row_sum > 0 else 0.0
                )
                per_class_recall[class_id] = recall

                # Precision: TP / (TP + FP)
                # TP is diagonal, FP is rest of column
                col_sum = np.sum(cm[:, class_idx])
                precision: float = float(
                    cm[class_idx, class_idx] / col_sum if col_sum > 0 else 0.0
                )
                per_class_precision[class_id] = precision

                # F1 score: 2 * (precision * recall) / (precision + recall)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1: float = 2 * (precision * recall) / (precision + recall)
                per_class_f1[class_id] = float(f1)

                # Only include classes that have actual support in the macro average
                # This makes it so that the unkown class does not affect the metrics when there are no unknown labels
                if row_sum > 0 or col_sum > 0:
                    supported_class_ids.append(class_id)

            # Store per-class metrics
            self.append_metric(
                split, f"per_class_precision{metric_suffix}", per_class_precision
            )
            self.append_metric(
                split, f"per_class_recall{metric_suffix}", per_class_recall
            )
            self.append_metric(split, f"per_class_f1{metric_suffix}", per_class_f1)

            # Overall metrics (macro average over supported classes only)
            self.append_metric(
                split,
                f"precision{metric_suffix}",
                float(np.mean([per_class_precision[c] for c in supported_class_ids])),
            )
            self.append_metric(
                split,
                f"recall{metric_suffix}",
                float(np.mean([per_class_recall[c] for c in supported_class_ids])),
            )
            self.append_metric(
                split,
                f"f1{metric_suffix}",
                float(np.mean([per_class_f1[c] for c in supported_class_ids])),
            )

    def add_marginal_metrics(self, control_exp_result: "ExperimentResult") -> None:
        """
        Add marginal (standard minus control) versions of all scalar and per-class metrics.

        For each layer, computes the per-layer difference between this result's metrics and
        the control result's, storing them with a "marginal_" prefix. Only valid for
        non-control results paired with 3x3 confusion matrices (experiments 1 and 2).

        Args:
            control_exp_result: An ExperimentResult whose probing_task contains "control".
        """
        assert (
            "control" not in self.probing_task
        ), "add_difference_with_control_metrics() cannot be used on a control experiment result"
        assert (
            "control" in control_exp_result.probing_task
        ), "control_exp_result must contain the results using some control task"

        # Helper function to get the marginal value for a particular metric
        def get_marginal_value(
            metric: str, layer_num: int, cls: None | str = None
        ) -> float:
            marginal_value: float = self.get_metric(
                split, metric, layer_num, cls
            ) - control_exp_result.get_metric(split, metric, layer_num, cls)  # type: ignore
            return marginal_value

        for split in self.splits:
            for layer_num in range(self.get_num_layers()):
                # Take the confusion matrix at the latest layer that has been recorded
                cm: ndarray = self.get_metric(split, "cm", layer_num)  # type: ignore

                if cm is None:
                    print(f"Warning: No confusion matrix found for {split} split")
                    continue

                # Accuracy
                self.append_metric(
                    split,
                    "marginal_accuracy",
                    get_marginal_value("accuracy", layer_num),
                )

                # Per-class metrics
                num_classes: int = cm.shape[0]
                marginal_per_class_precision: dict[str, float] = {}
                marginal_per_class_recall: dict[str, float] = {}
                marginal_per_class_f1: dict[str, float] = {}

                for class_idx in range(num_classes):
                    class_id = str(class_idx)
                    # Recall
                    marginal_recall: float = get_marginal_value(
                        "per_class_recall", layer_num, class_id
                    )
                    marginal_per_class_recall[class_id] = marginal_recall

                    # Precision
                    marginal_precision: float = get_marginal_value(
                        "per_class_precision", layer_num, class_id
                    )
                    marginal_per_class_precision[class_id] = marginal_precision

                    # F1 score
                    marginal_f1: float = get_marginal_value(
                        "per_class_f1", layer_num, class_id
                    )
                    marginal_per_class_f1[class_id] = marginal_f1

                # Store per-class metrics
                self.append_metric(
                    split, "marginal_per_class_precision", marginal_per_class_precision
                )
                self.append_metric(
                    split, "marginal_per_class_recall", marginal_per_class_recall
                )
                self.append_metric(
                    split, "marginal_per_class_f1", marginal_per_class_f1
                )

                # Overall metrics (macro average)
                self.append_metric(
                    split,
                    "marginal_precision",
                    float(np.mean(list(marginal_per_class_precision.values()))),
                )
                self.append_metric(
                    split,
                    "marginal_recall",
                    float(np.mean(list(marginal_per_class_recall.values()))),
                )
                self.append_metric(
                    split,
                    "marginal_f1",
                    float(np.mean(list(marginal_per_class_f1.values()))),
                )

    def __str__(self) -> str:
        return str(
            [
                self.experiment_number,
                self.language,
                self.probing_task,
                self.probe_type,
                self.model_name,
            ]
        )

    def add_idxs_per_cm_cell_metric(self, split, real_labels, preds) -> None:
        """
        Record which sample indices fell in each confusion matrix cell for the given split.

        Stores a dict keyed by "real:{true_label},pred:{pred_label}" mapping to a set of
        sample indices, appended as the "idxs_per_cm_cell" metric.
        """
        idxs_per_cm_cell: defaultdict[str, set[int]] = defaultdict(set)

        for idx in range(len(real_labels)):
            real_label: int = (
                int(real_labels[idx].item())
                if isinstance(real_labels, Tensor)
                else real_labels[idx]
            )
            pred_label: int = (
                int(preds[idx].item()) if isinstance(preds, Tensor) else preds[idx]
            )

            key: str = f"real:{real_label},pred:{pred_label}"
            idxs_per_cm_cell[key].add(idx)

        self.append_metric(split, "idxs_per_cm_cell", dict(idxs_per_cm_cell))

    def add_overlapping_idxs_metric(self, cummulative) -> None:
        """
        Track which samples maintain the same classification across consecutive layers.

        If cummulative=True, retains only the indices that had the same classification
        in ALL layers up to the current one. If cummulative=False, compares only against
        the immediately preceding layer. Appends "{prefix}_overlapping_idxs" and
        "{prefix}_overlapping_idx_amounts" metrics, where prefix is "cummulative" or
        "previous_layer". The first layer's entry is a placeholder with amounts=0.
        """
        for split in self.splits:
            idxs_per_cm_cell = self.get_metric(split, "idxs_per_cm_cell")

            cummulative_idxs: dict[str, set[int]] = idxs_per_cm_cell[0].copy()

            # Append the metrics for the first layer. Since it is the first, it doesn't really make sense to add an overlap metric,
            # so its values do not really mean anything.
            self.append_metric(
                split,
                f"{'cummulative' if cummulative else 'previous_layer'}_overlapping_idxs",
                idxs_per_cm_cell[0].copy(),
            )
            self.append_metric(
                split,
                f"{'cummulative' if cummulative else 'previous_layer'}_overlapping_idx_amounts",
                {key: 0 for key in idxs_per_cm_cell[0].keys()},
            )

            # Iterate over all the layers
            for i, idxs_per_cm_cell_for_this_layer in enumerate(idxs_per_cm_cell[1:]):
                if cummulative:
                    # If cummulative, the indices are the ones that had the same classifications at every single past layer
                    idxs_prev_layer: dict[str, set[int]] = cummulative_idxs
                else:
                    # If not cummulative, the indices are those of just the previous layer classifications
                    idxs_prev_layer = idxs_per_cm_cell[i].copy()

                overlapping_idxs: dict[str, set[int]] = {}
                overlapping_idx_amounts: dict[str, int] = {}
                # For each entry in idxs_for_this_cell, we only keep the common indices between the predictions of this layer and the previous one
                for key, idxs_for_this_cell in idxs_per_cm_cell_for_this_layer.items():
                    idxs_for_this_cell_prev_layer: set[int] = (
                        idxs_prev_layer[key] if key in idxs_prev_layer.keys() else set()
                    )
                    # print(f"idxs_for_this_cell:\n{idxs_for_this_cell}")
                    # print(f"idxs_prev_layer[key]:\n{idxs_for_this_cell_prev_layer}")
                    common_idxs: set = idxs_for_this_cell.intersection(
                        idxs_for_this_cell_prev_layer
                    )
                    # print(common_idxs)
                    overlapping_idxs[key] = common_idxs
                    overlapping_idx_amounts[key] = len(common_idxs)

                    cummulative_idxs[key] = common_idxs

                # Add the overlapping indices up to this layer
                self.append_metric(
                    split,
                    f"{'cummulative' if cummulative else 'previous_layer'}_overlapping_idxs",
                    overlapping_idxs.copy(),
                )
                self.append_metric(
                    split,
                    f"{'cummulative' if cummulative else 'previous_layer'}_overlapping_idx_amounts",
                    overlapping_idx_amounts,
                )

    def save_to_file(self) -> str:
        """
        Pickle the ExperimentResult to the default results directory.

        Returns:
            The path to the saved file.
        """
        save_dir = Path(
            f"{EXPERIMENT_RESULTS_FOLDER}/experiment_{self.experiment_number}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        filepath: Path = save_dir / self.get_filename(
            self.language,
            self.probing_task,
            self.probe_type,
            self.model_name,
            self.extra_iter_num,
            self.zeroed_out_activation_dims,
            self.zeroed_out_weight_dims,
            self.force_original_labels,
        )
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        return str(filepath)

    def get_name(self) -> str:
        """Return a space-separated string of the key experiment identifiers."""
        return (
            f"{self.language} {self.probing_task} {self.probe_type} {self.model_name}"
        )

    def get_num_layers(self) -> int:
        """Return the number of layers by checking the length of the stored accuracy list."""
        # Get the number of layers. We do it by getting the list of the accuracies over the layers.
        try:
            return len(self.get_metric("test", "accuracy"))
        except KeyError:
            return len(self.get_metric("test_a", "accuracy"))

    @staticmethod
    def get_filename(
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
        extra_iter_num: int,
        zeroed_out_activation_dims: int = 0,
        zeroed_out_weight_dims: int = 0,
        force_original_labels: bool = False,
    ) -> str:
        """Generate filename based on experiment parameters."""
        base = f"{language},{probing_task},{probe_type},{model_name}"
        suffixes = []
        if extra_iter_num:
            suffixes.append(f"{extra_iter_num}_extra_iters")
        if zeroed_out_activation_dims:
            suffixes.append(f"{zeroed_out_activation_dims}_zeroed_act_dims")
        if zeroed_out_weight_dims:
            suffixes.append(f"{zeroed_out_weight_dims}_zeroed_wt_dims")
        if force_original_labels and "jp" in language:
            suffixes.append("orig_labels")
        if suffixes:
            return f"{base},{','.join(suffixes)}.pkl"
        return f"{base}.pkl"

    @staticmethod
    def get_from_file(
        experiment_number: int,
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
        extra_iter_num: int = 0,
        zeroed_out_activation_dims: int = 0,
        zeroed_out_weight_dims: int = 0,
        force_original_labels: bool = False,
    ) -> "ExperimentResult":
        """
        Load an ExperimentResult from a pickle file.

        Returns:
            The loaded ExperimentResult object
        """
        filepath = f"{EXPERIMENT_RESULTS_FOLDER}/experiment_{experiment_number}/{ExperimentResult.get_filename(language, probing_task, probe_type, model_name, extra_iter_num, zeroed_out_activation_dims, zeroed_out_weight_dims, force_original_labels)}"
        with open(filepath, "rb") as f:
            return pickle.load(f)


def show_plots(
    experiment_number: int,
    model_names: list[str],
    splits: list[str],
    languages: list[str],
    probing_tasks: list[str],
    extra_iter_nums: list[int] = [0],
    probe_type: str = "lr",
    metric: str = "accuracy",
    separate_chars_within_plot: list[str] = ["language", "probing_task"],
    y_axis_range: tuple[float, float] | None = None,
    show: bool = True,
    save: bool = False,
    filename: str = "",
    legend_position="upper left",
    zeroed_out_activation_dims_list: list[int] = [0],
    zeroed_out_weight_dims_list: list[int] = [0],
    horizontal_line: str | int = "",
    subplot_titles: list[str] | None = None,
    as_bars: bool = True,
) -> None:
    """
    Load ExperimentResult files and generate plots grouped by characteristic combinations.

    Iterates over all combinations of the provided parameters. Characteristics listed in
    `separate_chars_within_plot` create multiple lines/bars within a single subplot; all
    other characteristics produce separate subplots. Delegates rendering to
    `plot_metrics_by_group_as_bars` when as_bars=True, otherwise `plot_metrics_by_group`.

    Args:
        experiment_number: Which experiment's results to load.
        model_names: Model names to include.
        splits: Data splits to plot (e.g. ["train", "test"]).
        languages: Language keys; may include "_original_labels" suffix or "→"-separated pairs.
        probing_tasks: Probing tasks to include (e.g. ["standard", "control"]).
        extra_iter_nums: Extra refit iteration counts to include (experiment 2 only).
        probe_type: Probe type used when loading results.
        metric: Metric key to retrieve from ExperimentResult.
        separate_chars_within_plot: Characteristics that vary within a single subplot.
            Valid values: "model_name", "split", "class_name", "language" (or "language_a"
            / "language_b"), "probing_task", "extra_iter_num",
            "zeroed_out_activation_dims", "zeroed_out_weight_dims".
        y_axis_range: Fixed (min, max) y-axis range; auto-computed if None.
        show: Whether to call plt.show().
        save: Whether to save the figure to disk (requires filename).
        filename: Output filename (required when save=True).
        legend_position: Legend location string passed to matplotlib.
        zeroed_out_activation_dims_list: Ablation values for zeroed activation dims.
        zeroed_out_weight_dims_list: Ablation values for zeroed weight dims.
        horizontal_line: Draw a horizontal reference line. Pass a numeric value,
            "baseline_f1", or "control_average".
        subplot_titles: Override auto-generated subplot titles; pass [] to suppress all.
        as_bars: If True (default) render grouped bar charts; otherwise render line plots.
    """
    if (save and not filename) or (filename and not save):
        raise KeyError("save and filename should both be specified or neither")

    # extended_class_names_list: list[str] = class_names_list + [
    #     "all"
    # ]  # This last element represents any label.
    extended_class_names: dict[str, str] = {str(v): k for k, v in LABEL_MAP.items()}
    extended_class_names[""] = "all"

    if "overlapping_idx_amounts" in metric:
        class_ids: list[str] = []
        for r in LABEL_MAP.values():
            for p in LABEL_MAP.values():
                class_id: str = f"real:{r},pred:{p}"
                class_ids.append(class_id)
                extended_class_names[class_id] = (
                    f"real:{REVERSE_LABEL_MAP[r]},pred:{REVERSE_LABEL_MAP[p]}"
                )
    elif "per_class" in metric:
        class_ids = [str(label) for label in LABEL_MAP.values()]
    else:
        class_ids = [""]

    # Validate that specified characteristics are valid
    all_valid_characteristics: set[str] = {
        "model_name",
        "split",
        "class_name",
        "language",
        "language_a",
        "language_b",
        "probing_task",
        "extra_iter_num",
        "zeroed_out_activation_dims",
        "zeroed_out_weight_dims",
    }

    for char in separate_chars_within_plot:
        if char not in all_valid_characteristics:
            raise ValueError(
                f"Invalid characteristic: {char}. Must be one of {sorted(all_valid_characteristics)}"
            )

    # At most one language-related characteristic is allowed at a time
    language_chars_in_separate = [
        c
        for c in separate_chars_within_plot
        if c in {"language", "language_a", "language_b"}
    ]
    if len(language_chars_in_separate) > 1:
        raise ValueError(
            f"separate_chars_within_plot can contain at most one of 'language', 'language_a', "
            f"'language_b', got: {language_chars_in_separate}"
        )

    # When language_a or language_b is used, language pairs are split into components
    use_language_split: bool = any(
        c in separate_chars_within_plot for c in ("language_a", "language_b")
    )

    if use_language_split:
        valid_characteristics: list[str] = [
            "model_name",
            "split",
            "class_name",
            "language_a",
            "language_b",
            "probing_task",
            "extra_iter_num",
            "zeroed_out_activation_dims",
            "zeroed_out_weight_dims",
        ]
    else:
        valid_characteristics = [
            "model_name",
            "split",
            "class_name",
            "language",
            "probing_task",
            "extra_iter_num",
            "zeroed_out_activation_dims",
            "zeroed_out_weight_dims",
        ]

    separate_chars_outside_plot: list[str] = [
        c for c in valid_characteristics if c not in separate_chars_within_plot
    ]

    # Generate all combinations of all characteristics
    all_combinations: list[dict[str, Any]] = []
    for (
        model_name,
        split,
        class_id,
        language_key,
        probing_task,
        extra_iter_num,
        zeroed_out_activation_dims,
        zeroed_out_weight_dims,
    ) in product(
        model_names,
        splits,
        class_ids,
        languages,
        probing_tasks,
        extra_iter_nums,
        zeroed_out_activation_dims_list,
        zeroed_out_weight_dims_list,
    ):
        if class_id in extended_class_names.keys():
            class_name: str = extended_class_names[class_id]
        else:
            class_name = class_id

        actual_language, force_original_labels = _resolve_language_key(language_key)

        combo: dict[str, Any] = {
            "model_name": model_name,
            "split": split,
            "class_id": class_id,
            "class_name": class_name,
            "language": language_key,
            "actual_language": actual_language,
            "force_original_labels": force_original_labels,
            "probing_task": probing_task,
            "extra_iter_num": extra_iter_num,
            "zeroed_out_activation_dims": zeroed_out_activation_dims,
            "zeroed_out_weight_dims": zeroed_out_weight_dims,
        }

        if use_language_split:
            lang_parts = language_key.split("→", 1)
            combo["language_a"] = lang_parts[0]
            combo["language_b"] = lang_parts[1] if len(lang_parts) > 1 else ""

        all_combinations.append(combo)

    # Group combinations by the specified characteristics to define each plot
    plots_dict: dict[tuple, list[dict]] = defaultdict(list)

    for combo in all_combinations:
        # Create key from specified characteristics
        key = tuple(combo[char] for char in separate_chars_outside_plot)

        # Load the experiment result
        exp_result: ExperimentResult = ExperimentResult.get_from_file(
            experiment_number,
            combo["actual_language"],
            combo["probing_task"],
            probe_type,
            combo["model_name"],
            combo["extra_iter_num"],
            combo["zeroed_out_activation_dims"],
            combo["zeroed_out_weight_dims"],
            combo["force_original_labels"],
        )

        # Create plot request
        line_request: dict[str, str | ExperimentResult] = {
            "exp_result": exp_result,
            "language_key": combo["language"],
            "split": combo["split"],
            "class_id": combo["class_id"],
            "class_name": combo["class_name"],
        }

        # print(
        #     f"Created line request:\n{[f'{key}: {str(value)}' for key, value in line_request.items()]}"
        # )

        plots_dict[key].append(line_request)

    # Convert grouped plots to list format
    plots_to_make: list[list[dict]] = list(plots_dict.values())

    if as_bars:
        plot_metrics_by_group_as_bars(
            plots_to_make,
            metric,
            show,
            save,
            filename,
            y_axis_range,
            legend_position=legend_position,
            horizontal_line=horizontal_line,
            subplot_titles=subplot_titles,
        )
    else:
        plot_metrics_by_group(
            plots_to_make,
            metric,
            show,
            save,
            filename,
            y_axis_range,
            legend_position=legend_position,
            horizontal_line=horizontal_line,
            subplot_titles=subplot_titles,
        )


_ORIG_LABELS_SUFFIX = "_original_labels"


def _resolve_language_key(lang_key: str) -> tuple[str, bool]:
    """
    Resolve a language key that may contain '_original_labels' on any component.

    Returns (actual_language_string, force_original_labels).

    Examples:
        "jp_original_labels"    → ("jp", True)
        "jp"                    → ("jp", False)
        "jp_original_labels→en" → ("jp→en", True)
        "en→jp_original_labels" → ("en→jp", True)
    """
    force = False
    if "→" in lang_key:
        parts = lang_key.split("→", 1)
        resolved = []
        for part in parts:
            if part.endswith(_ORIG_LABELS_SUFFIX):
                resolved.append(part[: -len(_ORIG_LABELS_SUFFIX)])
                force = True
            else:
                resolved.append(part)
        return "→".join(resolved), force
    else:
        if lang_key.endswith(_ORIG_LABELS_SUFFIX):
            return lang_key[: -len(_ORIG_LABELS_SUFFIX)], True
        return lang_key, False


_CROSS_LINGUAL_RE = re.compile(r"trained in (.*?), tested in (.*)")
_TEST_Z_RE = re.compile(r"test (.+)")

OKABE_ITO_PALETTE: list[str] = [
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
]

LANGUAGE_COLOURS = {
    "en": ["#D81B1B", "#B76060"],
    "es": ["#E5BC1E", "#C5B268"],
    "nl": ["#0752FF", "#6683C3"],
    "jp": ["#00AFCA", "#6CAFB9"],
}


def plot_metrics_by_group(
    plots_to_make: list[list[dict]],
    metric: str,
    show: bool,
    save: bool,
    filename: str,
    y_axis_range: tuple[float, float] | None,
    legend_position,
    xlabel: str = "Layer",
    scale: int = 1,
    sort_lines: bool = False,
    horizontal_line: str | int = "",
    subplot_titles: list[str] | None = None,
) -> None:
    """
    Plot metrics grouped by experiment groups as line plots over layers.

    Each group (inner list) becomes a separate subplot; each element within a group
    becomes one line. Titles and legends are automatically generated based on common
    and varying attributes (language, probing task, probe type, model name, split, metric).

    Args:
        plots_to_make: Output of show_plots' internal grouping — list of groups, where
            each group is a list of line-request dicts (keys: exp_result, split,
            class_id, class_name, language_key).
        metric: Metric key to retrieve from ExperimentResult.
        show: Whether to call plt.show().
        save: Whether to save the figure to disk (requires filename).
        filename: Output filename (required when save=True).
        y_axis_range: Fixed (min, max) y-axis range; auto-computed if None.
        legend_position: Legend location string passed to matplotlib.
        xlabel: Label for the x-axis (default "Layer").
        scale: Scaling factor applied to the default figure size.
        sort_lines: If True, sort lines within each subplot by descending average value.
        horizontal_line: Draw a reference line — numeric value, "baseline_f1", or
            "control_average" (renders the control series as a dashed horizontal average).
        subplot_titles: Override auto-generated subplot titles; pass [] to suppress all.
    """
    num_plots: int = len(plots_to_make)

    if (
        subplot_titles is not None
        and subplot_titles
        and len(subplot_titles) != num_plots
    ):
        print(
            f"Warning: subplot_titles has {len(subplot_titles)} entries but there are {num_plots} subplots."
        )

    # Calculate grid shape to approximate a square, but place exactly 3 plots
    # in a single row.
    if num_plots == 3:
        cols: int = 3
        rows: int = 1
    else:
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)

    single_subplot_size: tuple[int, int] = (7 * scale, 4 * scale)
    figsize: tuple[int, int] = (
        single_subplot_size[0] * cols,
        single_subplot_size[1] * rows,
    )
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, squeeze=False)

    # Flatten all experiments to calculate y_axis_range if needed
    if y_axis_range is None:
        all_values: list[float] = []
        y_axis_margin = 0.1

        for group_of_line_requests in plots_to_make:
            for line_request in group_of_line_requests:
                exp_result: ExperimentResult = line_request["exp_result"]
                split: str = line_request["split"]
                class_id: str = line_request["class_id"]
                class_name: str = line_request["class_name"]
                # results_for_determining_axis: list[float | list[float]] = exp_result.metrics[split][metric]
                if class_name != "all":
                    # print(exp_result.metrics[split][metric])
                    results_for_determining_axis: list[float] = exp_result.get_metric(
                        split, metric, layer_num=None, cls=class_id
                    )
                else:
                    results_for_determining_axis = exp_result.get_metric(
                        split, metric, layer_num=None, cls=None
                    )

                all_values.extend(results_for_determining_axis)
                # if isinstance(results_for_determining_axis[0], float):
                #     all_values.extend(results_for_determining_axis)  # type: ignore
                # else:
                #     for results_list in results_for_determining_axis:
                #         all_values.extend(results_list)  # type: ignore

        y_axis_range = (
            min(all_values) - y_axis_margin,
            max(all_values) + y_axis_margin,
        )

    # Define the natural order of attributes for consistent presentation
    attribute_sequence: list[str] = [
        "language",
        "probing_task",
        "label",
        "probe_type",
        "model_name",
        "extra_iter_num",
        "zeroed_out_activation_dims",
        "zeroed_out_weight_dims",
        "split",
        "metric",
    ]

    # Plot each group
    for plot_idx, group_of_line_requests in enumerate(plots_to_make):
        row_idx: int = plot_idx // cols
        col_idx: int = plot_idx % cols
        ax = axs[row_idx, col_idx]

        # Collect all attributes for each line in this plot
        attrs_per_line: list[dict[str, str]] = []

        for line_request in group_of_line_requests:
            exp_result: ExperimentResult = line_request["exp_result"]
            split: str = line_request["split"]
            class_name: str = line_request["class_name"]
            display_language: str = line_request.get(
                "language_key", exp_result.language
            )
            attrs: dict[str, str] = {
                "language": get_verbose_version_of_language_string(display_language),
                "probing_task": exp_result.probing_task,
                "probe_type": exp_result.probe_type,
                "model_name": exp_result.model_name,
                "extra_iter_num": str(exp_result.extra_iter_num),
                "zeroed_out_activation_dims": str(
                    getattr(exp_result, "zeroed_out_activation_dims", 0)
                ),
                "zeroed_out_weight_dims": str(
                    getattr(exp_result, "zeroed_out_weight_dims", 0)
                ),
                "split": split,
                "metric": metric,
                "label": class_name,
            }

            # Replace all underscores by spaces to create a nicer plot
            for key in attrs.keys():
                attrs[key] = str(attrs[key]).replace("_", " ")
            attrs_per_line.append(attrs)

        # Determine which attributes are common and which vary across all lines
        common_attrs: dict[str, str] = {}
        varying_attrs: set[str] = set()

        for attr_key in attribute_sequence:
            if not (
                attr_key == "label"
                and "per_class" not in metric
                and "overlapping_idx_amounts" not in metric
            ):
                values: list[str] = [attrs[attr_key] for attrs in attrs_per_line]
                if len(set(values)) == 1:
                    # This attribute is common across all lines
                    common_attrs[attr_key] = values[0]
                else:
                    # This attribute varies across lines
                    varying_attrs.add(attr_key)

        # print(f"Common attributes={common_attrs}")
        # print(f"Varying attributes={varying_attrs}")
        # Generate title: "[common aspects] for different [varying aspect names]"
        common_parts: list[str] = [
            common_attrs[key] for key in attribute_sequence if key in common_attrs
        ]
        different_parts: list[str] = [
            str(key).replace("_", " ") + "s"
            for key in attribute_sequence
            if key in varying_attrs
        ]

        if different_parts:
            title: str = f"Results of {', '.join(common_parts)} for different {', '.join(different_parts)} over the {xlabel.lower()}s"
        else:
            title = f"Results of {', '.join(common_parts)} over the {xlabel}s"

        if subplot_titles is None or (
            subplot_titles and plot_idx >= len(subplot_titles)
        ):
            ax.set_title(title, fontsize=10)
        elif subplot_titles:
            ax.set_title(subplot_titles[plot_idx], fontsize=10)
        # else subplot_titles == [] → no title

        # Prepare line data with average values for sorting
        lines_to_plot: list[dict] = []
        seen_test_a_base: bool = False
        for i, line_request in enumerate(group_of_line_requests):
            exp_result: ExperimentResult = line_request["exp_result"]
            split: str = line_request["split"]
            is_test_a_base: bool = split == "test_a" and exp_result.extra_iter_num == 0
            if is_test_a_base:
                if seen_test_a_base:
                    continue
                seen_test_a_base = True
            class_id: str = line_request["class_id"]
            class_name: str = line_request["class_name"]
            results: list[float | dict[str, float]] = exp_result.metrics[split][metric]
            layers: list[int] = list(range(len(results)))

            # Generate legend label with only varying attributes
            legend_parts: list[str] = [
                attrs_per_line[i][key]
                for key in attribute_sequence
                if key in varying_attrs
            ]

            cl_idx = next(
                (
                    idx
                    for idx, p in enumerate(legend_parts)
                    if _CROSS_LINGUAL_RE.fullmatch(p)
                ),
                None,
            )
            tz_idx = next(
                (idx for idx, p in enumerate(legend_parts) if _TEST_Z_RE.fullmatch(p)),
                None,
            )

            legend_label: str = " ".join(legend_parts)

            if cl_idx is not None and tz_idx is not None:
                cl = _CROSS_LINGUAL_RE.fullmatch(legend_parts[cl_idx])
                tz = _TEST_Z_RE.fullmatch(legend_parts[tz_idx])
                X, Y, raw_Z = cl.group(1), cl.group(2), tz.group(1)  # type: ignore
                if raw_Z == "a":
                    other_parts = [
                        p
                        for j, p in enumerate(legend_parts)
                        if j not in {cl_idx, tz_idx}
                    ]
                    special = f"trained in {X}, re-trained in {Y}, and tested in {X}"
                    legend_label = " ".join([*other_parts, special]).strip()
                elif raw_Z == "b":
                    other_parts = [
                        p
                        for j, p in enumerate(legend_parts)
                        if j not in {cl_idx, tz_idx}
                    ]
                    special = f"trained in {X}, re-trained and tested in {Y}"
                    legend_label = " ".join([*other_parts, special]).strip()
            elif (
                cl_idx is not None
                and len(legend_parts) == 1
                and exp_result.extra_iter_num > 0
            ):
                cl = _CROSS_LINGUAL_RE.fullmatch(legend_parts[cl_idx])
                X, Y = cl.group(1), cl.group(2)  # type: ignore
                legend_label = f"trained in {X}, re-trained and tested in {Y}"

            if is_test_a_base:
                lang_a = exp_result.language.split("→")[0]
                legend_label = f"trained and tested in {LANGUAGE_FULL_NAME_MAP[lang_a]}"

            if "per_class" in metric or "overlapping_idx_amounts" in metric:
                # If we are dealing with a per-class metric, we add a plot for the requested label id
                results_for_this_label: list[float] = [
                    r.get(class_id, 0)  # type: ignore
                    for r in results  # type: ignore
                ]  # type: ignore
                average_value: float = float(np.mean(results_for_this_label))

                # If plotting overlapping_idx_amounts, it doesn't make sense to plot the first layer, so it is removed
                if "overlapping_idx_amounts" in metric:
                    layers.pop(0)
                    results_for_this_label.pop(0)

                lines_to_plot.append(
                    {
                        "layers": layers,
                        "results": results_for_this_label,
                        "legend_label": legend_label,
                        "average_value": average_value,
                        "is_control": "control" in exp_result.probing_task,
                        "is_test_a_base": is_test_a_base,
                        "probing_task": exp_result.probing_task,
                        "language": exp_result.language,
                    }
                )
            else:
                # If we are dealing with a non-per-class metric, we simply plot the results
                average_value: float = float(np.mean(results))  # type: ignore
                lines_to_plot.append(
                    {
                        "layers": layers,
                        "results": results,
                        "legend_label": legend_label,
                        "average_value": average_value,
                        "is_control": "control" in exp_result.probing_task,
                        "is_test_a_base": is_test_a_base,
                        "probing_task": exp_result.probing_task,
                        "language": exp_result.language,
                    }
                )

        if sort_lines:
            lines_to_plot.sort(key=lambda x: x["average_value"], reverse=True)

        lang_color_counts: dict[str, int] = defaultdict(int)
        okabe_ito_idx: int = 0
        for i, line_data in enumerate(lines_to_plot):
            raw_lang: str = line_data["language"]
            lang_parts = raw_lang.split("→")
            lang_key: str = (
                lang_parts[0] if line_data["is_test_a_base"] else lang_parts[-1]
            )
            lang_count: int = lang_color_counts[lang_key]
            lang_color_counts[lang_key] += 1

            if lang_key in LANGUAGE_COLOURS:
                if lang_count == 0:
                    colour = LANGUAGE_COLOURS[lang_key][0]
                elif lang_count == 1:
                    colour = mcolors.to_rgb(LANGUAGE_COLOURS[lang_key][1])
                else:
                    colour = OKABE_ITO_PALETTE[okabe_ito_idx % len(OKABE_ITO_PALETTE)]
                    okabe_ito_idx += 1
            else:
                colour = OKABE_ITO_PALETTE[okabe_ito_idx % len(OKABE_ITO_PALETTE)]
                okabe_ito_idx += 1

            line_data["colour"] = colour
            if horizontal_line == "control_average" and line_data["is_control"]:
                ax.axhline(
                    y=line_data["average_value"],
                    linestyle="--",
                    alpha=0.7,
                    color=colour,
                    label=f"avg {line_data['legend_label']}",
                )
            else:
                ax.plot(
                    line_data["layers"],
                    line_data["results"],
                    # marker="o",
                    label=line_data["legend_label"],
                    color=colour,
                    linewidth=3,
                )

        if horizontal_line == "baseline_f1":
            if "f1" not in metric:
                print(
                    f"Warning: horizontal_line='baseline_f1' was specified but metric is '{metric}', not an F1 metric."
                )
            baseline_values: list[float] = [
                calculate_majority_class_baseline_f1(
                    probing_task=line_data["probing_task"],
                    language="en",  # the label proportions are very similar in Japanese and other languages, so we just use English
                )
                for line_data in lines_to_plot
            ]
            if max(baseline_values) - min(baseline_values) <= 0.05:
                ax.axhline(
                    y=float(np.mean(baseline_values)),
                    linestyle="--",
                    alpha=0.7,
                    color="gray",
                    label="baseline",
                )
            else:
                for i, (line_data, baseline) in enumerate(
                    zip(lines_to_plot, baseline_values)
                ):
                    colour = line_data["colour"]
                    label = f"majority baseline {line_data['legend_label']}".strip()
                    ax.axhline(
                        y=baseline, linestyle="--", alpha=0.7, color=colour, label=label
                    )
        elif horizontal_line != "" and horizontal_line != "control_average":
            ax.axhline(y=float(horizontal_line), linestyle="--", alpha=0.7)

        ax.set_xlabel(xlabel)
        if metric.startswith("marginal_"):
            ylabel = f"selectivity ({metric[len('marginal_'):].replace('_', ' ')})"
        else:
            ylabel = metric.replace("_", " ")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_position, fontsize=7)
        ax.set_ylim(y_axis_range)

    # Hide unused subplots
    for plot_idx in range(num_plots, rows * cols):
        row_idx = plot_idx // cols
        col_idx = plot_idx % cols
        axs[row_idx, col_idx].set_visible(False)

    fig.tight_layout()

    if save:
        save_dir = Path(PLOTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename.endswith(".png"):
            filepath: Path = save_dir / filename
        else:
            filepath: Path = save_dir / f"{filename}.png"

        fig.savefig(filepath)  # type: ignore
        print(f"Plot saved to {filepath}")

    if show:
        plt.show()


def plot_metrics_by_group_as_bars(
    plots_to_make: list[list[dict]],
    metric: str,
    show: bool = True,
    save: bool = False,
    filename: str = "",
    y_axis_range: tuple[float, float] | None = None,
    legend_position: str = "upper left",
    bins: list[str] | None = None,
    xlabel: str = "Layer group",
    scale: int = 1,
    sort_lines: bool = False,
    horizontal_line: str | int = "",
    subplot_titles: list[str] | None = None,
) -> None:
    """
    Like plot_metrics_by_group but creates grouped bar plots with layer-averaged values.

    Layers are split into len(bins) equally-sized bins (extra layers go to earlier bins),
    and values within each bin are averaged to produce one bar. Multiple line requests
    within the same subplot appear as grouped bars, one group per bin.

    Args:
        plots_to_make: Output of show_plots' internal grouping — list of groups, where
            each group is a list of line-request dicts (keys: exp_result, split,
            class_id, class_name, language_key).
        metric: Metric key to retrieve from ExperimentResult.
        show: Whether to call plt.show().
        save: Whether to save the figure to disk (requires filename).
        filename: Output filename (required when save=True).
        y_axis_range: Fixed (min, max) y-axis range; auto-computed from binned values if None.
        legend_position: Legend location string passed to matplotlib.
        bins: Labels for the layer bins. Defaults to ["early", "middle", "late"].
        xlabel: Label for the x-axis (default "Layer group").
        scale: Scaling factor applied to the default figure size.
        sort_lines: If True, sort bars within each subplot by descending average value.
        horizontal_line: Draw a reference line — numeric value, "baseline_f1", or
            "control_average" (renders the control series as a dashed horizontal average).
        subplot_titles: Override auto-generated subplot titles; pass [] to suppress all.
    """
    if (save and not filename) or (filename and not save):
        raise KeyError("save and filename should both be specified or neither")

    if bins is None:
        bins = ["early", "middle", "late"]

    num_bins = len(bins)
    num_plots = len(plots_to_make)

    # Place exactly 3 plots in a single row; otherwise approximate a square.
    if num_plots == 3:
        cols = 3
        rows = 1
    else:
        cols = math.ceil(math.sqrt(num_plots))
        rows = math.ceil(num_plots / cols)
    figsize = (7 * scale * cols, 4 * scale * rows)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=figsize, squeeze=False)

    attribute_sequence: list[str] = [
        "language",
        "probing_task",
        "label",
        "probe_type",
        "model_name",
        "extra_iter_num",
        "zeroed_out_activation_dims",
        "zeroed_out_weight_dims",
        "split",
        "metric",
    ]

    def _extract_values(line_request: dict) -> list[float]:
        exp_result: ExperimentResult = line_request["exp_result"]
        split: str = line_request["split"]
        class_id: str = line_request["class_id"]
        results = exp_result.metrics[split][metric]
        if "per_class" in metric or "overlapping_idx_amounts" in metric:
            values: list[float] = [r.get(class_id, 0) for r in results]
            if "overlapping_idx_amounts" in metric:
                values = values[1:]
        else:
            values = [float(v) for v in results]
        return values

    def _bin_average(values: list[float]) -> list[float]:
        n = len(values)
        base = n // num_bins
        remainder = n % num_bins
        averages: list[float] = []
        start = 0
        for i in range(num_bins):
            size = base + (1 if i < remainder else 0)
            chunk = values[start : start + size]
            averages.append(float(np.mean(chunk)) if chunk else 0.0)
            start += size
        return averages

    # Auto y-axis range from binned values
    if y_axis_range is None:
        all_binned: list[float] = []
        for group in plots_to_make:
            seen_test_a_base = False
            for line_request in group:
                exp_result = line_request["exp_result"]
                is_test_a_base = (
                    line_request["split"] == "test_a" and exp_result.extra_iter_num == 0
                )
                if is_test_a_base:
                    if seen_test_a_base:
                        continue
                    seen_test_a_base = True
                all_binned.extend(_bin_average(_extract_values(line_request)))
        margin = 0.1
        y_axis_range = (min(all_binned) - margin, max(all_binned) + margin)

    for plot_idx, group_of_line_requests in enumerate(plots_to_make):
        ax = axs[plot_idx // cols, plot_idx % cols]

        # Build per-line attribute dicts for title/legend generation
        attrs_per_line: list[dict[str, str]] = []
        for line_request in group_of_line_requests:
            exp_result: ExperimentResult = line_request["exp_result"]
            display_language: str = line_request.get(
                "language_key", exp_result.language
            )
            raw_attrs: dict[str, str] = {
                "language": get_verbose_version_of_language_string(display_language),
                "probing_task": exp_result.probing_task,
                "probe_type": exp_result.probe_type,
                "model_name": exp_result.model_name,
                "extra_iter_num": str(exp_result.extra_iter_num),
                "zeroed_out_activation_dims": str(
                    getattr(exp_result, "zeroed_out_activation_dims", 0)
                ),
                "zeroed_out_weight_dims": str(
                    getattr(exp_result, "zeroed_out_weight_dims", 0)
                ),
                "split": line_request["split"],
                "metric": metric,
                "label": line_request["class_name"],
            }
            attrs_per_line.append(
                {k: v.replace("_", " ") for k, v in raw_attrs.items()}
            )

        # Determine which attributes are common vs. varying across all lines
        common_attrs: dict[str, str] = {}
        varying_attrs: set[str] = set()
        for attr_key in attribute_sequence:
            if (
                attr_key == "label"
                and "per_class" not in metric
                and "overlapping_idx_amounts" not in metric
            ):
                continue
            values_for_attr = [a[attr_key] for a in attrs_per_line]
            if len(set(values_for_attr)) == 1:
                common_attrs[attr_key] = values_for_attr[0]
            else:
                varying_attrs.add(attr_key)

        # Title
        common_parts = [
            common_attrs[k] for k in attribute_sequence if k in common_attrs
        ]
        different_parts = [
            k.replace("_", " ") + "s" for k in attribute_sequence if k in varying_attrs
        ]
        if different_parts:
            title = f"Results of {', '.join(common_parts)} for different {', '.join(different_parts)} over layer groups"
        else:
            title = f"Results of {', '.join(common_parts)} over layer groups"

        if subplot_titles is None or (
            subplot_titles and plot_idx >= len(subplot_titles)
        ):
            ax.set_title(title, fontsize=10)
        elif subplot_titles:
            ax.set_title(subplot_titles[plot_idx], fontsize=10)

        # Collect bar data for each line request
        bars_to_plot: list[dict] = []
        seen_test_a_base = False
        for i, line_request in enumerate(group_of_line_requests):
            exp_result = line_request["exp_result"]
            split = line_request["split"]
            is_test_a_base = split == "test_a" and exp_result.extra_iter_num == 0
            if is_test_a_base:
                if seen_test_a_base:
                    continue
                seen_test_a_base = True

            binned = _bin_average(_extract_values(line_request))

            # Legend label — same cross-lingual label rewriting as plot_metrics_by_group
            legend_parts = [
                attrs_per_line[i][k] for k in attribute_sequence if k in varying_attrs
            ]
            cl_idx = next(
                (
                    j
                    for j, p in enumerate(legend_parts)
                    if _CROSS_LINGUAL_RE.fullmatch(p)
                ),
                None,
            )
            tz_idx = next(
                (j for j, p in enumerate(legend_parts) if _TEST_Z_RE.fullmatch(p)), None
            )
            legend_label = " ".join(legend_parts)

            if cl_idx is not None and tz_idx is not None:
                cl = _CROSS_LINGUAL_RE.fullmatch(legend_parts[cl_idx])
                tz = _TEST_Z_RE.fullmatch(legend_parts[tz_idx])
                X, Y, raw_Z = cl.group(1), cl.group(2), tz.group(1)  # type: ignore
                other = [
                    p for j, p in enumerate(legend_parts) if j not in {cl_idx, tz_idx}
                ]
                if raw_Z == "a":
                    legend_label = " ".join([*other, f"{X}→{Y}→{X}"]).strip()
                elif raw_Z == "b":
                    legend_label = " ".join([*other, f"{X}→{Y}"]).strip()
            elif (
                cl_idx is not None
                and len(legend_parts) == 1
                and exp_result.extra_iter_num > 0
            ):
                cl = _CROSS_LINGUAL_RE.fullmatch(legend_parts[cl_idx])
                X, Y = cl.group(1), cl.group(2)  # type: ignore
                legend_label = f"t{X}→{Y}"

            if is_test_a_base:
                lang_a = exp_result.language.split("→")[0]
                legend_label = f"{LANGUAGE_FULL_NAME_MAP[lang_a]}"

            bars_to_plot.append(
                {
                    "binned": binned,
                    "average_value": float(np.mean(binned)),
                    "legend_label": legend_label,
                    "is_control": "control" in exp_result.probing_task,
                    "is_test_a_base": is_test_a_base,
                    "probing_task": exp_result.probing_task,
                    "language": exp_result.language,
                }
            )

        if sort_lines:
            bars_to_plot.sort(key=lambda x: x["average_value"], reverse=True)

        # Assign colours (same language-based scheme as plot_metrics_by_group)
        lang_color_counts: dict[str, int] = defaultdict(int)
        okabe_ito_idx = 0
        for bar_data in bars_to_plot:
            lang_parts = bar_data["language"].split("→")
            lang_key = lang_parts[0] if bar_data["is_test_a_base"] else lang_parts[-1]
            lang_count = lang_color_counts[lang_key]
            lang_color_counts[lang_key] += 1

            if lang_key in LANGUAGE_COLOURS:
                if lang_count == 0:
                    colour = LANGUAGE_COLOURS[lang_key][0]
                elif lang_count == 1:
                    colour = mcolors.to_rgb(LANGUAGE_COLOURS[lang_key][1])
                else:
                    colour = OKABE_ITO_PALETTE[okabe_ito_idx % len(OKABE_ITO_PALETTE)]
                    okabe_ito_idx += 1
            else:
                colour = OKABE_ITO_PALETTE[okabe_ito_idx % len(OKABE_ITO_PALETTE)]
                okabe_ito_idx += 1

            bar_data["colour"] = colour

        # Draw grouped bars
        n_lines = len(bars_to_plot)
        x = np.arange(num_bins)
        bar_width = (0.8 / n_lines) if n_lines > 1 else 0.5
        # Each subplot is 7*scale inches wide, spanning num_bins data units.
        # Fit 5 chars ("-0.00") in 80% of the bar width, assuming ~0.6x aspect ratio for the font.
        bar_width_inches = bar_width / num_bins * 7 * scale
        label_fontsize = max(5, min(9, int(bar_width_inches * 72 / (5 * 0.6))))

        for i, bar_data in enumerate(bars_to_plot):
            offset = (i - (n_lines - 1) / 2) * bar_width
            colour = bar_data["colour"]
            if horizontal_line == "control_average" and bar_data["is_control"]:
                ax.axhline(
                    y=bar_data["average_value"],
                    linestyle="--",
                    alpha=0.7,
                    color=colour,
                    label=f"avg {bar_data['legend_label']}",
                )
            else:
                ax.bar(
                    x + offset,
                    bar_data["binned"],
                    width=bar_width,
                    label=bar_data["legend_label"],
                    color=colour,
                    alpha=0.85,
                )
                for bin_x, val in zip(x + offset, bar_data["binned"]):
                    va = "top" if val >= 0 else "bottom"
                    pad = -0.01 if val >= 0 else 0.01
                    ax.text(
                        bin_x,
                        val + pad,
                        f"{val:.2f}",
                        ha="center",
                        va=va,
                        fontsize=label_fontsize,
                        color="black",
                    )

        # Horizontal reference lines
        if horizontal_line == "baseline_f1":
            if "f1" not in metric:
                print(
                    f"Warning: horizontal_line='baseline_f1' specified but metric is '{metric}'."
                )
            baseline_values = [
                calculate_majority_class_baseline_f1(
                    probing_task=b["probing_task"], language="en"
                )
                for b in bars_to_plot
            ]
            if max(baseline_values) - min(baseline_values) <= 0.05:
                ax.axhline(
                    y=float(np.mean(baseline_values)),
                    linestyle="--",
                    alpha=0.7,
                    color="gray",
                    label="baseline",
                )
            else:
                for bar_data, baseline in zip(bars_to_plot, baseline_values):
                    label = f"majority baseline {bar_data['legend_label']}".strip()
                    ax.axhline(
                        y=baseline,
                        linestyle="--",
                        alpha=0.7,
                        color=bar_data["colour"],
                        label=label,
                    )
        elif horizontal_line != "" and horizontal_line != "control_average":
            ax.axhline(y=float(horizontal_line), linestyle="--", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel(xlabel)
        ylabel = (
            f"selectivity ({metric[len('marginal_'):].replace('_', ' ')})"
            if metric.startswith("marginal_")
            else metric.replace("_", " ")
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(loc=legend_position, fontsize=7)
        ax.set_ylim(y_axis_range)

    for plot_idx in range(num_plots, rows * cols):
        axs[plot_idx // cols, plot_idx % cols].set_visible(False)

    if save:
        save_dir = Path(PLOTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)
        filepath = save_dir / (
            filename if filename.endswith(".png") else f"{filename}.png"
        )
        fig.savefig(filepath)  # type: ignore
        print(f"Plot saved to {filepath}")

    if show:
        plt.show()


def plot_confusion_matrix(
    exp_result: ExperimentResult,
    split: str,
    layer_num: int = 0,
    show: bool = True,
    save: bool = False,
    filename: str = "",
    figsize: tuple[int, int] = (8, 6),
    include_unknown=False,
) -> None:
    """
    Plot a confusion matrix as a heatmap using seaborn.

    Args:
        exp_result: ExperimentResult object containing the confusion matrix.
        split: Data split (e.g., "train" or "test").
        layer_num: Index into the stored confusion matrix list (0 for experiment 3,
            layer number for experiments 1 and 2).
        show: Whether to call plt.show().
        save: Whether to save the figure to disk (requires filename).
        filename: Output filename (required when save=True).
        figsize: Figure size as (width, height).
        include_unknown: If True, keep the -1 "unknown" row/column in the 4x4 matrix
            (experiment 3 only). Defaults to False.

    Raises:
        KeyError: If save and filename are not both specified or both omitted.
    """
    if (save and not filename) or (filename and not save):
        raise KeyError("save and filename should both be specified or neither")

    assert not (
        include_unknown and exp_result.experiment_number != 3
    ), f"include_unknown={include_unknown} only makes sense if experiment_number is 3"

    class_names_list: list[str] = [""] * len(LABEL_MAP)
    for label_str, label_idx in LABEL_MAP.items():
        class_names_list[label_idx] = label_str

    # Get confusion matrix for the specified layer
    cm: ndarray = exp_result.metrics[split]["cm"][layer_num]

    if exp_result.experiment_number == 3:
        if include_unknown:
            class_names_list.insert(0, "unknown")
        else:
            # If include_unknown is False, cut off the first column and row (unknown)
            cm = cm[1:, 1:]

    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)

    # Create title
    title: str = f"Confusion matrix for {exp_result.probe_type} probe in layer {layer_num} of {exp_result.model_name} trained with {LANGUAGE_FULL_NAME_MAP[exp_result.language]} activations of the {split} dataset and {exp_result.probing_task} labels"

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names_list,  # type: ignore
        yticklabels=class_names_list,  # type: ignore
        cbar_kws={"label": "Count"},
        ax=ax,
    )

    ax.set_title(title, fontsize=10, wrap=True)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")

    fig.tight_layout()

    if save:
        save_dir = Path(PLOTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)

        if filename.endswith(".png"):
            filepath: Path = save_dir / filename
        else:
            filepath: Path = save_dir / f"{filename}.png"

        fig.savefig(filepath)  # type: ignore
        print(f"Confusion matrix plot saved to {filepath}")

    if show:
        plt.show()
