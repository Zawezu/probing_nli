import math
import pickle
from pathlib import Path
from typing import Any
from itertools import product
from collections import defaultdict
from matplotlib.pylab import ndarray
import matplotlib.pyplot as plt
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
    ) -> None:
        self.experiment_number: int = experiment_number
        self.language: str = language
        self.probing_task: str = probing_task
        self.probe_type: str = probe_type
        self.model_name: str = model_name
        self.extra_iter_num: int = extra_iter_num

        match experiment_number:
            case 1 | 3:
                self.splits: list[str] = ["train", "test"]
            case 2:
                self.splits = ["train_a", "test_a", "train_b", "test_b"]
            case _:
                raise ValueError(f"Incorrect experiment_number: {experiment_number}")

        self.metrics: dict[str, dict[str, Any]] = {split: {} for split in self.splits}

    def append_metric(self, split: str, metric: str, value: Any) -> None:
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
        Calculate and add overall and per-class metrics to an ExperimentResult object
        based on confusion matrices already stored in the result.

        Adds the following metrics to result.metrics[split]:
        - accuracy: Overall accuracy (float)
        - precision: Overall precision (macro average, float)
        - recall: Overall recall (macro average, float)
        - f1: Overall F1 score (macro average, float)
        - per_class_precision: List of per-class precision scores with label keys
        - per_class_recall: List of per-class recall scores with label keys
        - per_class_f1: List of per-class F1 scores with label keys

        Label keys depend on confusion matrix shape:
        - 3x3 matrix: keys are ["0", "1", "2"] (LABEL_MAP values)
        - 4x4 matrix: keys are ["-1", "0", "1", "2"] (including unknown label)

        Args:
            result: ExperimentResult object with confusion matrices stored in metrics["train"]["cm"] and metrics["test"]["cm"]
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
                    idxs_prev_layer = idxs_per_cm_cell[i - 1].copy()

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
        Save the ExperimentResult to a pickle file.

        Args:
            base_dir: Base directory where results will be saved (default: "./{EXPERIMENT_RESULTS_FOLDER}/experiment_{experiment_number}")

        Returns:
            The path to the saved file
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
        )
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

        return str(filepath)

    def get_name(self) -> str:
        return (
            f"{self.language} {self.probing_task} {self.probe_type} {self.model_name}"
        )

    def get_num_layers(self) -> int:
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
    ) -> str:
        """Generate filename based on experiment parameters."""
        if extra_iter_num:
            return f"{language},{probing_task},{probe_type},{model_name},{extra_iter_num}_extra_iters.pkl"
        else:
            return f"{language},{probing_task},{probe_type},{model_name}.pkl"

    @staticmethod
    def get_from_file(
        experiment_number: int,
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
        extra_iter_num: int = 0,
    ) -> "ExperimentResult":
        """
        Load an ExperimentResult from a pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            The loaded ExperimentResult object
        """
        filepath = f"{EXPERIMENT_RESULTS_FOLDER}/experiment_{experiment_number}/{ExperimentResult.get_filename(language, probing_task, probe_type, model_name, extra_iter_num)}"
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
) -> None:
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
    valid_characteristics: list[str] = [
        "model_name",
        "split",
        "class_name",
        "language",
        "probing_task",
        "extra_iter_num",
    ]

    for char in separate_chars_within_plot:
        if char not in valid_characteristics:
            raise ValueError(
                f"Invalid characteristic: {char}. Must be one of {valid_characteristics}"
            )

    separate_chars_outside_plot: list[str] = [
        c for c in valid_characteristics if c not in separate_chars_within_plot
    ]

    # Generate all combinations of all characteristics
    all_combinations: list[dict[str, Any]] = []
    for model_name, split, class_id, language, probing_task, extra_iter_num in product(
        model_names, splits, class_ids, languages, probing_tasks, extra_iter_nums
    ):
        if class_id in extended_class_names.keys():
            class_name: str = extended_class_names[class_id]
        else:
            class_name = class_id

        all_combinations.append(
            {
                "model_name": model_name,
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "language": language,
                "probing_task": probing_task,
                "extra_iter_num": extra_iter_num,
            }
        )

    # Group combinations by the specified characteristics to define each plot
    plots_dict: dict[tuple, list[dict]] = defaultdict(list)

    for combo in all_combinations:
        # Create key from specified characteristics
        key = tuple(combo[char] for char in separate_chars_outside_plot)

        # Load the experiment result
        exp_result: ExperimentResult = ExperimentResult.get_from_file(
            experiment_number,
            combo["language"],
            combo["probing_task"],
            probe_type,
            combo["model_name"],
            combo["extra_iter_num"],
        )

        # Create plot request
        line_request: dict[str, str | ExperimentResult] = {
            "exp_result": exp_result,
            "split": combo["split"],
            "class_id": combo["class_id"],
            "class_name": combo["class_name"],
        }

        print(
            f"Created line request:\n{[f'{key}: {str(value)}' for key, value in line_request.items()]}"
        )

        plots_dict[key].append(line_request)

    # Convert grouped plots to list format
    plots_to_make: list[list[dict]] = list(plots_dict.values())

    plot_metrics_by_group(
        plots_to_make,
        metric,
        show,
        save,
        filename,
        y_axis_range,
        legend_position=legend_position,
    )


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
) -> None:
    """
    Plot metrics grouped by experiment groups.

    Each group (inner list) becomes a separate subplot containing all experiments
    in that group, where each experiment is plotted using its specified split and metric.

    Titles and legends are automatically generated based on common and varying attributes
    (language, probing task, probe type, model name, split, metric).

    Args:
        plots_to_make: List of groups, where each group is a list of dictionaries
        xlabel: Label for x-axis
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Filename for saving (required if save=True)
        y_axis_range: Optional y-axis range (min, max)
        scale: Scaling factor for figure size
    """
    num_plots: int = len(plots_to_make)

    # Calculate grid shape to approximate a square
    cols: int = math.ceil(math.sqrt(num_plots))
    rows: int = math.ceil(num_plots / cols)

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
                    print(exp_result.metrics[split][metric])
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
            attrs: dict[str, str] = {
                "language": get_verbose_version_of_language_string(exp_result.language),
                "probing_task": exp_result.probing_task,
                "probe_type": exp_result.probe_type,
                "model_name": exp_result.model_name,
                "extra_iter_num": str(exp_result.extra_iter_num),
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

        ax.set_title(title, fontsize=7)

        # Prepare line data with average values for sorting
        lines_to_plot: list[dict] = []
        for i, line_request in enumerate(group_of_line_requests):
            exp_result: ExperimentResult = line_request["exp_result"]
            split: str = line_request["split"]
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
            legend_label: str = " ".join(legend_parts)

            if "per_class" in metric or "overlapping_idx_amounts" in metric:
                # If we are dealing with a per-class metric, we add a plot for the requested label id
                results_for_this_label: list[float] = [
                    r.get(class_id, 0)
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
                    }
                )

        # Sort lines by average value in descending order (higher values first)
        lines_to_plot.sort(key=lambda x: x["average_value"], reverse=True)

        # Plot each line in sorted order
        for line_data in lines_to_plot:
            ax.plot(
                line_data["layers"],
                line_data["results"],
                marker="o",
                label=line_data["legend_label"],
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend(loc=legend_position)
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
        exp_result: ExperimentResult object containing the confusion matrix
        split: Data split (e.g., "train" or "test")
        layer_num: Layer number to retrieve the confusion matrix from
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Filename for saving (required if save=True)
        figsize: Figure size as (width, height)

    Raises:
        KeyError: If save and filename are not both specified or both omitted
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
