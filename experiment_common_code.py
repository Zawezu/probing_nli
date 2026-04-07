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

from common_constants import (
    EXPERIMENT_RESULTS_FOLDER,
    PLOTS_FOLDER,
    LABEL_MAP,
    LANGUAGE_FULL_NAME_MAP,
)

mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}

class_names_list: list[str] = [""] * len(LABEL_MAP)
for label_str, label_idx in LABEL_MAP.items():
    class_names_list[label_idx] = label_str


class ExperimentResult:
    def __init__(
        self,
        experiment_number: int,
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
    ) -> None:
        self.experiment_number: int = experiment_number
        self.language: str = language
        self.probing_task: str = probing_task
        self.probe_type: str = probe_type
        self.model_name: str = model_name

        self.metrics: dict[str, dict[str, Any]] = {"train": {}, "test": {}}

    def append_metric(self, split: str, metric: str, value: Any) -> None:
        try:
            self.metrics[split][metric].append(value)
        except KeyError:
            self.metrics[split][metric] = [value]

    def add_metrics_from_confusion_matrix(self) -> None:
        """
        Calculate and add overall and per-class metrics to an ExperimentResult object
        based on confusion matrices already stored in the result.

        Adds the following metrics to result.metrics[split]:
        - accuracy: Overall accuracy (float)
        - precision: Overall precision (macro average, float)
        - recall: Overall recall (macro average, float)
        - f1: Overall F1 score (macro average, float)
        - per_class_precision: List of per-class precision scores (list[float])
        - per_class_recall: List of per-class recall scores (list[float])
        - per_class_f1: List of per-class F1 scores (list[float])

        Args:
            result: ExperimentResult object with confusion matrices stored in metrics["train"]["cm"] and metrics["test"]["cm"]
        """
        for split in ["train", "test"]:
            # Take the confusion matrix at the latest layer that has been recorded
            cm: ndarray = self.metrics[split]["cm"][-1]

            if cm is None:
                print(f"Warning: No confusion matrix found for {split} split")
                continue

            # Accuracy: sum of diagonal / sum of all elements
            self.append_metric(split, "accuracy", float(np.trace(cm) / np.sum(cm)))

            # Per-class metrics
            num_classes: int = cm.shape[0]
            per_class_precision: list[float] = []
            per_class_recall: list[float] = []
            per_class_f1: list[float] = []

            for class_idx in range(num_classes):
                # Recall (sensitivity/true positive rate): TP / (TP + FN)
                # TP is diagonal, FN is rest of row
                recall: float = float(
                    cm[class_idx, class_idx] / np.sum(cm[class_idx, :])
                )
                per_class_recall.append(recall)

                # Precision: TP / (TP + FP)
                # TP is diagonal, FP is rest of column
                precision: float = float(
                    cm[class_idx, class_idx] / np.sum(cm[:, class_idx])
                )
                per_class_precision.append(precision)

                # F1 score: 2 * (precision * recall) / (precision + recall)
                if precision + recall == 0:
                    f1 = 0.0
                else:
                    f1: float = 2 * (precision * recall) / (precision + recall)
                per_class_f1.append(float(f1))

            # Store per-class metrics
            self.append_metric(split, "per_class_precision", per_class_precision)
            self.append_metric(split, "per_class_recall", per_class_recall)
            self.append_metric(split, "per_class_f1", per_class_f1)

            # Overall metrics (macro average)
            self.append_metric(split, "precision", float(np.mean(per_class_precision)))
            self.append_metric(split, "recall", float(np.mean(per_class_recall)))
            self.append_metric(split, "f1", float(np.mean(per_class_f1)))

    def add_marginal_metrics(self, control_exp_result: "ExperimentResult") -> None:
        assert (
            "control" not in self.probing_task
        ), "add_difference_with_control_metrics() cannot be used on a control experiment result"
        assert (
            "control" in control_exp_result.probing_task
        ), "control_exp_result must contain the results using some control task"

        # Helper function to get the marginal value for a particular metric
        def get_marginal_value(
            metric: str, layer_num: int, cls: None | int = None
        ) -> float:
            if cls is None:
                marginal_value: float = (
                    self.metrics[split][metric][layer_num]
                    - control_exp_result.metrics[split][metric][layer_num]
                )
            else:
                marginal_value: float = (
                    self.metrics[split][metric][layer_num][cls]
                    - control_exp_result.metrics[split][metric][layer_num][cls]
                )
            return marginal_value

        for split in ["train", "test"]:
            for layer_num in range(self.get_num_layers()):
                # Take the confusion matrix at the latest layer that has been recorded
                cm: ndarray = self.metrics[split]["cm"][layer_num]

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
                marginal_per_class_precision: list[float] = []
                marginal_per_class_recall: list[float] = []
                marginal_per_class_f1: list[float] = []

                for class_idx in range(num_classes):
                    # Recall
                    marginal_recall: float = get_marginal_value(
                        "per_class_recall", layer_num, class_idx
                    )
                    marginal_per_class_recall.append(marginal_recall)

                    # Precision
                    marginal_precision: float = get_marginal_value(
                        "per_class_precision", layer_num, class_idx
                    )
                    marginal_per_class_precision.append(marginal_precision)

                    # F1 score
                    marginal_f1: float = get_marginal_value(
                        "per_class_f1", layer_num, class_idx
                    )
                    marginal_per_class_f1.append(marginal_f1)

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
                    float(np.mean(marginal_per_class_precision)),
                )
                self.append_metric(
                    split, "marginal_recall", float(np.mean(marginal_per_class_recall))
                )
                self.append_metric(
                    split, "marginal_f1", float(np.mean(marginal_per_class_f1))
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
            self.language, self.probing_task, self.probe_type, self.model_name
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
        return len(self.metrics["test"]["accuracy"])

    @staticmethod
    def get_filename(
        language: str, probing_task: str, probe_type: str, model_name: str
    ) -> str:
        """Generate filename based on experiment parameters."""
        return f"{language},{probing_task},{probe_type},{model_name}.pkl"

    @staticmethod
    def get_from_file(
        experiment_number: int,
        language: str,
        probing_task: str,
        probe_type: str,
        model_name: str,
    ) -> "ExperimentResult":
        """
        Load an ExperimentResult from a pickle file.

        Args:
            filepath: Path to the pickle file

        Returns:
            The loaded ExperimentResult object
        """
        filepath = f"{EXPERIMENT_RESULTS_FOLDER}/experiment_{experiment_number}/{ExperimentResult.get_filename(language, probing_task, probe_type, model_name)}"
        with open(filepath, "rb") as f:
            return pickle.load(f)


def calculate_y_axis_range(
    experiments: list[ExperimentResult],
    metric_types: list[str],
) -> tuple[float, float]:
    """
    Calculate the y-axis range needed to encompass all values across all experiments and metric types.

    Args:
        experiments: List of ExperimentResult objects
        metric_types: List of metric type strings (attribute names)

    Returns:
        Tuple of (min_value, max_value) for the y-axis range
    """
    all_values: list[float] = []
    for experiment in experiments:
        for metric_type in metric_types:
            results: list[float] = getattr(experiment, metric_type, [])
            if results:
                all_values.extend(results)

    if not all_values:
        return (0, 1)

    return (min(all_values), max(all_values))


def show_plots(
    model_names: list[str],
    splits: list[str],
    languages: list[str],
    probing_tasks: list[str],
    probe_type: str,
    experiment_number: int,
    metric: str,
    separate_chars_within_plot: list[str] = ["language", "probing_task"],
    show: bool = True,
    save: bool = False,
    filename: str = "",
) -> None:
    if (save and not filename) or (filename and not save):
        raise KeyError("save and filename should both be specified or neither")

    extended_class_names_list: list[str] = class_names_list + [
        "all"
    ]  # This last element represents any label.

    if "per_class" in metric:
        class_ids: list[int] = list(LABEL_MAP.values())
    else:
        class_ids = [-1]

    # Validate that specified characteristics are valid
    valid_characteristics: list[str] = [
        "model_name",
        "split",
        "class",
        "language",
        "probing_task",
    ]

    separate_chars_outside_plot: list[str] = [
        c for c in valid_characteristics if c not in separate_chars_within_plot
    ]

    for char in separate_chars_within_plot:
        if char not in valid_characteristics:
            raise ValueError(
                f"Invalid characteristic: {char}. Must be one of {valid_characteristics}"
            )

    # Generate all combinations of all characteristics
    all_combinations: list[dict[str, Any]] = []
    for model_name, split, class_id, language, probing_task in product(
        model_names, splits, class_ids, languages, probing_tasks
    ):
        all_combinations.append(
            {
                "model_name": model_name,
                "split": split,
                "class": extended_class_names_list[class_id],
                "language": language,
                "probing_task": probing_task,
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
        )

        # Create plot request
        line_request = {
            "exp_result": exp_result,
            "split": combo["split"],
            "class": combo["class"],
        }
        plots_dict[key].append(line_request)

    # Convert grouped plots to list format
    plots_to_make: list[list[dict]] = list(plots_dict.values())

    plot_metrics_by_group(plots_to_make, metric, show, save, filename)


def plot_metrics_by_group(
    plots_to_make: list[list[dict]],
    metric: str,
    show: bool,
    save: bool,
    filename: str,
    xlabel: str = "Layer",
    y_axis_range: tuple[float, float] | None = None,
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
                results_for_determining_axis: list[float | list[float]] = line_request[
                    "exp_result"
                ].metrics[line_request["split"]][metric]
                if isinstance(results_for_determining_axis[0], float):
                    all_values.extend(results_for_determining_axis)  # type: ignore
                else:
                    for results_list in results_for_determining_axis:
                        all_values.extend(results_list)  # type: ignore

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
            attrs: dict[str, str] = {
                "language": LANGUAGE_FULL_NAME_MAP[line_request["exp_result"].language],
                "probing_task": line_request["exp_result"].probing_task,
                "probe_type": line_request["exp_result"].probe_type,
                "model_name": line_request["exp_result"].model_name,
                "split": line_request["split"],
                "metric": metric,
                "label": line_request["class"],
            }

            # Replace all underscores by spaces to create a nicer plot
            for key in attrs.keys():
                attrs[key] = attrs[key].replace("_", " ")
            attrs_per_line.append(attrs)

        # Determine which attributes are common and which vary across all lines
        common_attrs: dict[str, str] = {}
        varying_attrs: set[str] = set()

        for attr_key in attribute_sequence:
            if not (attr_key == "label" and "per_class" not in metric):
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
            key.replace("_", " ") + "s"
            for key in attribute_sequence
            if key in varying_attrs
        ]

        if different_parts:
            title: str = f"Results of {', '.join(common_parts)} for different {', '.join(different_parts)} over the {xlabel.lower()}s"
        else:
            title = f"Results of {', '.join(common_parts)} over the {xlabel}s"

        ax.set_title(title, fontsize=7)

        # Plot each line with intelligently generated legend label
        for i, line_request in enumerate(group_of_line_requests):
            results: list[float | list[float]] = line_request["exp_result"].metrics[
                line_request["split"]
            ][metric]
            layers = range(len(results))

            # Generate legend label with only varying attributes
            legend_parts: list[str] = [
                attrs_per_line[i][key]
                for key in attribute_sequence
                if key in varying_attrs
            ]
            legend_label: str = " ".join(legend_parts)

            if "per_class" in metric:
                # If we are dealing with a per-class metric, we add a plot for the requested label id
                results_for_this_label: list[float] = [
                    result[LABEL_MAP[line_request["class"]]]
                    for result in results  # type: ignore
                ]
                ax.plot(layers, results_for_this_label, marker="o", label=legend_label)
            else:
                # If we are dealing with a non-per-class metric, we simply plot the results
                ax.plot(layers, results, marker="o", label=legend_label)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left")
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
    layer_num: int,
    show: bool = True,
    save: bool = False,
    filename: str = "",
    figsize: tuple[int, int] = (8, 6),
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

    # Get confusion matrix for the specified layer
    cm: ndarray = exp_result.metrics[split]["cm"][layer_num]

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
