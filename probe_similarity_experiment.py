from typing import Any
from pathlib import Path
import math
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from probes import AnyProbe, load_probe
from experiment_common_code import OKABE_ITO_PALETTE
from utils import (
    LANGUAGES,
    MODEL_NAMES,
    MODEL_THESIS_NAMES,
    PLOTS_FOLDER,
    REVERSE_LABEL_MAP,
    get_language_pair_combinations,
    get_number_of_layers_from_file,
    get_language_merged_string,
    get_language_pair_permutations,
)

# Create plots directory if it doesn't exist
Path(PLOTS_FOLDER).mkdir(exist_ok=True)


def get_similarity_function(probe: AnyProbe, sim_func: str):
    """
    Return the appropriate similarity function based on sim_func parameter.

    Args:
        probe: The AnyProbe instance
        sim_func: Either "cos_sim" or "l2_dist"

    Returns:
        The corresponding method (calculate_cosine_similarity or calculate_l2_dist)
    """
    if sim_func == "cos_sim":
        return probe.calculate_cosine_similarity
    elif sim_func == "l2_dist":
        return probe.calculate_l2_dist
    elif sim_func == "maha_cos_sim":
        return probe.calculate_maha_cos_sim
    else:
        raise ValueError(
            f"Unknown similarity function: {sim_func}. Must be 'cos_sim', 'l2_dist', or 'maha_cos_sim'"
        )


def get_similarity_metric_name(sim_func: str) -> str:
    """
    Get human-readable name for the similarity metric.

    Args:
        sim_func: Either "cos_sim" or "l2_dist"

    Returns:
        Human-readable metric name
    """
    if sim_func == "cos_sim":
        return "Cosine similarity"
    elif sim_func == "l2_dist":
        return "Euclidean distance"
    elif sim_func == "maha_cos_sim":
        return "Mahalanobis cosine similarity"
    else:
        raise ValueError(
            f"Unknown similarity function: {sim_func}. Must be 'cos_sim', 'l2_dist', or 'maha_cos_sim'"
        )


def get_class_name(class_num: int, per_class: bool) -> str:
    """
    Convert class number to class name.
    If per_class is False and class_num is 0, return "flattened".
    Otherwise, use REVERSE_LABEL_MAP to get the class name.
    """
    if not per_class and class_num == 0:
        return "flattened"
    return REVERSE_LABEL_MAP[class_num]


def get_similarity_range(sims) -> tuple[float, float]:
    vmin = min(
        (val for d1 in sims.values() for d2 in d1.values() for val in d2.values())
    )
    vmax = max(
        (val for d1 in sims.values() for d2 in d1.values() for val in d2.values())
    )

    return vmin, vmax


def print_highest_values_in_probe_coeficcients(probe, n=10) -> None:
    with np.printoptions(threshold=sys.maxsize):
        coef: np.ndarray = probe.get_vector()
        idxs = np.argsort(coef)
        highest_values = coef[idxs][-n:]

        print(f"-------------\n{n} highest values of the coefficient of {probe}:")
        print(highest_values)


def calculate_per_layer_sims_between_langs(
    model_name: str,
    probing_task: str,
    languages: list[str],
    extra_prints: bool,
    per_class: bool,
    sim_func: str = "cos_sim",
    extra_iters: int = 0,
    probe_type: str = "lr",
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
) -> dict[int, dict[int, dict[Any, float]]]:
    sims: dict[int, dict[int, dict[Any, float]]] = {}

    language_pairs: list[tuple[str, str]] = get_language_pair_permutations(languages)
    print(language_pairs)
    num_layers: int = get_number_of_layers_from_file(model_name)
    for layer_num in range(num_layers):
        sims[layer_num] = {}
        for language_a, language_b in language_pairs:
            probe_a: AnyProbe = load_probe(
                language_a,
                layer_num,
                probing_task,
                probe_type,
                model_name,
                extra_iters=extra_iters,
                zeroed_out_activation_dims=zeroed_out_activation_dims,
                zeroed_out_weight_dims=zeroed_out_weight_dims,
            )
            probe_b: AnyProbe = load_probe(
                language_b,
                layer_num,
                probing_task,
                probe_type,
                model_name,
                extra_iters=extra_iters,
                zeroed_out_activation_dims=zeroed_out_activation_dims,
                zeroed_out_weight_dims=zeroed_out_weight_dims,
            )

            sim_method = get_similarity_function(probe_a, sim_func)
            sims_dict: dict[int, float] = sim_method(probe_b, per_class=per_class)

            # Initialize class keys if needed
            for class_num in sims_dict.keys():
                if class_num not in sims[layer_num]:
                    sims[layer_num][class_num] = {}
                sims[layer_num][class_num][f"{language_a},{language_b}"] = sims_dict[
                    class_num
                ]

            if extra_prints:
                print_highest_values_in_probe_coeficcients(probe_a)
                print_highest_values_in_probe_coeficcients(probe_b)

    return sims


def calculate_per_layer_sims_over_extra_iters(
    model_name: str,
    probing_task: str,
    language_pair: tuple[str, str],
    num_refits: int,
    iterations_per_refit: int,
    extra_prints: bool,
    per_class: bool,
    sim_func: str = "cos_sim",
    probe_type: str = "lr",
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
) -> dict[int, dict[int, dict[Any, float]]]:
    sims: dict[int, dict[int, dict[Any, float]]] = {}

    num_layers: int = get_number_of_layers_from_file(
        "olmo_model"
    )  # TODO change this to model_name

    for layer_num in range(num_layers):
        sims[layer_num] = {}

        # Get original probe trained on language a
        original_probe: AnyProbe = load_probe(
            language_pair[0],
            layer_num,
            probing_task,
            probe_type,
            model_name,
            0,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
        )

        for refit_num in range(1, num_refits + 1):
            extra_iters: int = refit_num * iterations_per_refit

            # Get probe refitted on language b for a certain number of extra iterations
            refitted_probe: AnyProbe = load_probe(
                get_language_merged_string(language_pair),
                layer_num,
                probing_task,
                probe_type,
                model_name,
                extra_iters,
                zeroed_out_activation_dims=zeroed_out_activation_dims,
                zeroed_out_weight_dims=zeroed_out_weight_dims,
            )

            sim_method = get_similarity_function(original_probe, sim_func)
            sims_dict: dict[int, float] = sim_method(
                refitted_probe, per_class=per_class
            )

            # Initialize class keys if needed
            for class_num in sims_dict.keys():
                if class_num not in sims[layer_num]:
                    sims[layer_num][class_num] = {}
                sims[layer_num][class_num][extra_iters] = sims_dict[class_num]

            if extra_prints:
                print_highest_values_in_probe_coeficcients(original_probe)
                print_highest_values_in_probe_coeficcients(refitted_probe)

    return sims


def get_filepath(title) -> Path:
    clean_title: str = title.lower().replace(" ", "_").replace("/", "_") + ".png"

    if "between_layers" in clean_title:
        folder = "between_layers"
    elif "at_max_extra_iters" in clean_title:
        folder = "per_extra_iter"
    elif "over_layers" in clean_title or "comparison" in clean_title:
        folder = "per_layers"
    elif "over_extra_iters" in clean_title:
        folder = "per_extra_iter"
    else:
        folder = "."

    folder_path = Path(f"{PLOTS_FOLDER}/{folder}")
    folder_path.mkdir(exist_ok=True)

    filepath: Path = folder_path / clean_title

    return filepath


def plot_probe_weight_magnitudes(
    model_name: str,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    extra_iters: int = 0,
    save: bool = False,
    show: bool = True,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
) -> None:
    probe: AnyProbe = load_probe(
        language,
        layer_num,
        probing_task,
        probe_type,
        model_name,
        extra_iters,
        zeroed_out_activation_dims=zeroed_out_activation_dims,
        zeroed_out_weight_dims=zeroed_out_weight_dims,
    )

    weights = probe.get_vector(per_class=True)  # shape (n_classes, n_features+1)
    n_classes = weights.shape[0]

    fig, axes = plt.subplots(1, n_classes, figsize=(6 * n_classes, 4), sharey=True)
    if n_classes == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        magnitudes = np.abs(weights[i])
        ax.plot(magnitudes, linewidth=0.8)
        ax.set_xlabel("Dimension")
        ax.set_ylabel("|Weight|")
        class_name = get_class_name(i, per_class=True)
        ax.set_title(class_name.capitalize())
        ax.grid(True, alpha=0.3)

    title = f"Weight magnitudes of {probe_type} probe for {model_name} {language} layer {layer_num} {probing_task}"
    if extra_iters:
        title += f" ({extra_iters} extra iters)"
    if zeroed_out_activation_dims:
        title += f" zad={zeroed_out_activation_dims}"
    if zeroed_out_weight_dims:
        title += f" zwd={zeroed_out_weight_dims}"
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_sim_confusion_matrix(
    sims: dict[int, dict[int, dict[Any, float]]],
    layer_num: int,
    title: str,
    save: bool,
    show: bool,
    per_class: bool,
    sim_func: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot confusion matrix(ces) of similarities between language pairs.
    If per_class data is provided, plots one matrix per class.
    """
    class_nums = sorted(sims[layer_num].keys())
    num_classes = len(class_nums)

    # Create subplots, one for each class
    fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5))
    if num_classes == 1:
        axes = [axes]

    for idx, class_num in enumerate(class_nums):
        sims_for_this_class: dict[Any, float] = sims[layer_num][class_num]  # type: ignore
        # Extract unique languages from the keys
        language_pairs: list[str] = list(sims_for_this_class.keys())
        languages: list[str] = sorted(
            list(
                set(
                    [pair.split(",")[0] for pair in language_pairs]
                    + [pair.split(",")[1] for pair in language_pairs]
                )
            )
        )

        # Create matrix
        n: int = len(languages)
        matrix = np.zeros((n, n)) if sim_func == "l2_dist" else np.ones((n, n))

        for pair, value in sims_for_this_class.items():
            source, target = pair.split(",")
            i: int = languages.index(source)
            j: int = languages.index(target)
            matrix[i, j] = value

        # Plot
        ax = axes[idx]
        metric_name = get_similarity_metric_name(sim_func)
        sns.heatmap(
            matrix,
            xticklabels=languages,  # type: ignore
            yticklabels=languages,  # type: ignore
            cmap="Blues",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": metric_name},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        class_name = get_class_name(class_num, per_class)
        ax.set_title(f"{class_name.capitalize()}")  # type: ignore
        ax.set_xlabel("Second Language")  # type: ignore
        ax.set_ylabel("First Language")  # type: ignore

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        # Save plot
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()


def plot_sim_over_the_layers(
    sims,
    language_pairs,
    title: str,
    save: bool,
    show: bool,
    per_class: bool,
    sim_func: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot similarity over layers for each language pair.
    If per_class data is provided, plots one line per class.

    Args:
        sims: Dictionary mapping layer number to dict of class numbers to dict of language pairs and their similarities
        language_pairs: List of tuples containing (source_lang, target_lang) pairs
        per_class: Whether per-class mode is enabled
        sim_func: Similarity function type ("cos_sim" or "l2_dist")
    """
    n_pairs: int = len(language_pairs)
    n_cols: int = min(3, n_pairs)  # Use up to 3 columns
    n_rows: int = (n_pairs + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )

    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    else:
        axes = axes.reshape(n_rows, n_cols)

    # Sort layers for plotting
    layers: list[int] = sorted(sims.keys())

    # Get class numbers
    class_nums: list[int] = sorted(sims[layers[0]].keys())

    # Plot each language pair
    for idx, (lang_a, lang_b) in enumerate(language_pairs):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax = axes[row, col]

        # Extract cosine similarities for this language pair across layers and classes
        pair_key: str = f"{lang_a},{lang_b}"

        # Plot a line for each class
        for class_num in class_nums:
            sims_for_this_class: list[float] = [
                sims[layer][class_num][pair_key] for layer in layers
            ]
            class_name = get_class_name(class_num, per_class)
            ax.plot(
                layers,
                sims_for_this_class,
                marker="o",
                linewidth=2,
                markersize=6,
                label=class_name.capitalize(),
            )

        ax.set_xlabel("Layer")
        metric_name = get_similarity_metric_name(sim_func)
        ax.set_ylabel(metric_name)
        ax.set_title(f"{lang_a}, {lang_b}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim((vmin, vmax))
        ax.legend()

    # Hide unused subplots
    for idx in range(n_pairs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    if save:
        # Save plot
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def calculate_between_layers_sims(
    model_name: str,
    probing_task: str,
    language: str,
    per_class: bool,
    sim_func: str = "cos_sim",
    probe_type: str = "lr",
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
) -> dict[int, dict[str, float]]:
    """
    Calculate similarities between probes at each pair of layers for a single language.
    Returns: {class_num: {f"{layer_a},{layer_b}": similarity}}
    """
    num_layers: int = get_number_of_layers_from_file(model_name)

    probes: dict[int, AnyProbe] = {
        layer_num: load_probe(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
        )
        for layer_num in range(num_layers)
    }

    sims: dict[int, dict[str, float]] = {}

    for layer_a in range(num_layers):
        for layer_b in range(num_layers):
            if layer_a == layer_b:
                continue
            sim_method = get_similarity_function(probes[layer_a], sim_func)
            sims_dict: dict[int, float] = sim_method(
                probes[layer_b], per_class=per_class
            )
            for class_num, value in sims_dict.items():
                if class_num not in sims:
                    sims[class_num] = {}
                sims[class_num][f"{layer_a},{layer_b}"] = value

    return sims


def plot_between_layers_confusion_matrix(
    sims: dict[int, dict[str, float]],
    title: str,
    save: bool,
    show: bool,
    per_class: bool,
    sim_func: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    class_nums = sorted(sims.keys())
    num_classes = len(class_nums)

    fig, axes = plt.subplots(1, num_classes, figsize=(7 * num_classes, 6))
    if num_classes == 1:
        axes = [axes]

    for idx, class_num in enumerate(class_nums):
        sims_for_class = sims[class_num]

        layers = sorted(
            set(int(k.split(",")[0]) for k in sims_for_class)
            | set(int(k.split(",")[1]) for k in sims_for_class)
        )
        layer_to_idx = {layer: i for i, layer in enumerate(layers)}
        n = len(layers)
        matrix = np.zeros((n, n)) if sim_func == "l2_dist" else np.ones((n, n))

        for pair_key, value in sims_for_class.items():
            layer_a, layer_b = pair_key.split(",")
            matrix[layer_to_idx[int(layer_a)], layer_to_idx[int(layer_b)]] = value

        ax = axes[idx]
        metric_name = get_similarity_metric_name(sim_func)
        layer_labels = [str(layer) for layer in layers]
        has_negatives = vmin < 0.0
        heatmap_kwargs: dict = dict(
            xticklabels=layer_labels,  # type: ignore
            yticklabels=layer_labels,  # type: ignore
            annot=False,
            cbar_kws={"label": metric_name},
            vmin=vmin,
            vmax=vmax,
            ax=ax,
        )
        if has_negatives:
            heatmap_kwargs["cmap"] = "RdBu"
            heatmap_kwargs["center"] = 0.0
        else:
            heatmap_kwargs["cmap"] = "Blues"
        sns.heatmap(matrix, **heatmap_kwargs)
        class_name = get_class_name(class_num, per_class)
        ax.set_title(f"{class_name.capitalize()}")  # type: ignore
        ax.set_xlabel("Layer B")  # type: ignore
        ax.set_ylabel("Layer A")  # type: ignore

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_sim_over_the_layers_three_metrics(
    sims_cos: dict,
    sims_l2: dict,
    sims_maha: dict,
    language_pairs: list[tuple[str, str]],
    title: str,
    save: bool,
    show: bool,
    per_class: bool,
) -> None:
    """
    Plot cosine similarity, Mahalanobis cosine similarity, and L2 distance over layers.
    Each subplot has two y-axes:
      - Left [0, 1]: cosine similarity (blue, solid) and Mahalanobis cosine (green, dotted)
      - Right (dynamic): L2 distance (magenta, dashed)
    Each class gets its own shade within each metric's color family.
    """
    n_pairs: int = len(language_pairs)
    n_cols: int = min(3, n_pairs)
    n_rows: int = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True
    )

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)

    layers: list[int] = sorted(sims_cos.keys())
    class_nums: list[int] = sorted(sims_cos[layers[0]].keys())
    n_classes: int = len(class_nums)
    cos_colors = matplotlib.colormaps["Blues"](np.linspace(0.45, 0.9, n_classes))  # type: ignore[attr-defined]
    maha_colors = matplotlib.colormaps["Greens"](np.linspace(0.45, 0.9, n_classes))  # type: ignore[attr-defined]
    l2_colors = matplotlib.colormaps["RdPu"](np.linspace(0.45, 0.9, n_classes))  # type: ignore[attr-defined]

    for idx, (lang_a, lang_b) in enumerate(language_pairs):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax1 = axes[row, col]
        ax2 = ax1.twinx()

        pair_key: str = f"{lang_a},{lang_b}"

        for i, class_num in enumerate(class_nums):
            class_name = get_class_name(class_num, per_class)

            cos_vals: list[float] = [
                sims_cos[layer][class_num][pair_key] for layer in layers
            ]
            maha_vals: list[float] = [
                sims_maha[layer][class_num][pair_key] for layer in layers
            ]
            l2_vals: list[float] = [
                sims_l2[layer][class_num][pair_key] for layer in layers
            ]

            ax1.plot(
                layers,
                cos_vals,
                marker="o",
                linewidth=2,
                markersize=5,
                linestyle="-",
                color=cos_colors[i],
                label=f"Cos ({class_name.capitalize()})",
            )
            ax1.plot(
                layers,
                maha_vals,
                marker="^",
                linewidth=2,
                markersize=5,
                linestyle=":",
                color=maha_colors[i],
                label=f"Maha cos ({class_name.capitalize()})",
            )
            ax2.plot(
                layers,
                l2_vals,
                marker="s",
                linewidth=2,
                markersize=5,
                linestyle="--",
                color=l2_colors[i],
                label=f"L2 ({class_name.capitalize()})",
            )

        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Cosine similarity")
        ax1.set_ylim(0.0, 1.0)
        ax2.set_ylabel("Euclidean distance")
        ax1.set_title(f"{lang_a}, {lang_b}")
        ax1.grid(True, alpha=0.3)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7)

    for idx in range(n_pairs, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_sim_over_extra_iters(
    sims: dict[int, dict[int, dict[Any, float]]],
    layer_nums_to_plot: list[int],
    title: str,
    save: bool,
    show: bool,
    per_class: bool,
    sim_func: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    num_layers: int = len(layer_nums_to_plot)
    n_cols: int = min(3, num_layers)  # Use up to 3 columns
    n_rows: int = (num_layers + n_cols - 1) // n_cols  # Calculate rows needed

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True, sharey=True
    )

    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, n_cols)
    else:
        axes = axes.reshape(n_rows, n_cols)

    # Get class numbers from the first layer
    first_layer = layer_nums_to_plot[0]
    class_nums: list[int] = sorted(sims[first_layer].keys())

    # Plot each layer
    for idx, layer_num in enumerate(layer_nums_to_plot):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax = axes[row, col]

        # Plot a line for each class
        for class_num in class_nums:
            # Extract similarities for this class across extra iterations
            extra_iters: list[int] = list(sims[layer_num][class_num].keys())
            sims_for_this_class: list[float] = list(sims[layer_num][class_num].values())

            class_name = get_class_name(class_num, per_class)
            ax.plot(
                extra_iters,
                sims_for_this_class,
                marker="o",
                linewidth=2,
                markersize=6,
                label=class_name.capitalize(),
            )

        ax.set_xlabel("Extra iters", fontsize=8)
        metric_name = get_similarity_metric_name(sim_func)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Layer {layer_num}", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim((vmin, vmax))
        ax.legend()

    # Hide unused subplots
    for idx in range(num_layers, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save:
        # Save plot
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


def plot_sim_over_layers_at_max_iters(
    sims_per_model_per_pair: dict[
        str, dict[str, dict[int, dict[int, dict[Any, float]]]]
    ],
    max_extra_iters: int,
    title: str,
    save: bool,
    show: bool,
    sim_func: str,
    per_language_pair: bool,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot similarity at max extra iters over layers.
    x-axis: layer, y-axis: similarity (averaged across classes).
    If per_language_pair: grid of subplots (one per model), one line per language pair.
    Otherwise: single plot, one line per model labeled with its thesis name.
    """
    model_names_in_plot = list(sims_per_model_per_pair.keys())
    n_models = len(model_names_in_plot)

    first_model_sims = next(iter(sims_per_model_per_pair.values()))
    first_pair_sims = next(iter(first_model_sims.values()))
    layers: list[int] = sorted(first_pair_sims.keys())
    lang_pair_strs: list[str] = list(first_model_sims.keys())
    metric_name = get_similarity_metric_name(sim_func)

    if per_language_pair:
        n_cols: int = math.ceil(math.sqrt(n_models))
        n_rows: int = math.ceil(n_models / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(7 * n_cols, 4 * n_rows), squeeze=False
        )

        for idx, model_name in enumerate(model_names_in_plot):
            row: int = idx // n_cols
            col: int = idx % n_cols
            ax = axes[row, col]
            model_sims = sims_per_model_per_pair[model_name]

            for palette_idx, lang_pair_str in enumerate(lang_pair_strs):
                pair_sims = model_sims[lang_pair_str]
                class_nums: list[int] = sorted(pair_sims[layers[0]].keys())
                sim_values: list[float] = [
                    float(
                        np.mean(
                            [
                                pair_sims[layer][class_num][max_extra_iters]
                                for class_num in class_nums
                            ]
                        )
                    )
                    for layer in layers
                ]
                colour = OKABE_ITO_PALETTE[palette_idx % len(OKABE_ITO_PALETTE)]
                ax.plot(
                    layers, sim_values, label=lang_pair_str, color=colour, linewidth=3
                )

            ax.set_title(MODEL_THESIS_NAMES.get(model_name, model_name), fontsize=10)
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_name)
            ax.grid(True, alpha=0.3)
            ax.set_ylim((vmin, vmax))
            ax.legend(loc="upper left", fontsize=7)

        for idx in range(n_models, n_rows * n_cols):
            axes[idx // n_cols, idx % n_cols].set_visible(False)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))

        for palette_idx, model_name in enumerate(model_names_in_plot):
            model_sims = sims_per_model_per_pair[model_name]
            class_nums = sorted(next(iter(model_sims.values()))[layers[0]].keys())
            avg_values: list[float] = [
                float(
                    np.mean(
                        [
                            model_sims[lp][layer][class_num][max_extra_iters]
                            for lp in lang_pair_strs
                            for class_num in class_nums
                        ]
                    )
                )
                for layer in layers
            ]
            colour = OKABE_ITO_PALETTE[palette_idx % len(OKABE_ITO_PALETTE)]
            label = MODEL_THESIS_NAMES.get(model_name, model_name)
            ax.plot(layers, avg_values, label=label, color=colour, linewidth=3)

        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(vmin, vmax)
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save:
        filepath: Path = get_filepath(title)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", help="enter the model names", nargs="*", default=MODEL_NAMES
    )
    parser.add_argument("-l", help="enter the languages", nargs="*", default=LANGUAGES)
    parser.add_argument(
        "-t", help="enter the probing tasks", nargs="*", default=["standard"]
    )
    parser.add_argument(
        "-ei",
        help="enter the extra iterations (only for per_layer)",
        nargs="*",
        default=[0],
    )

    parser.add_argument(
        "-e",
        help="enter the experiment to perform: per_layer, per_layer_three_metrics, per_extra_iter, between_layers, or weight_magnitudes",
        default="",
    )

    parser.add_argument(
        "-sv",
        help="whether to save the plot or not",
        nargs="?",
        default="False",
        const="True",
    )

    parser.add_argument(
        "-sh",
        help="whether to save the plot or not",
        nargs="?",
        default="False",
        const="True",
    )

    parser.add_argument(
        "-ep",
        help="whether to do some extra prints (highest values in probe coeficcients)",
        nargs="?",
        default="False",
        const="True",
    )

    parser.add_argument(
        "-pc",
        help="whether to calculate cosine similarity per class",
        nargs="?",
        default="False",
        const="True",
    )

    parser.add_argument(
        "-sf",
        help="similarity function to use: cos_sim or l2_dist",
        default="cos_sim",
        choices=["cos_sim", "l2_dist", "maha_cos_sim"],
    )

    parser.add_argument(
        "-pt",
        help="probe type to use: lr or mm",
        default="lr",
        choices=["lr", "mm"],
    )
    parser.add_argument(
        "-zad",
        help="number of highest-magnitude activation dims zeroed during probe training (0 = disabled)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-zwd",
        help="number of highest-magnitude weight dims to zero out per class after loading (0 = disabled)",
        type=int,
        default=0,
    )

    args: argparse.Namespace = parser.parse_args()
    print(args)

    model_names: list[str] = args.m
    languages: list[str] = args.l
    probing_tasks: list[str] = args.t
    extra_iter_nums: list[int] = [int(ei) for ei in args.ei]
    experiment_type: str = args.e
    save: bool = args.sv.lower() == "true"
    show: bool = args.sh.lower() == "true"
    extra_prints: bool = args.ep.lower() == "true"
    per_class: bool = args.pc.lower() == "true"
    sim_func: str = args.sf
    probe_type: str = args.pt
    zeroed_out_activation_dims: int = args.zad
    zeroed_out_weight_dims: int = args.zwd

    zeroing_suffix: str = ""
    if zeroed_out_activation_dims:
        zeroing_suffix += f" zad={zeroed_out_activation_dims}"
    if zeroed_out_weight_dims:
        zeroing_suffix += f" zwd={zeroed_out_weight_dims}"

    if extra_iter_nums != [0] and experiment_type not in (
        "per_layer",
        "per_layer_three_metrics",
    ):
        raise ValueError(
            "ei can only be specified if the experiment type is per_layer or per_layer_three_metrics"
        )

    if experiment_type == "per_layer":
        for model_name in model_names:
            for probing_task in probing_tasks:
                for extra_iters in extra_iter_nums:
                    if extra_iters == 0:
                        languages_to_calculate: list[str] = languages
                    else:
                        # We replace the languages by the permutations of lanugages, in merged string form
                        languages_to_calculate = [
                            get_language_merged_string(lp)
                            for lp in get_language_pair_permutations(languages)
                        ]

                    language_pairs: list[tuple[str, str]] = (
                        get_language_pair_combinations(languages_to_calculate)
                    )

                    sims: dict[int, dict[int, dict[Any, float]]] = (
                        calculate_per_layer_sims_between_langs(
                            model_name,
                            probing_task,
                            languages_to_calculate,
                            extra_prints,
                            per_class,
                            sim_func=sim_func,
                            extra_iters=extra_iters,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )

                    print(sims)

                    metric_name: str = get_similarity_metric_name(sim_func)

                    if sim_func == "cos_sim" or sim_func == "maha_cos_sim":
                        # vmin, vmax = 0.0, 1.0
                        vmin, vmax = -1.0, 1.0
                    elif sim_func == "l2_dist":
                        vmin, vmax = get_similarity_range(sims)
                    else:
                        raise ValueError(f"Unknown sim_func ({sim_func})")

                    # for value in [
                    #     val
                    #     for d1 in sims.values()
                    #     for d2 in d1.values()
                    #     for val in d2.values()
                    # ]:
                    #     if value < 0.0:
                    #         print(f"Found negative similarity: {value}")

                    # for layer_num in list(sims.keys())[::10]:
                    #     plot_sim_confusion_matrix(
                    #         sims,
                    #         layer_num,
                    #         f"{metric_name} comparison of {model_name} {probing_task} {probe_type} probes at layer {layer_num} refitted for {extra_iters} iterations with per_class={per_class}",
                    #         save,
                    #         show,
                    #         per_class,
                    #         sim_func,
                    #         vmin=vmin,
                    #         vmax=vmax,
                    #     )

                    plot_sim_over_the_layers(
                        sims,
                        language_pairs,
                        f"{metric_name} over layers for {model_name} {probing_task} {probe_type} probes of different language pairs refitted for {extra_iters} iterations with per_class={per_class}{zeroing_suffix}",
                        save,
                        show,
                        per_class,
                        sim_func,
                        vmin=vmin,
                        vmax=vmax,
                    )
    elif experiment_type == "per_layer_three_metrics":
        if "-sf" in sys.argv:
            print(
                "Warning: -sf argument is not used in per_layer_three_metrics experiment "
                "and will be ignored. Both cos_sim and l2_dist are always computed."
            )

        for model_name in model_names:
            for probing_task in probing_tasks:
                for extra_iters in extra_iter_nums:
                    if extra_iters == 0:
                        languages_to_calculate: list[str] = languages
                    else:
                        languages_to_calculate = [
                            get_language_merged_string(lp)
                            for lp in get_language_pair_permutations(languages)
                        ]

                    language_pairs: list[tuple[str, str]] = (
                        get_language_pair_combinations(languages_to_calculate)
                    )

                    sims_cos: dict[int, dict[int, dict[Any, float]]] = (
                        calculate_per_layer_sims_between_langs(
                            model_name,
                            probing_task,
                            languages_to_calculate,
                            extra_prints,
                            per_class,
                            sim_func="cos_sim",
                            extra_iters=extra_iters,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )
                    sims_l2: dict[int, dict[int, dict[Any, float]]] = (
                        calculate_per_layer_sims_between_langs(
                            model_name,
                            probing_task,
                            languages_to_calculate,
                            extra_prints,
                            per_class,
                            sim_func="l2_dist",
                            extra_iters=extra_iters,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )
                    sims_maha: dict[int, dict[int, dict[Any, float]]] = (
                        calculate_per_layer_sims_between_langs(
                            model_name,
                            probing_task,
                            languages_to_calculate,
                            extra_prints,
                            per_class,
                            sim_func="maha_cos_sim",
                            extra_iters=extra_iters,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )

                    plot_sim_over_the_layers_three_metrics(
                        sims_cos,
                        sims_l2,
                        sims_maha,
                        language_pairs,
                        f"Cosine, Mahalanobis cosine, and Euclidean distance over layers for {model_name} {probing_task} {probe_type} probes of different language pairs refitted for {extra_iters} iterations with per_class={per_class}{zeroing_suffix}",
                        save,
                        show,
                        per_class,
                    )
    elif experiment_type == "per_extra_iter":
        import pandas as pd

        num_refits = 2
        iterations_per_refit = 1000
        for probing_task in probing_tasks:
            language_pairs: list[tuple[str, str]] = get_language_pair_permutations(
                languages
            )
            # {model_name: {lang_pair_str: {class_num: avg_sim_at_max_iters}}}
            table_data: dict[str, dict[str, dict[int, float]]] = {
                mn: {} for mn in model_names
            }

            sims_per_model_per_pair: dict[
                str, dict[str, dict[int, dict[int, dict[Any, float]]]]
            ] = {}
            for model_name in model_names:
                sims_per_model_per_pair[model_name] = {}
                for language_pair in language_pairs:
                    sims: dict[int, dict[int, dict[Any, float]]] = (
                        calculate_per_layer_sims_over_extra_iters(
                            model_name,
                            probing_task,
                            language_pair,
                            num_refits,
                            iterations_per_refit,
                            extra_prints,
                            per_class,
                            sim_func,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )

                    if sim_func == "cos_sim" or sim_func == "maha_cos_sim":
                        vmin, vmax = get_similarity_range(sims)
                    elif sim_func == "l2_dist":
                        vmin, vmax = get_similarity_range(sims)
                    else:
                        raise ValueError(f"Unknown sim_func ({sim_func})")

                    metric_name = get_similarity_metric_name(sim_func)
                    layer_nums_to_plot: list[int] = list(sims.keys())[::4]
                    # if save or show:
                    #     plot_sim_over_extra_iters(
                    #         sims,
                    #         layer_nums_to_plot,
                    #         f"{metric_name} over extra iters for {probe_type} {probing_task} probes of {model_name} on the {probing_task} {language_pair} task at different layers with per_class={per_class}{zeroing_suffix}",
                    #         save,
                    #         show,
                    #         per_class,
                    #         sim_func,
                    #         vmin=vmin,
                    #         vmax=vmax,
                    #     )

                    # Accumulate similarity at max extra_iters, averaged across layers
                    lang_pair_str = f"{language_pair[0]}→{language_pair[1]}"
                    sims_per_model_per_pair[model_name][lang_pair_str] = sims
                    first_layer = next(iter(sims))
                    class_nums_in_sims = sorted(sims[first_layer].keys())
                    max_iters = max(sims[first_layer][class_nums_in_sims[0]].keys())
                    table_data[model_name][lang_pair_str] = {
                        class_num: float(
                            np.mean(
                                [
                                    sims[layer_num][class_num][max_iters]
                                    for layer_num in sims
                                ]
                            )
                        )
                        for class_num in class_nums_in_sims
                    }

            if sims_per_model_per_pair and (save or show):
                max_extra_iters = num_refits * iterations_per_refit

                metric_name_new = get_similarity_metric_name(sim_func)
                plot_sim_over_layers_at_max_iters(
                    sims_per_model_per_pair,
                    max_extra_iters,
                    f"{metric_name_new} after {max_extra_iters} iters over layers for {probing_task} {probe_type} probes with per_class={per_class}{zeroing_suffix}",
                    save,
                    show,
                    sim_func,
                    per_language_pair=False,
                )

            # Build and print one DataFrame per class
            lang_pair_strs = [f"{lp[0]}→{lp[1]}" for lp in language_pairs]
            thesis_names = [MODEL_THESIS_NAMES[mn] for mn in model_names]
            class_nums_for_df = sorted(
                table_data[model_names[0]][lang_pair_strs[0]].keys()
            )
            for class_num in class_nums_for_df:
                class_name = get_class_name(class_num, per_class)
                df = pd.DataFrame(
                    {
                        lps: [table_data[mn][lps][class_num] for mn in model_names]
                        for lps in lang_pair_strs
                    },
                    index=thesis_names,
                )
                col_fmt = "l" + "r" * len(lang_pair_strs)
                latex_str = df.to_latex(float_format="%.4f", column_format=col_fmt)  # type: ignore[call-overload]
                latex_str = "\\resizebox{\\textwidth}{!}{\n" + latex_str + "}"
                print(f"\n--- {probing_task} | {class_name} ---")
                print(latex_str)
    elif experiment_type == "between_layers":
        for model_name in model_names:
            for language in languages:
                for probing_task in probing_tasks:
                    sims_between: dict[int, dict[str, float]] = (
                        calculate_between_layers_sims(
                            model_name,
                            probing_task,
                            language,
                            per_class,
                            sim_func,
                            probe_type=probe_type,
                            zeroed_out_activation_dims=zeroed_out_activation_dims,
                            zeroed_out_weight_dims=zeroed_out_weight_dims,
                        )
                    )

                    metric_name = get_similarity_metric_name(sim_func)

                    all_values: list[float] = [
                        v for d in sims_between.values() for v in d.values()
                    ]
                    if sim_func == "cos_sim" or sim_func == "maha_cos_sim":
                        # vmin, vmax = 0.0, 1.0
                        vmin, vmax = -1.0, 1.0
                    else:
                        vmin, vmax = float(min(all_values)), float(max(all_values))

                    # for value in all_values:
                    #     if value < 0.0:
                    #         print(f"Found negative similarity: {value}")

                    plot_between_layers_confusion_matrix(
                        sims_between,
                        f"{metric_name} between layers for {model_name} {language} {probing_task} {probe_type} probes with per_class={per_class}{zeroing_suffix}",
                        save,
                        show,
                        per_class,
                        sim_func,
                        vmin=vmin,
                        vmax=vmax,
                    )
    elif experiment_type == "weight_magnitudes":
        for language in languages:
            for model_name in model_names:
                layer_nums: int = get_number_of_layers_from_file(model_name)
                for layer_num in range(layer_nums):
                    for probing_task in probing_tasks:
                        for extra_iters in extra_iter_nums:
                            plot_probe_weight_magnitudes(
                                model_name,
                                language,
                                layer_num,
                                probing_task,
                                probe_type,
                                extra_iters,
                                zeroed_out_activation_dims=zeroed_out_activation_dims,
                                zeroed_out_weight_dims=zeroed_out_weight_dims,
                            )
    else:
        raise ValueError(
            f"{experiment_type} invalid. exp must be per_layer, per_layer_three_metrics, per_extra_iter, or between_layers."
        )
