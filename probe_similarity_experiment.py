from typing import Any
import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from probes import LRProbe, load_probe
from utils import (
    LANGUAGES,
    MODEL_NAMES,
    PLOTS_FOLDER,
    REVERSE_LABEL_MAP,
    get_language_pair_combinations,
    get_number_of_layers_from_file,
    get_language_merged_string,
    get_language_pair_permutations,
)

# Create plots directory if it doesn't exist
Path(PLOTS_FOLDER).mkdir(exist_ok=True)


def get_similarity_function(probe: LRProbe, sim_func: str):
    """
    Return the appropriate similarity function based on sim_func parameter.

    Args:
        probe: The LRProbe instance
        sim_func: Either "cos_sim" or "l2_dist"

    Returns:
        The corresponding method (calculate_cosine_similarity or calculate_l2_dist)
    """
    if sim_func == "cos_sim":
        return probe.calculate_cosine_similarity
    elif sim_func == "l2_dist":
        return probe.calculate_l2_dist
    else:
        raise ValueError(
            f"Unknown similarity function: {sim_func}. Must be 'cos_sim' or 'l2_dist'"
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
    else:
        raise ValueError(
            f"Unknown similarity function: {sim_func}. Must be 'cos_sim' or 'l2_dist'"
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
) -> dict[int, dict[int, dict[Any, float]]]:
    sims: dict[int, dict[int, dict[Any, float]]] = {}

    language_pairs: list[tuple[str, str]] = get_language_pair_permutations(languages)
    print(language_pairs)
    num_layers: int = get_number_of_layers_from_file(model_name)
    for layer_num in range(num_layers):
        sims[layer_num] = {}
        for language_a, language_b in language_pairs:
            probe_a: LRProbe = load_probe(
                language_a,
                layer_num,
                probing_task,
                "lr",
                model_name,
                extra_iters=extra_iters,
            )
            probe_b: LRProbe = load_probe(
                language_b,
                layer_num,
                probing_task,
                "lr",
                model_name,
                extra_iters=extra_iters,
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
) -> dict[int, dict[int, dict[Any, float]]]:
    sims: dict[int, dict[int, dict[Any, float]]] = {}

    num_layers: int = get_number_of_layers_from_file(
        "olmo_model"
    )  # TODO change this to model_name

    for layer_num in range(num_layers):
        sims[layer_num] = {}

        # Get original probe trained on language a
        original_probe: LRProbe = load_probe(
            language_pair[0], layer_num, probing_task, "lr", model_name, 0
        )

        for refit_num in range(1, num_refits + 1):
            extra_iters: int = refit_num * iterations_per_refit

            # Get probe refitted on language b for a certain number of extra iterations
            refitted_probe: LRProbe = load_probe(
                get_language_merged_string(language_pair),
                layer_num,
                probing_task,
                "lr",
                model_name,
                extra_iters,
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
        matrix = np.ones((n, n))

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
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(PLOTS_FOLDER, filename)
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
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(PLOTS_FOLDER, filename)
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
) -> dict[int, dict[str, float]]:
    """
    Calculate similarities between probes at each pair of layers for a single language.
    Returns: {class_num: {f"{layer_a},{layer_b}": similarity}}
    """
    num_layers: int = get_number_of_layers_from_file(model_name)

    probes: dict[int, LRProbe] = {
        layer_num: load_probe(language, layer_num, probing_task, "lr", model_name)
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
        matrix = np.ones((n, n))

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
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(PLOTS_FOLDER, filename)
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
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(PLOTS_FOLDER, filename)
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
        help="enter the experiment to perform: per_layer, per_extra_iter, or between_layers",
        default="per_extra_iter",
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
        choices=["cos_sim", "l2_dist"],
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

    if extra_iter_nums != [0] and experiment_type != "per_layer":
        raise ValueError("ei can only be specified if the experiment type is per_layer")

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
                        )
                    )

                    print(sims)

                    metric_name: str = get_similarity_metric_name(sim_func)

                    if sim_func == "cos_sim":
                        vmin, vmax = 0.0, 1.0
                    elif sim_func == "l2_dist":
                        vmin, vmax = get_similarity_range(sims)
                    else:
                        raise ValueError(f"Unknown sim_func ({sim_func})")

                    for value in [
                        val
                        for d1 in sims.values()
                        for d2 in d1.values()
                        for val in d2.values()
                    ]:
                        if value < 0.0:
                            print(f"Found negative similarity: {value}")

                    for layer_num in list(sims.keys())[::10]:
                        plot_sim_confusion_matrix(
                            sims,
                            layer_num,
                            f"{metric_name} comparison of {model_name} probes at layer {layer_num} refitted for {extra_iters} iterations with per_class={per_class}",
                            save,
                            show,
                            per_class,
                            sim_func,
                            vmin=vmin,
                            vmax=vmax,
                        )

                    plot_sim_over_the_layers(
                        sims,
                        language_pairs,
                        f"{metric_name} over layers for {model_name} probes of different language pairs refitted for {extra_iters} iterations with per_class={per_class}",
                        save,
                        show,
                        per_class,
                        sim_func,
                        vmin=vmin,
                        vmax=vmax,
                    )
    elif experiment_type == "per_extra_iter":
        num_refits = 2
        iterations_per_refit = 1000
        for model_name in model_names:
            for probing_task in probing_tasks:
                language_pairs: list[tuple[str, str]] = get_language_pair_permutations(
                    languages
                )

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
                        )
                    )

                    # print(sims)

                    # print(sims)

                    if sim_func == "cos_sim":
                        # vmin, vmax = 0.99, 1.01
                        vmin, vmax = get_similarity_range(sims)
                    elif sim_func == "l2_dist":
                        vmin, vmax = get_similarity_range(sims)
                    else:
                        raise ValueError(f"Unknown sim_func ({sim_func})")

                    for value in [
                        val
                        for d1 in sims.values()
                        for d2 in d1.values()
                        for val in d2.values()
                    ]:
                        if value < 0.0:
                            print(f"Found negative similarity: {value}")

                    metric_name = get_similarity_metric_name(sim_func)
                    layer_nums_to_plot: list[int] = list(sims.keys())[::4]
                    plot_sim_over_extra_iters(
                        sims,
                        layer_nums_to_plot,
                        f"{metric_name} over extra iters for probes of {model_name} on the {probing_task} {language_pair} task at different layers with per_class={per_class}",
                        save,
                        show,
                        per_class,
                        sim_func,
                        vmin=vmin,
                        vmax=vmax,
                    )
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
                        )
                    )

                    metric_name = get_similarity_metric_name(sim_func)

                    all_values: list[float] = [
                        v for d in sims_between.values() for v in d.values()
                    ]
                    if sim_func == "cos_sim":
                        vmin, vmax = -0.15, 1.0
                    else:
                        vmin, vmax = float(min(all_values)), float(max(all_values))

                    for value in all_values:
                        if value < 0.0:
                            print(f"Found negative similarity: {value}")

                    plot_between_layers_confusion_matrix(
                        sims_between,
                        f"{metric_name} between layers for {model_name} {language} {probing_task} probes with per_class={per_class}",
                        save,
                        show,
                        per_class,
                        sim_func,
                        vmin=vmin,
                        vmax=vmax,
                    )
    else:
        raise ValueError(
            f"{experiment_type} invalid. exp must be per_layer, per_extra_iter, or between_layers."
        )
