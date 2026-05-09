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
    get_language_pair_combinations,
    get_number_of_layers_from_file,
    get_language_merged_string,
    get_language_pair_permutations,
)

# Create plots directory if it doesn't exist
Path(PLOTS_FOLDER).mkdir(exist_ok=True)


def print_highest_values_in_probe_coeficcients(probe, n=30):
    with np.printoptions(threshold=sys.maxsize):
        coef: np.ndarray = probe.get_vector()
        idxs = np.argsort(coef)
        highest_values = coef[idxs][-n:]

        print(f"-------------\n{n} highest values of the coefficient of {probe}:")
        print(highest_values)


def calculate_per_layer_cos_sims_between_langs(
    model_name: str, probing_task: str, languages: list[str], extra_iters: int = 0
) -> dict[int, dict[str, float]]:
    cos_sims_per_layer: dict[int, dict[str, float]] = {}

    language_pairs: list[tuple[str, str]] = get_language_pair_permutations(languages)
    print(language_pairs)
    num_layers: int = get_number_of_layers_from_file(model_name)
    for layer_num in range(num_layers):
        cos_sims_per_layer[layer_num] = {}
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

            cos_sim: float = probe_a.calculate_cosine_similarity(probe_b)
            cos_sims_per_layer[layer_num][f"{language_a},{language_b}"] = cos_sim

            print_highest_values_in_probe_coeficcients(probe_a)
            print_highest_values_in_probe_coeficcients(probe_b)

    return cos_sims_per_layer


def calculate_per_layer_cos_sims_over_extra_iters(
    model_name: str,
    probing_task: str,
    language_pair: tuple[str, str],
    num_refits: int,
    iterations_per_refit: int,
):
    cos_sims_per_extra_iters: dict[int, dict[Any, float]] = {}

    num_layers: int = get_number_of_layers_from_file(
        "olmo_model"
    )  # TODO change this to model_name

    for layer_num in range(num_layers):
        cos_sims_per_extra_iters[layer_num] = {}

        # Get original probe trained on language a
        original_probe: LRProbe = load_probe(
            language_pair[0], layer_num, probing_task, "lr", model_name, 0
        )

        for refit_num in range(1, num_refits):
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

            print_highest_values_in_probe_coeficcients(original_probe)
            print_highest_values_in_probe_coeficcients(refitted_probe)

            cos_sim: float = original_probe.calculate_cosine_similarity(refitted_probe)
            cos_sims_per_extra_iters[layer_num][extra_iters] = cos_sim

    return cos_sims_per_extra_iters


def plot_cos_sim_confusion_matrix(
    cos_sims_per_layer: dict[int, dict[Any, float]],
    layer_num: int,
    title: str,
    save: bool,
    show: bool,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    """
    Plot a confusion matrix of cosine similarities between language pairs.
    """
    cos_sims: dict[str, float] = cos_sims_per_layer[layer_num]
    # Extract unique languages from the keys
    language_pairs: list[str] = list(cos_sims.keys())
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

    for pair, value in cos_sims.items():
        source, target = pair.split(",")
        i: int = languages.index(source)
        j: int = languages.index(target)
        matrix[i, j] = value

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        xticklabels=languages,  # type: ignore
        yticklabels=languages,  # type: ignore
        cmap="Blues",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Cosine Similarity"},
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(title)
    plt.xlabel("Second Language")
    plt.ylabel("First Language")
    plt.tight_layout()

    if save:
        # Save plot
        filename = title.replace(" ", "_").replace("/", "_") + ".png"
        filepath = os.path.join(PLOTS_FOLDER, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")

    if show:
        plt.show()


def plot_cos_sim_over_the_layers(
    cos_sims_per_layer, language_pairs, title: str, save: bool, show: bool
) -> None:
    """
    Plot cosine similarity over layers for each language pair.

    Args:
        cos_sims_per_layer: Dictionary mapping layer number to dict of language pairs and their cosine similarities
        language_pairs: List of tuples containing (source_lang, target_lang) pairs
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
    layers: list[int] = sorted(cos_sims_per_layer.keys())

    # Plot each language pair
    for idx, (lang_a, lang_b) in enumerate(language_pairs):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax = axes[row, col]

        # Extract cosine similarities for this language pair across layers
        pair_key: str = f"{lang_a},{lang_b}"
        cos_sims: list[float] = [
            cos_sims_per_layer[layer][pair_key] for layer in layers
        ]

        ax.plot(layers, cos_sims, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"{lang_a}, {lang_b}")
        ax.grid(True, alpha=0.3)

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


def plot_cos_sim_over_extra_iters(
    cos_sims_per_extra_iters: dict[int, dict[int, float]],
    layer_nums_to_plot: list[int],
    title: str,
    save: bool,
    show: bool,
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

    # Plot each language pair
    for idx, layer_num in enumerate(layer_nums_to_plot):
        row: int = idx // n_cols
        col: int = idx % n_cols
        ax = axes[row, col]

        # Extract cosine similarities for this language pair across layers
        extra_iters: list[int] = list(cos_sims_per_extra_iters[layer_num].keys())
        cos_sims: list[float] = list(cos_sims_per_extra_iters[layer_num].values())

        ax.plot(extra_iters, cos_sims, marker="o", linewidth=2, markersize=6)
        ax.set_xlabel("Extra iters", fontsize=8)
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Layer {layer_num}", fontsize=8)
        ax.grid(True, alpha=0.3)

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
        help="enter the experiment to perform: per_layer or per_extra_iter",
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

    args: argparse.Namespace = parser.parse_args()
    print(args)

    model_names: list[str] = args.m
    languages: list[str] = args.l
    probing_tasks: list[str] = args.t
    extra_iter_nums: list[int] = [int(ei) for ei in args.ei]
    experiment_type: str = args.e
    save: bool = args.sv.lower() == "true"
    show: bool = args.sh.lower() == "true"

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

                    cos_sims_per_layer: dict[int, dict[str, float]] = (
                        calculate_per_layer_cos_sims_between_langs(
                            model_name,
                            probing_task,
                            languages_to_calculate,
                            extra_iters=extra_iters,
                        )
                    )

                    print(cos_sims_per_layer)

                    for layer_num in list(cos_sims_per_layer.keys())[::10]:
                        plot_cos_sim_confusion_matrix(
                            cos_sims_per_layer,
                            layer_num,
                            f"Cosine similarity comparison of {model_name} probes at layer {layer_num} refitted for {extra_iters} iterations",
                            save,
                            show,
                        )

                    plot_cos_sim_over_the_layers(
                        cos_sims_per_layer,
                        language_pairs,
                        f"Cosine similarity over layers for {model_name} probes of different language pairs refitted for {extra_iters} iterations",
                        save,
                        show,
                    )
    elif experiment_type == "per_extra_iter":
        num_refits = 5
        iterations_per_refit = 1
        for model_name in model_names:
            for probing_task in probing_tasks:
                language_pairs: list[tuple[str, str]] = get_language_pair_permutations(
                    languages
                )

                for language_pair in language_pairs:
                    cos_sims_per_extra_iters: dict[int, dict[int, float]] = (
                        calculate_per_layer_cos_sims_over_extra_iters(
                            model_name,
                            probing_task,
                            language_pair,
                            num_refits,
                            iterations_per_refit,
                        )
                    )

                    print(cos_sims_per_extra_iters)
                    layer_nums_to_plot: list[int] = list(
                        cos_sims_per_extra_iters.keys()
                    )[::4]
                    plot_cos_sim_over_extra_iters(
                        cos_sims_per_extra_iters,
                        layer_nums_to_plot,
                        f"Cosine similarity over extra iters for probes of {model_name} on the {probing_task} {language_pair} task at different layers",
                        save,
                        show,
                    )

    else:
        raise ValueError(
            f"{experiment_type} invalid. exp must be either per_layer or per_extra_iter."
        )
