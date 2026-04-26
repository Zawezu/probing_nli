from probes import LRProbe, get_probe
import argparse
from itertools import permutations, combinations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import LANGUAGES, MODEL_NAMES, get_number_of_layers_from_file


def calculate_per_layer_cos_sims_between_langs(
    model_name: str, probing_task: str, languages: list[str]
) -> dict[int, dict[str, float]]:
    cos_sims_per_layer: dict[int, dict[str, float]] = {}

    language_pairs: list[tuple[str, str]] = list(permutations(languages, 2))
    print(language_pairs)
    num_layers: int = get_number_of_layers_from_file(model_name)
    for layer_num in range(num_layers):
        cos_sims_per_layer[layer_num] = {}
        for language_a, language_b in language_pairs:
            probe_a: LRProbe = get_probe(
                language_a, layer_num, probing_task, "lr", model_name
            )
            probe_b: LRProbe = get_probe(
                language_b, layer_num, probing_task, "lr", model_name
            )

            cos_sim: float = probe_a.calculate_cosine_similarity(probe_b)
            cos_sims_per_layer[layer_num][f"{language_a},{language_b}"] = cos_sim

    return cos_sims_per_layer


def plot_cos_sim_confusion_matrix(
    cos_sims_per_layer: dict[int, dict[str, float]], layer_num: int
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
        xticklabels=languages,
        yticklabels=languages,
        cmap="Blues",
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Cosine Similarity"},
    )
    plt.title(f"Cosine similarity confusion matrix of probes at layer {layer_num}")
    plt.xlabel("Second Language")
    plt.ylabel("First Language")
    plt.tight_layout()
    plt.show()


def plot_cos_sim_over_the_layers(cos_sims_per_layer, language_pairs) -> None:
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
        "Cosine similarity over layers for probes of different language pairs",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-models", help="enter the model names", nargs="*", default=MODEL_NAMES
    )
    parser.add_argument(
        "-langs", help="enter the languages", nargs="*", default=LANGUAGES
    )
    parser.add_argument(
        "-tasks", help="enter the probing tasks", nargs="*", default=["standard"]
    )

    args: argparse.Namespace = parser.parse_args()
    print(args)

    model_names: list[str] = args.models
    languages: list[str] = args.langs
    probing_tasks: list[str] = args.tasks

    for model_name in model_names:
        for probing_task in probing_tasks:
            cos_sims_per_layer: dict[int, dict[str, float]] = (
                calculate_per_layer_cos_sims_between_langs(
                    model_name, probing_task, languages
                )
            )

            print(cos_sims_per_layer)

            for layer_num in list(cos_sims_per_layer.keys())[::10]:
                plot_cos_sim_confusion_matrix(cos_sims_per_layer, layer_num)

            language_pairs: list[tuple[str, str]] = list(combinations(languages, 2))
            plot_cos_sim_over_the_layers(cos_sims_per_layer, language_pairs)
