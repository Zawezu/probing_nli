import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.pylab import Generator
import numpy as np
from sklearn.metrics import f1_score
import sys

from experiment_common_code import ExperimentResult
from utils import (
    LANGUAGES,
    PLOTS_FOLDER,
    SIGNIFICANCE_RESULTS_FOLDER,
    get_language_merged_string,
    get_language_pair_combinations,
    get_language_pair_permutations,
)


def macro_f1_metric(real_labels: list[int], pred_labels: list[int]) -> float:
    """Macro-averaged F1 score. Classes with no support are skipped (zero_division=0)."""
    return float(f1_score(real_labels, pred_labels, average="macro", zero_division=0))


def accuracy_metric(real_labels: list[int], pred_labels: list[int]) -> float:
    """Standard accuracy."""
    if not real_labels:
        return 0.0
    return sum(r == p for r, p in zip(real_labels, pred_labels)) / len(real_labels)


def _parse_idxs_per_cm_cell(
    idxs_per_cm_cell: dict[str, set[int]],
) -> dict[int, tuple[int, int]]:
    """
    Flatten idxs_per_cm_cell into {idx: (real_label, pred_label)}.

    Input format: {"real:{r},pred:{p}": {idx1, idx2, ...}, ...}
    """
    idx_map: dict[int, tuple[int, int]] = {}
    for key, idxs in idxs_per_cm_cell.items():
        parts: list[str] = key.split(",")
        real_label = int(float(parts[0].split(":")[1]))
        pred_label = int(float(parts[1].split(":")[1]))
        for idx in idxs:
            idx_map[idx] = (real_label, pred_label)
    return idx_map


class BootstrapSignificanceTester:
    """
    Paired bootstrap significance tester comparing two ExperimentResults.

    The test is "paired" because the same sampled indices are used for both results on
    every bootstrap iteration. This preserves the correlation between classifiers and
    makes the test more sensitive to genuine performance differences.

    Prerequisite: index i must refer to the same data point in both results. For
    same-split comparisons this holds trivially. For cross-language comparisons it holds
    because SICK is a parallel corpus with aligned train/test splits.

    For experiment 3 (probe_type="model_pred") there is only one "layer" (index 0) since
    the experiment evaluates model output directly rather than per-layer probes. Call
    run() with num_layers=None and it will automatically use get_num_layers()=1.

    When evaluating experiment 3 results with unknown predictions (label -1), pass
    exclude_labels=[-1] so the metric is computed only over the three NLI classes.
    """

    _GROUP_COLORS: list[tuple[str, str]] = [
        ("#56B4E9", "#0072B2"),  # sky blue / blue
        ("#F0C070", "#D55E00"),  # light orange / vermillion
        ("#70C4B4", "#009E73"),  # light teal / bluish green
        ("#D4A0C4", "#CC79A7"),  # light pink / reddish purple
    ]
    _INSIGNIFICANT_COLOR: str = "#AAAAAA"

    def __init__(
        self,
        result_1: ExperimentResult,
        result_2: ExperimentResult,
        split: str,
        n_bootstrap: int = 100,
        seed: int | None = 42,
    ) -> None:
        """
        Args:
            result_1: First experiment result.
            result_2: Second experiment result.
            split: Data split to evaluate on (e.g. "test", "test_a", "test_b").
            n_bootstrap: Number of bootstrap samples.
            seed: Random seed for reproducibility.
        """
        self.result_1: ExperimentResult = result_1
        self.result_2: ExperimentResult = result_2
        self.split: str = split
        self.n_bootstrap: int = n_bootstrap
        self.rng: Generator = np.random.default_rng(seed)

    def _get_idx_map(
        self, result: ExperimentResult, layer_num: int
    ) -> dict[int, tuple[int, int]]:
        idxs_per_cm_cell = result.get_metric(self.split, "idxs_per_cm_cell", layer_num)
        return _parse_idxs_per_cm_cell(idxs_per_cm_cell)

    def run_layer(
        self,
        layer_num: int,
        metric_fn: Callable[[list[int], list[int]], float],
        alpha,
        exclude_labels: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Run bootstrap significance test for one layer.

        Args:
            layer_num: Layer index. Use 0 for experiment 3.
            metric_fn: (real_labels, pred_labels) -> float performance metric.
            exclude_labels: Real labels to exclude (e.g. [-1] for unknown in exp 3).
            confidence: Confidence level for the bootstrap interval, e.g. 0.95.

        Returns:
            Dict with keys:
              layer, n_common                          -- metadata
              observed_1, observed_2, observed_diff    -- metric on full data
              metric_1_samples, metric_2_samples       -- bootstrap arrays (shape n_bootstrap)
              diff_samples                             -- bootstrap differences (1 - 2)
              ci_1, ci_2, ci_diff                      -- (lo, hi) confidence intervals
              p_value                                  -- two-tailed bootstrap p-value
              confidence                               -- confidence level used
        """
        idx_map_1 = self._get_idx_map(self.result_1, layer_num)
        idx_map_2 = self._get_idx_map(self.result_2, layer_num)

        if set(idx_map_1.keys()) != set(idx_map_1.keys()):
            raise ValueError(
                f"The results have different indices:\n{idx_map_1}\n{idx_map_2}"
            )

        common_idxs: list[int] = list(idx_map_1.keys() & idx_map_2.keys())

        if exclude_labels is not None:
            exclude_set: set[int] = set(exclude_labels)
            common_idxs = [i for i in common_idxs if idx_map_1[i][0] not in exclude_set]

        n: int = len(common_idxs)
        if n == 0:
            raise ValueError(
                f"No common indices for split='{self.split}', layer={layer_num} "
                "after applying exclude_labels."
            )

        # Pre-extract arrays so the bootstrap loop is fast
        real_1 = np.array([idx_map_1[i][0] for i in common_idxs])
        pred_1 = np.array([idx_map_1[i][1] for i in common_idxs])
        real_2 = np.array([idx_map_2[i][0] for i in common_idxs])
        pred_2 = np.array([idx_map_2[i][1] for i in common_idxs])

        metric_1_samples = np.empty(self.n_bootstrap)
        metric_2_samples = np.empty(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Same positions for both -- this is what makes the test paired
            sample_idxs = self.rng.integers(0, n, size=n)
            metric_1_samples[b] = metric_fn(
                real_1[sample_idxs].tolist(), pred_1[sample_idxs].tolist()
            )
            metric_2_samples[b] = metric_fn(
                real_2[sample_idxs].tolist(), pred_2[sample_idxs].tolist()
            )

        diff_samples = metric_1_samples - metric_2_samples

        lo_pct, hi_pct = 100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)

        observed_1 = float(metric_fn(real_1.tolist(), pred_1.tolist()))
        observed_2 = float(metric_fn(real_2.tolist(), pred_2.tolist()))

        # p_raw = 2.0 * float(min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0)))
        # p_value = max(p_raw, 2.0 / self.n_bootstrap)

        p_value: float = 2.0 * float(
            min(np.mean(diff_samples <= 0), np.mean(diff_samples >= 0))
        )

        return {
            "layer": layer_num,
            "n_common": n,
            "observed_1": observed_1,
            "observed_2": observed_2,
            "observed_diff": observed_1 - observed_2,
            "metric_1_samples": metric_1_samples,
            "metric_2_samples": metric_2_samples,
            "diff_samples": diff_samples,
            "ci_1": (
                float(np.percentile(metric_1_samples, lo_pct)),
                float(np.percentile(metric_1_samples, hi_pct)),
            ),
            "ci_2": (
                float(np.percentile(metric_2_samples, lo_pct)),
                float(np.percentile(metric_2_samples, hi_pct)),
            ),
            "ci_diff": (
                float(np.percentile(diff_samples, lo_pct)),
                float(np.percentile(diff_samples, hi_pct)),
            ),
            "p_value": p_value,
            "confidence": 1 - alpha,
        }

    def run(
        self,
        metric_fn: Callable[[list[int], list[int]], float],
        num_layers: int | None = None,
        exclude_labels: list[int] | None = None,
        alpha: float = 0.05,
        save: bool = False,
        test_name: str = "",
    ) -> list[dict[str, Any]]:
        """
        Run bootstrap test for all layers.

        Args:
            metric_fn: Performance metric function.
            num_layers: Test layers 0..num_layers-1. If None, use result_1.get_num_layers()
                        (for experiment 3 this equals 1, so the loop runs once at layer 0).
            exclude_labels: Real labels to exclude (e.g. [-1] to drop unknowns in exp 3).
            confidence: Confidence level for bootstrap intervals.
            save: If True, persist results to JSON in SIGNIFICANCE_RESULTS_FOLDER.
            test_name: Filename stem for the saved file (required when save=True).

        Returns:
            List of per-layer result dicts (same keys as run_layer).
        """
        if save and not test_name:
            raise ValueError("test_name must be specified when save=True")

        n_layers = (
            num_layers if num_layers is not None else self.result_1.get_num_layers()
        )
        # Use bonferroni correction, since we are doing a test on many different layers
        alpha_corrected = alpha / n_layers
        layer_results = [
            self.run_layer(layer_num, metric_fn, alpha_corrected, exclude_labels)
            for layer_num in range(n_layers)
        ]

        if save:
            self._save_results(layer_results, test_name)

        return layer_results

    @staticmethod
    def _save_results(layer_results: list[dict[str, Any]], test_name: str) -> None:
        save_dir = Path(SIGNIFICANCE_RESULTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)

        serializable: list[dict[str, Any]] = []
        for r in layer_results:
            entry: dict[str, Any] = {}
            for k, v in r.items():
                entry[k] = v.tolist() if isinstance(v, np.ndarray) else v
            serializable.append(entry)

        fname = test_name if test_name.endswith(".json") else f"{test_name}.json"
        filepath = save_dir / fname
        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"Significance results saved to {filepath}")

    @staticmethod
    def load_results(test_name: str) -> list[dict[str, Any]]:
        """
        Load previously saved results from JSON back into the layer_results format.

        Numpy arrays (metric_1_samples, metric_2_samples, diff_samples) are
        reconstructed from the stored lists. CI tuples are returned as lists,
        which is compatible with all plotting and printing methods.

        Args:
            test_name: The test_name used when saving (with or without .json extension).
        """
        fname = test_name if test_name.endswith(".json") else f"{test_name}.json"
        filepath = Path(SIGNIFICANCE_RESULTS_FOLDER) / fname
        with open(filepath, "r") as f:
            data: list[dict[str, Any]] = json.load(f)

        array_keys = {"metric_1_samples", "metric_2_samples", "diff_samples"}
        for r in data:
            for k in array_keys:
                if k in r:
                    r[k] = np.array(r[k])
        return data

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_summary(
        self,
        layer_results: list[dict[str, Any]],
        label_1: str = "Result 1",
        label_2: str = "Result 2",
        metric_name: str = "metric",
    ) -> None:
        """Print a per-layer table of observed metrics, CIs, differences, and p-values."""
        if not layer_results:
            return

        conf: float = layer_results[0]["confidence"]
        alpha: float = 1.0 - conf
        ci_pct = int(round(conf * 100))

        w = 26
        header = (
            f"{'Layer':>6}  "
            f"{label_1:>{w}}  "
            f"{label_2:>{w}}  "
            f"{'Diff (1-2)':>{w}}  "
            f"{'p-value':>8}  Sig."
        )
        print(f"\n{header}")
        print("-" * (len(header) + 2))

        for r in layer_results:
            sig = "*" if r["p_value"] < alpha else ""
            print(
                f"{r['layer']:>6}  "
                f"{r['observed_1']:.4f} [{r['ci_1'][0]:.4f},{r['ci_1'][1]:.4f}]  "
                f"{r['observed_2']:.4f} [{r['ci_2'][0]:.4f},{r['ci_2'][1]:.4f}]  "
                f"{r['observed_diff']:+.4f} [{r['ci_diff'][0]:+.4f},{r['ci_diff'][1]:+.4f}]  "
                f"{r['p_value']:.4f}  {sig}"
            )

        print(
            f"\n* p < {alpha:.3f} (two-tailed); {ci_pct}% CI shown for {metric_name}."
        )

    def plot_histograms(
        self,
        layer_results: list[dict[str, Any]],
        layers_to_plot: list[int] | None = None,
        label_1: str = "Result 1",
        label_2: str = "Result 2",
        metric_name: str = "metric",
        show: bool = True,
        save: bool = False,
        filename: str = "",
        n_bins: int = 50,
    ) -> None:
        """
        Plot bootstrap distribution histograms for selected layers.

        For each selected layer: two subplots side by side.
        - Left: overlapping bootstrap metric distributions for result_1 and result_2,
                with observed values (dashed lines) and CI spans shaded.
        - Right: bootstrap distribution of the difference (metric_1 - metric_2),
                 with observed difference, CI span, and the zero line.

        Args:
            layers_to_plot: Which layers to include. Defaults to all in layer_results.
                            For models with many layers, pick a representative subset.
            n_bins: Histogram bin count.
        """
        if save and not filename:
            raise ValueError("filename must be specified when save=True")

        if layers_to_plot is None:
            layers_to_plot = list(range(0, len(layer_results), 6))

        selected = [r for r in layer_results if r["layer"] in set(layers_to_plot)]

        if not selected:
            print("No layers to plot.")
            return

        n = len(selected)
        fig, axs = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)

        for i, r in enumerate(selected):
            layer = r["layer"]
            ci_pct = int(round(r["confidence"] * 100))

            # Left: overlapping metric distributions
            ax = axs[i, 0]
            ax.hist(
                r["metric_1_samples"],
                bins=n_bins,
                alpha=0.5,
                density=True,
                label=label_1,
            )
            ax.hist(
                r["metric_2_samples"],
                bins=n_bins,
                alpha=0.5,
                density=True,
                label=label_2,
            )
            ax.axvline(
                r["observed_1"],
                linestyle="--",
                color="tab:blue",
                label=f"{label_1} observed ({r['observed_1']:.4f})",
            )
            ax.axvline(
                r["observed_2"],
                linestyle="--",
                color="tab:orange",
                label=f"{label_2} observed ({r['observed_2']:.4f})",
            )
            ax.axvspan(*r["ci_1"], alpha=0.10, color="tab:blue")
            ax.axvspan(*r["ci_2"], alpha=0.10, color="tab:orange")
            ax.set_title(
                f"Layer {layer}: bootstrap {metric_name} distributions  (p={r['p_value']:.4f})"
            )
            ax.set_xlabel(metric_name)
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Right: difference distribution
            ax = axs[i, 1]
            ax.hist(
                r["diff_samples"],
                bins=n_bins,
                alpha=0.7,
                density=True,
                color="mediumpurple",
            )
            ax.axvline(
                r["observed_diff"],
                color="red",
                linestyle="--",
                label=f"Observed diff ({r['observed_diff']:+.4f})",
            )
            ax.axvline(0, color="black", alpha=0.6, label="No difference")
            ax.axvspan(
                *r["ci_diff"],
                alpha=0.15,
                color="mediumpurple",
                label=f"{ci_pct}% CI [{r['ci_diff'][0]:+.4f}, {r['ci_diff'][1]:+.4f}]",
            )
            ax.set_title(f"Layer {layer}: difference (1-2)  (p={r['p_value']:.4f})")
            ax.set_xlabel(f"Δ {metric_name}")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout()
        self._maybe_save(fig, save, filename)
        if show:
            plt.show()

    @staticmethod
    def plot_metric_over_layers(
        layer_results_groups: list[list[dict[str, Any]]],
        label_1: str = "Result 1",
        label_2: str = "Result 2",
        group_labels: list[str] | None = None,
        metric_name: str = "Macro F1",
        model_name: str = "",
        extra_iter_num: int = 0,
        y_range: tuple[float, float] | None = (0.0, 1.0),
        show: bool = True,
        save: bool = False,
        filename: str = "",
    ) -> None:
        """
        Plot per-layer CI error bars for result_1 and result_2 across one or more groups.

        Each group uses a colorblind-friendly light/dark color pair. Dots are drawn in the
        group color when the layer is significant (p < alpha) and grey otherwise.

        For a single group pass layer_results_groups=[layer_results].
        For multiple groups with the same label_1/label_2 semantics, pass a list of groups
        and optionally supply group_labels to name each group in the legend.
        """
        if save and not filename:
            raise ValueError("filename must be specified when save=True")

        n_bootstrap = len(layer_results_groups[0][0]["metric_1_samples"])
        conf = layer_results_groups[0][0]["confidence"]
        alpha = 1.0 - conf
        ci_pct = int(round(conf * 100))
        n_layers = len(layer_results_groups[0])
        n_groups = len(layer_results_groups)

        fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.5 + 4), 5))

        for g_idx, layer_results in enumerate(layer_results_groups):
            color_light, color_dark = BootstrapSignificanceTester._GROUP_COLORS[
                g_idx % len(BootstrapSignificanceTester._GROUP_COLORS)
            ]

            layers = [r["layer"] for r in layer_results]
            obs_1 = [r["observed_1"] for r in layer_results]
            obs_2 = [r["observed_2"] for r in layer_results]

            err_1 = (
                [o - r["ci_1"][0] for o, r in zip(obs_1, layer_results)],
                [r["ci_1"][1] - o for o, r in zip(obs_1, layer_results)],
            )
            err_2 = (
                [o - r["ci_2"][0] for o, r in zip(obs_2, layer_results)],
                [r["ci_2"][1] - o for o, r in zip(obs_2, layer_results)],
            )

            if group_labels is not None:
                leg_1 = f"{label_1} ({group_labels[g_idx]})"
                leg_2 = f"{label_2} ({group_labels[g_idx]})"
            elif n_groups == 1:
                leg_1, leg_2 = label_1, label_2
            else:
                leg_1 = f"{label_1} (group {g_idx})"
                leg_2 = f"{label_2} (group {g_idx})"

            ax.errorbar(
                layers,
                obs_1,
                yerr=err_1,
                fmt="-",
                color=color_light,
                capsize=4,
                label=leg_1,
            )
            ax.errorbar(
                layers,
                obs_2,
                yerr=err_2,
                fmt="--",
                color=color_dark,
                capsize=4,
                label=leg_2,
            )

            for i, r in enumerate(layer_results):
                sig = r["p_value"] < alpha
                dot_c = BootstrapSignificanceTester._INSIGNIFICANT_COLOR
                ax.scatter(
                    [layers[i]],
                    [obs_1[i]],
                    color=color_light if sig else dot_c,
                    zorder=5,
                    s=40,
                )
                ax.scatter(
                    [layers[i]],
                    [obs_2[i]],
                    color=color_dark if sig else dot_c,
                    zorder=5,
                    s=40,
                )

        info_parts: list[str] = []
        if model_name:
            info_parts.append(f"model={model_name}")
        info_parts.append(f"n_bootstrap={n_bootstrap}")
        if extra_iter_num != 0:
            info_parts.append(f"extra_iter={extra_iter_num}")
        info_parts.append(f"grey dots = p ≥ {alpha:.2f}")

        ax.set_title(
            f"{metric_name} with {ci_pct}% CI — {label_1} vs {label_2}\n"
            + " | ".join(info_parts)
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_name)
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        BootstrapSignificanceTester._maybe_save(fig, save, filename)
        if show:
            plt.show()

    @staticmethod
    def plot_metric_difference_over_layers(
        layer_results: list[dict[str, Any]],
        label_1: str = "Result 1",
        label_2: str = "Result 2",
        metric_name: str = "Macro F1",
        model_name: str = "",
        extra_iter_num: int = 0,
        y_range: tuple[float, float] | None = (0.0, 1.0),
        show: bool = True,
        save: bool = False,
        filename: str = "",
    ) -> None:
        """
        Plot per-layer CI error bars for the difference (result_1 - result_2).

        Dots are drawn in purple when the layer is significant (p < alpha) and grey otherwise.
        P-values are annotated above each point.
        """
        if save and not filename:
            raise ValueError("filename must be specified when save=True")

        n_bootstrap = len(layer_results[0]["metric_1_samples"])
        conf = layer_results[0]["confidence"]
        alpha = 1.0 - conf
        ci_pct = int(round(conf * 100))
        n_layers = len(layer_results)

        layers = [r["layer"] for r in layer_results]
        obs_diff = [r["observed_diff"] for r in layer_results]

        err_diff = (
            [o - r["ci_diff"][0] for o, r in zip(obs_diff, layer_results)],
            [r["ci_diff"][1] - o for o, r in zip(obs_diff, layer_results)],
        )

        fig, ax = plt.subplots(figsize=(max(10, n_layers * 0.5 + 4), 5))

        ax.errorbar(
            layers,
            obs_diff,
            yerr=err_diff,
            fmt="-",
            color="purple",
            capsize=4,
            label=f"Difference ({label_1} − {label_2})",
        )
        ax.axhline(0, color="black", linestyle="--", alpha=0.5, label="No difference")

        dot_c = BootstrapSignificanceTester._INSIGNIFICANT_COLOR
        for i, r in enumerate(layer_results):
            ax.scatter(
                [layers[i]],
                [obs_diff[i]],
                color="purple" if r["p_value"] < alpha else dot_c,
                zorder=5,
                s=40,
            )
            ax.annotate(
                f"p={r['p_value']:.3f}",
                (r["layer"], obs_diff[i]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=7,
            )

        info_parts: list[str] = []
        if model_name:
            info_parts.append(f"model={model_name}")
        info_parts.append(f"n_bootstrap={n_bootstrap}")
        if extra_iter_num != 0:
            info_parts.append(f"extra_iter={extra_iter_num}")
        info_parts.append(f"grey dots = p ≥ {alpha:.2f}")

        ax.set_title(
            f"Δ {metric_name} ({label_1} − {label_2}) with {ci_pct}% CI\n"
            + " | ".join(info_parts)
        )
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Δ {metric_name}")
        if y_range is not None:
            ax.set_ylim(*y_range)
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        BootstrapSignificanceTester._maybe_save(fig, save, filename)
        if show:
            plt.show()

    @staticmethod
    def _maybe_save(fig: Figure, save: bool, filename: str) -> None:
        if not save:
            return
        save_dir = Path(PLOTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname: str = filename if filename.endswith(".png") else f"{filename}.png"
        fig.savefig(str(save_dir / fname))
        print(f"Plot saved to {save_dir / fname}")


def get_test_name(
    test, model_name, split, changing_var_1, changing_var_2, extra_iter_num=0
):
    return f"{test},{model_name},{split},{extra_iter_num},{changing_var_1}_vs_{changing_var_2}"


def _run_or_load(
    tester: BootstrapSignificanceTester,
    test_name: str,
    force_fresh_tests: bool,
    save: bool,
) -> list[dict[str, Any]]:
    """Run the bootstrap test or load saved results if force_fresh_tests is False and a file exists."""
    if force_fresh_tests:
        print(f"Running test {test_name}")
        return tester.run(macro_f1_metric, save=save, test_name=test_name)
    else:
        filepath = Path(SIGNIFICANCE_RESULTS_FOLDER) / f"{test_name}.json"
        if filepath.exists():
            print(f"Loading existing results from {filepath}")
            return BootstrapSignificanceTester.load_results(test_name)
        else:
            print(f"Could not find {test_name} file. Running fresh test")
            return tester.run(macro_f1_metric, save=save, test_name=test_name)


def run_tests_between_langs(
    language_pairs: list[tuple[str, str]],
    model_name: str,
    split: str,
    save: bool = False,
    make_plots: bool = False,
    force_fresh_tests: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    For each (language_1, language_2) pair, compare experiment 1 probes trained on each
    language. One test is run per pair.
    """
    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for language_1, language_2 in language_pairs:
        print(f"\n--- between_langs: {language_1} vs {language_2} ---")
        result_1 = ExperimentResult.get_from_file(
            1, language_1, "standard", "lr", model_name
        )
        result_2 = ExperimentResult.get_from_file(
            1, language_2, "standard", "lr", model_name
        )

        test_name = get_test_name(
            "between_langs", model_name, split, language_1, language_2
        )
        tester = BootstrapSignificanceTester(result_1, result_2, split)
        layer_results = _run_or_load(tester, test_name, force_fresh_tests, save)
        tester.print_summary(layer_results, label_1=language_1, label_2=language_2)

        if make_plots:
            tester.plot_metric_over_layers(
                [layer_results], label_1=language_1, label_2=language_2
            )
            tester.plot_histograms(
                layer_results, label_1=language_1, label_2=language_2
            )

        all_results[(language_1, language_2)] = layer_results

    return all_results


def run_tests_between_language_pairs(
    language_pairs: list[tuple[str, str]],
    model_name: str,
    split: str,
    extra_iter_num: int,
    save: bool = False,
    make_plots: bool = False,
    force_fresh_tests: bool = False,
) -> dict[tuple[tuple[str, str], tuple[str, str]], list[dict[str, Any]]]:
    """
    For every combination of two experiment-2 language pairs, compare the probes.
    One test is run per pair-of-pairs.
    """
    all_results: dict[
        tuple[tuple[str, str], tuple[str, str]], list[dict[str, Any]]
    ] = {}

    for lp_1, lp_2 in combinations(language_pairs, 2):
        lp_str_1 = get_language_merged_string(lp_1)
        lp_str_2 = get_language_merged_string(lp_2)
        print(f"\n--- between_language_pairs: {lp_str_1} vs {lp_str_2} ---")

        result_1 = ExperimentResult.get_from_file(
            2, lp_str_1, "standard", "lr", model_name, extra_iter_num=extra_iter_num
        )
        result_2 = ExperimentResult.get_from_file(
            2, lp_str_2, "standard", "lr", model_name, extra_iter_num=extra_iter_num
        )

        test_name = get_test_name(
            "between_language_pairs",
            model_name,
            split,
            lp_str_1,
            lp_str_2,
            extra_iter_num=extra_iter_num,
        )
        tester = BootstrapSignificanceTester(result_1, result_2, split)
        layer_results = _run_or_load(tester, test_name, force_fresh_tests, save)
        tester.print_summary(layer_results, label_1=lp_str_1, label_2=lp_str_2)

        if make_plots:
            tester.plot_metric_over_layers(
                [layer_results], label_1=lp_str_1, label_2=lp_str_2
            )
            tester.plot_histograms(layer_results, label_1=lp_str_1, label_2=lp_str_2)

        all_results[(lp_1, lp_2)] = layer_results

    return all_results


def run_tests_between_probing_tasks(
    language_pairs: list[tuple[str, str]],
    model_name: str,
    split: str,
    extra_iter_num: int,
    save: bool = False,
    make_plots: bool = False,
    force_fresh_tests: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    For each experiment-2 language pair, compare "standard" vs "control" probing tasks.
    One test is run per language pair.
    """
    task_1, task_2 = "standard", "control"
    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for language_pair in language_pairs:
        lp_str = get_language_merged_string(language_pair)
        print(f"\n--- between_probing_tasks: {lp_str} ({task_1} vs {task_2}) ---")

        result_1 = ExperimentResult.get_from_file(
            2, lp_str, task_1, "lr", model_name, extra_iter_num=extra_iter_num
        )
        result_2 = ExperimentResult.get_from_file(
            2, lp_str, task_2, "lr", model_name, extra_iter_num=extra_iter_num
        )

        # Include lp_str in the name to avoid collisions across language pairs
        test_name = (
            get_test_name(
                "between_probing_tasks",
                model_name,
                split,
                task_1,
                task_2,
                extra_iter_num=extra_iter_num,
            )
            + f",{lp_str}"
        )
        tester = BootstrapSignificanceTester(result_1, result_2, split)
        layer_results = _run_or_load(tester, test_name, force_fresh_tests, save)
        tester.print_summary(layer_results, label_1=task_1, label_2=task_2)

        if make_plots:
            tester.plot_metric_over_layers(
                [layer_results], label_1=task_1, label_2=task_2
            )
            tester.plot_histograms(layer_results, label_1=task_1, label_2=task_2)

        all_results[language_pair] = layer_results

    return all_results


def run_tests_between_extra_iter_nums(
    language_pairs: list[tuple[str, str]],
    model_name: str,
    split: str,
    extra_iter_nums: tuple[int, int],
    save: bool = False,
    make_plots: bool = False,
    force_fresh_tests: bool = False,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """
    For each language pair, compare experiment-2 probes refitted for different numbers of iterations.
    """
    iter_1, iter_2 = extra_iter_nums
    all_results: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for language_pair in language_pairs:
        lp_str = get_language_merged_string(language_pair)
        print(f"\n--- between_extra_iter_nums: {lp_str}, {iter_1} vs {iter_2} ---")

        result_1 = ExperimentResult.get_from_file(
            2, lp_str, "standard", "lr", model_name, extra_iter_num=iter_1
        )
        result_2 = ExperimentResult.get_from_file(
            2, lp_str, "standard", "lr", model_name, extra_iter_num=iter_2
        )

        # Include lp_str in the name to avoid collisions across language pairs
        test_name = (
            get_test_name(
                "between_extra_iter_nums", model_name, split, str(iter_1), str(iter_2)
            )
            + f",{lp_str}"
        )
        tester = BootstrapSignificanceTester(result_1, result_2, split)
        layer_results = _run_or_load(tester, test_name, force_fresh_tests, save)
        tester.print_summary(layer_results, label_1=str(iter_1), label_2=str(iter_2))

        if make_plots:
            tester.plot_metric_over_layers(
                [layer_results], label_1=str(iter_1), label_2=str(iter_2)
            )
            tester.plot_histograms(
                layer_results, label_1=str(iter_1), label_2=str(iter_2)
            )

        all_results[language_pair] = layer_results

    return all_results


if __name__ == "__main__":
    test_type: str = sys.argv[1]
    model_name: str = sys.argv[2]
    languages: list[str] = LANGUAGES

    # extra_iter_num is relevant for the between_language_pairs and between_probing_tasks tests.
    # It controls which probe we take in terms of how many times it has been refitted
    extra_iter_num = 0

    extra_iter_nums = (0, 1000)

    save = True
    make_plots = True

    match test_type:
        case "between_langs":
            language_pairs = get_language_pair_combinations(languages)
            split = "test"
            run_tests_between_langs(
                language_pairs, model_name, split, save=save, make_plots=make_plots
            )
        case "between_language_pairs":
            language_pairs = get_language_pair_permutations(languages)
            split = "test_b"
            run_tests_between_language_pairs(
                language_pairs,
                model_name,
                split,
                extra_iter_num,
                save=save,
                make_plots=make_plots,
            )
        case "between_probing_tasks":
            language_pairs = get_language_pair_permutations(languages)
            split = "test_b"
            run_tests_between_probing_tasks(
                language_pairs,
                model_name,
                split,
                extra_iter_num,
                save=save,
                make_plots=make_plots,
            )
        case "between_extra_iter_nums":
            language_pairs = get_language_pair_permutations(languages)
            split = "test_b"
            run_tests_between_extra_iter_nums(
                language_pairs,
                model_name,
                split,
                extra_iter_nums,
                save=save,
                make_plots=make_plots,
            )
