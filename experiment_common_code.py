import pickle
from pathlib import Path
from typing import Any
import torch as t
import matplotlib.pyplot as plt

from common_constants import EXPERIMENT_RESULTS_FOLDER, PLOTS_FOLDER

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
    ) -> None:
        self.experiment_number: int = experiment_number
        self.language: str = language
        self.probing_task: str = probing_task
        self.probe_type: str = probe_type
        self.model_name: str = model_name

        self.train_accuracies: list[float] | None = None
        self.test_accuracies: list[float] | None = None

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

    def get_attributes(self, metrics: list[str]) -> dict[str, Any]:
        return {metric: getattr(self, metric, None) for metric in metrics}

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


def get_accuracy(preds: t.Tensor, labels: t.Tensor) -> float:
    return (preds == labels).float().mean().item()


def plot_multiple_metrics(
    experiments: list[ExperimentResult],
    metric_types: list[str],
    title: str,
    xlabel: str,
    ylabel: str,
    show: bool = True,
    save: bool = False,
) -> None:
    single_subplot_size = (10, 6)
    num_subplots: int = len(metric_types)
    fig, axs = plt.subplots(
        nrows=num_subplots,
        figsize=(single_subplot_size[0], single_subplot_size[1] * num_subplots),
        sharey=True,
        squeeze=False,
    )

    for i, metric_type in enumerate(metric_types):
        ax = axs[i, 0]
        for experiment in experiments:
            name: str = experiment.get_name()
            results: list[float] = getattr(experiment, metric_type)
            layers = range(len(results))

            ax.plot(layers, results, marker="o", label=name)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(metric_type.replace("_", " ").title())
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)

    if save:
        save_dir = Path(PLOTS_FOLDER)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Replace spaces ny underscores
        file_name: str = f"{title}.png".replace(" ", "_")
        filepath: Path = save_dir / file_name

        fig.savefig(filepath)
        print(f"Plot saved to {filepath}")

    if show:
        plt.show()
