from typing import Any
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from pathlib import Path
import json
from pandas.core.frame import DataFrame

from experiment_common_code import ExperimentResult
from sick import SICKMergedDataset, calculate_majority_class_baseline_f1
from utils import LABEL_MAP, REVERSE_LABEL_MAP

RESPONSES_FOLDER = "./data/responses"
LABEL_ACCEPTED_VERSIONS: dict[str, dict[str, list[str]]] = {
    "en": {
        "neutral": ["neutral"],
        "entailment": ["entail"],
        "contradiction": ["contradict", "contradition"],
    },
    "es": {
        "neutral": ["neutral", "neutro"],
        "entailment": ["implica"],
        "contradiction": ["contradic"],
    },
    "jp": {"neutral": ["中立"], "entailment": ["含意"], "contradiction": ["矛盾"]},
    "nl": {
        "neutral": ["neutral", "neutraal"],
        "entailment": ["implicatie"],
        "contradiction": ["tegensp", "contradict"],
    },
}

STRICT_VALID_RESPONSES: dict[str, dict[str, str]] = {
    "en": {
        "neutral": "neutral",
        "entailment": "entailment",
        "contradiction": "contradiction",
    },
    "es": {
        "neutral": "neutral",
        "entailment": "implicación",
        "contradiction": "contradicción",
    },
    "jp": {"neutral": "中立", "entailment": "含意", "contradiction": "矛盾"},
    "nl": {
        "neutral": "neutraal",
        "entailment": "implicatie",
        "contradiction": "tegenspraak",
    },
}

experiment_number = 3


class Response:
    def __init__(self, response_dict: dict, i: int, language: str) -> None:
        """
        Parse a response dict into a Response object.

        Extracts the sentence pair from the question text, the raw response string
        (lowercased and stripped), and the original sentence ID. Computes both a
        lenient label (`label`) and a strict label (`strict_label`), with -1
        indicating an unparseable response in either scheme.

        Args:
            response_dict: A dict with keys "message", "response", and optionally "sentence_id".
            i: Fallback index used as the original_id when "sentence_id" is absent.
            language: Language code ("en", "es", "jp", or "nl"); controls parsing rules.
        """
        self.language: str = language
        sentence_split_char: str = "：" if language == "jp" else ":"

        full_question: str = response_dict["message"][1]["content"]
        question_parts: list[str] = full_question.split("\n")
        sentence_a: str = question_parts[0].split(sentence_split_char)[1].strip()
        sentence_b: str = question_parts[1].split(sentence_split_char)[1].strip()
        # print("----------------")
        # print(f"sentence_a = {sentence_a}")
        # print(f"sentence_b = {sentence_b}")

        self.sentence_pair: tuple[str, str] = (sentence_a, sentence_b)
        self.response: str = response_dict["response"].strip().lower()
        self.original_id: int = (
            response_dict["sentence_id"] if "sentence_id" in response_dict.keys() else i
        )

        self.label: int = self.parse_label_from_response()
        self.strict_label: int = self.parse_strict_label_from_response()

    def parse_strict_label_from_response(self) -> int:
        """Return the label integer if the response exactly matches an expected string, else -1."""
        valid_responses: dict[str, str] = STRICT_VALID_RESPONSES[self.language]
        for label, expected in valid_responses.items():
            if self.response == expected:
                return LABEL_MAP[label]
        # if self.label != -1:
        #     print(f'Lenient accepted but strict rejected: "{self.response}"')
        return -1

    def parse_label_from_response(self) -> int:
        """
        Return the label integer using lenient substring matching, or -1 if ambiguous.

        Returns -1 immediately if the response contains "no" or "ない" (ambiguity guard).
        Otherwise counts occurrences of each label's accepted substrings; returns -1 if
        zero or more than one label keyword is matched, otherwise returns the matched label.
        """
        label_found_counts: dict[str, int] = {}

        # print(f"Response: {self.response}")

        no_in_aswer: bool = "no" in self.response or "ない" in self.response
        if no_in_aswer:
            # If "no" is found in the answer, we automatically mark it as unknown label, since "no" makes the response ambiguous
            # print('Found "no". Returning unknown')
            return -1

        for label in LABEL_MAP.keys():
            # Count the amount of times possible versions of the label appear in the response
            label_found_counts[label] = 0
            for label_substring in LABEL_ACCEPTED_VERSIONS[self.language][label]:
                label_found_counts[label] += self.response.count(label_substring)

        nonzero_labels: list[str] = [
            label for label, count in label_found_counts.items() if count > 0
        ]
        if len(nonzero_labels) != 1:
            # If there is not exactly one label found, mark it as unknown label
            # print(f'Found {"zero" if len(nonzero_labels) == 0 else "multiple"} labels {nonzero_labels}: {self.response}. Returning unknown')

            return -1

        label: str = nonzero_labels[0]
        label_id: int = LABEL_MAP[label]
        # print(f"Found {label_found_counts[label]} instances of {label}. Returning {label_id}")
        return label_id


class ResponseDataset(Dataset):
    def __init__(self, model_name: str, language: str, split: str) -> None:
        """
        Initialize the dataset by loading all response JSON files for the given model,
        language, and split from data/responses/{model_name}/{language}/{split}/.
        """
        self.original_ids: list[int] = []

        self.model_name: str = model_name
        self.language: str = language
        self.split: str = split

        self.load_dataset()

    def load_dataset(self) -> None:
        """
        Load response JSON files from the responses directory, sorted numerically by batch number.

        Populates self.responses and self.response_counts, then prints response counts.
        """
        # Find correct directory for this language and split
        directory = Path(
            f"{RESPONSES_FOLDER}/{self.model_name}/{self.language}/{self.split}"
        )

        # Find all json files in this directory, sorted numerically by batch number
        pattern: str = "*.json"
        batch_files: list[Path] = sorted(
            directory.glob(pattern), key=lambda p: int(p.stem.split("_batch")[1])
        )

        # print(batch_files)

        self.responses: list[Response] = []
        self.response_counts = {}

        for i, filepath in enumerate(batch_files):
            with open(filepath, "r", encoding="utf-8") as file:
                response_dicts: list[dict] = json.load(file)
                for response_dict in response_dicts:
                    response = Response(response_dict, i, self.language)
                    self.responses.append(response)
                    try:
                        self.response_counts[response.response] += 1
                    except KeyError:
                        self.response_counts[response.response] = 1

        print(
            f"Response counts for {self.model_name, self.language, self.split}:\n{self.response_counts}"
        )

    def get_labels(self) -> list[int]:
        """Return lenient label integers for all responses (-1 for unparseable responses)."""
        return [response.label for response in self.responses]

    def get_strict_labels(self) -> list[int]:
        """Return strict label integers for all responses (-1 for non-exact-match responses)."""
        return [response.strict_label for response in self.responses]

    def __getitem__(self, i: int) -> Response:
        return self.responses[i]

    def __len__(self) -> int:
        return len(self.responses)


class E3DataframeCreator:
    def __init__(
        self,
        model_name: str,
        languages: list[str],
        include_unk: bool = False,
        include_control: bool = False,
        include_strict=False,
    ) -> None:
        """
        Store configuration for building the experiment 3 metrics dataframe.

        Args:
            model_name: Model whose saved ExperimentResult files to load.
            languages: Language codes to include as rows in the dataframe.
            include_unk: If True, add an extra "f1 including unk" row.
            include_control: If True, add a "{language}_control" column alongside each language.
            include_strict: If True, add a "{language}_strict" column alongside each language.
        """
        self.model_name: str = model_name
        self.languages: list[str] = languages
        self.include_unk: bool = include_unk
        self.include_control: bool = include_control
        self.include_strict: bool = include_strict

    def create_dataframe(self) -> DataFrame:
        """
        Build and return a DataFrame of F1 metrics from saved experiment 3 results.

        Rows are languages (plus optional control/strict variants); columns are metrics
        (f1, per-class f1 for each label, unknown count, majority class baseline f1).
        The DataFrame is transposed so languages are rows and metrics are columns.
        """
        self.data: dict[str, Any] = {}

        self.add_row("f1")
        if self.include_unk:
            self.add_row("f1_including_unk")
        self.add_row("per_class_f1", cls=0)
        self.add_row("per_class_f1", cls=1)
        self.add_row("per_class_f1", cls=2)
        self.add_row("unk_count")

        df: DataFrame = pd.DataFrame.from_dict(self.data, orient="index")

        df = df.astype(object)

        df.loc["unk count"] = df.loc["unk count"].astype(int)

        # Transpose so rows=languages, columns=metrics
        df = df.T

        # Add majority class baseline F1 as a new column
        df["baseline f1"] = [
            calculate_majority_class_baseline_f1("standard", lang) for lang in df.index
        ]

        return df

    def add_value(self, metric, cls, language, control=False, strict=False):
        """
        Load a saved ExperimentResult and return the test-split value for the given metric.

        Args:
            metric: Metric key to retrieve (e.g., "f1", "per_class_f1", "unk_count").
            cls: Class index (int) to index into a per-class dict, or None for scalar metrics.
            language: Language code for the ExperimentResult to load.
            control: If True, load the control-task result.
            strict: If True, load the strict-task result.

        Raises:
            ValueError: If the metric list has more than one stored value, or if both
                control and strict are True.
        """
        if control:
            probing_task = "control"
            if self.include_strict:
                raise ValueError("control and strict cannot both be true")
        elif strict:
            probing_task = "strict"
        else:
            probing_task = "standard"

        exp_result: ExperimentResult = ExperimentResult.get_from_file(
            3, language, probing_task, "model_pred", self.model_name
        )
        metric_values = exp_result.get_metric("test", metric)

        if len(metric_values) != 1:
            raise ValueError("picked metric that has more than one value stored")
        metric_value = metric_values[0]

        if cls is not None:
            metric_value = metric_value[str(cls)]

        print(metric_value)

        return metric_value

    def add_row(self, metric: str, cls: None | int = None) -> None:
        """
        Add a row to self.data for the given metric, collecting values per language.

        The row name is derived from the metric (underscores replaced with spaces; "per
        class" stripped for per-class metrics, replaced by the label name instead).
        Optionally appends control/strict variant columns if the corresponding flags are set.

        Args:
            metric: Metric key (e.g., "f1", "per_class_f1", "unk_count").
            cls: Integer class index for per-class metrics; must be None for scalar metrics.
        """
        if cls is None:
            row_name: str = metric
        else:
            if "per_class" not in metric:
                raise AttributeError(
                    f"cls must be None for non-per-class metric {metric}"
                )
            row_name = f"{metric} for {REVERSE_LABEL_MAP[cls]}"

        row_name = row_name.replace("_", " ")
        row_name = row_name.replace("per class", "")
        self.data[row_name] = {}

        for language in self.languages:
            self.data[row_name][language] = self.add_value(metric, cls, language)

            if self.include_control:
                self.data[row_name][f"{language}_control"] = self.add_value(
                    metric, cls, language, control=True
                )

            if self.include_strict:
                self.data[row_name][f"{language}_strict"] = self.add_value(
                    metric, cls, language, strict=True
                )


def get_unk_count(exp_result, split) -> int:
    """
    Count samples involved in "unknown" (-1) labels in the stored confusion matrix.

    Sums the entire first row (all true-label = -1 entries) and the first column
    below the diagonal (pred = -1 with a non-unknown true label) to avoid
    double-counting cm[0, 0]. In practice true labels in experiment 3 are never -1,
    so this effectively counts the total number of -1 predictions.

    Args:
        exp_result: An experiment 3 ExperimentResult with a 4x4 confusion matrix.
        split: Data split to use (e.g., "train" or "test").
    """
    cm = exp_result.get_metric(split, "cm", 0)
    first_row = cm[0, :]
    first_column = cm[1:, 0]
    unk_count: int = int(sum(first_row) + sum(first_column))
    return unk_count


def run_full_experiment(
    language: str,
    model_name: str,
    control: bool = False,
    strict: bool = False,
) -> ExperimentResult:
    """
    Run a single experiment 3 instance for one language and model.

    Stores 4x4 confusion matrices with labels [-1, 0, 1, 2] and computes metrics
    both excluding and including the unknown class. Also records per-cell sample
    indices and unknown counts.

    In control mode the model always predicts neutral (label 1).
    In strict mode, model responses must exactly match the expected string to be
    accepted; otherwise lenient substring matching is used.

    Returns:
        An ExperimentResult with task set to "control", "strict", or "standard".
    """
    print(
        f"Running experiment {experiment_number} control. {language}, {model_name}, control={control}, strict={strict}"
    )
    # Create empty ExperimentResult that we will fill with the results
    if control:
        task = "control"
    elif strict:
        task = "strict"
    else:
        task = "standard"
    exp_result = ExperimentResult(
        experiment_number, language, task, "model_pred", model_name
    )

    train_sick_dataset: SICKMergedDataset = SICKMergedDataset(language, "train")
    test_sick_dataset: SICKMergedDataset = SICKMergedDataset(language, "test")

    train_labels: list[int] = train_sick_dataset.get_labels()
    test_labels: list[int] = test_sick_dataset.get_labels()

    if control:
        # The control version always predicts neutral
        train_preds: list[int] = [1] * len(train_labels)
        test_preds: list[int] = [1] * len(test_labels)
    else:
        train_response_dataset: ResponseDataset = ResponseDataset(
            model_name, language, "train"
        )
        test_response_dataset: ResponseDataset = ResponseDataset(
            model_name, language, "test"
        )
        if strict:
            train_preds = train_response_dataset.get_strict_labels()
            test_preds = test_response_dataset.get_strict_labels()
        else:
            train_preds = train_response_dataset.get_labels()
            test_preds = test_response_dataset.get_labels()

    # Save confusion matrix of train predictions
    # Specify labels to ensure a 4x4 matrix for all possible labels: -1 (unknown), 0, 1, 2
    labels: list[int] = [-1]
    labels.extend(list(LABEL_MAP.values()))

    exp_result.append_metric(
        "train", "cm", confusion_matrix(train_labels, train_preds, labels=labels)
    )  # type: ignore

    # Save confusion matrix of test predictions
    exp_result.append_metric(
        "test", "cm", confusion_matrix(test_labels, test_preds, labels=labels)
    )  # type: ignore

    exp_result.add_metrics_from_confusion_matrix(include_unknown=False)
    exp_result.add_metrics_from_confusion_matrix(include_unknown=True)

    # Add indices per confusion matrix cell for both splits
    exp_result.add_idxs_per_cm_cell_metric("train", train_labels, train_preds)
    exp_result.add_idxs_per_cm_cell_metric("test", test_labels, test_preds)

    exp_result.append_metric("train", "unk_count", get_unk_count(exp_result, "train"))
    exp_result.append_metric("test", "unk_count", get_unk_count(exp_result, "test"))

    # print(confusion_matrix(train_labels, train_preds))

    # print(exp_result.metrics)
    return exp_result


def run_experiment_3(
    languages: list[str],
    model_names: list[str],
    save_results: bool = True,
) -> list[ExperimentResult]:
    """
    Run experiment 3 for all combinations of languages and model names.

    For each (language, model) pair, runs the standard, strict, and control variants
    in that order. Results are optionally saved to disk.

    Returns:
        List of ExperimentResult objects (standard, strict, control per pair).
    """
    exp_results: list[ExperimentResult] = []

    # Run the experiment for each combination of model name, language, and probing task
    for model_name in model_names:
        for language in languages:
            # Run full experiment
            standard_exp_result: ExperimentResult = run_full_experiment(
                language,
                model_name,
            )

            exp_results.append(standard_exp_result)

            strict_exp_result: ExperimentResult = run_full_experiment(
                language,
                model_name,
                strict=True,
            )

            exp_results.append(strict_exp_result)

            # Run control experiment
            control_exp_result: ExperimentResult = run_full_experiment(
                language,
                model_name,
                control=True,
            )

            exp_results.append(control_exp_result)

    # Save results if requested
    if save_results:
        for exp_result in exp_results:
            filepath: str = exp_result.save_to_file()
            print(f"Saved result to {filepath}")

    return exp_results


if __name__ == "__main__":
    languages = ["en"]
    model_names = ["olmo_model"]

    run_experiment_3(languages, model_names, save_results=False)

    exp_result = ExperimentResult.get_from_file(
        3, "en", "standard", "model_pred", "olmo_model"
    )
    # print(exp_result.metrics)
