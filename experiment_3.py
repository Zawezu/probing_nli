from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from pathlib import Path
import json

from experiment_common_code import ExperimentResult
from sick import SICKMergedDataset
from utils import LABEL_MAP

RESPONSES_FOLDER = "./data/responses"
LABEL_ACCEPTED_VERSIONS: dict[str, dict[str, list[str]]] = {
    "en": {
        "neutral": ["neutr"],
        "entailment": ["entail"],
        "contradiction": ["contrad"],
    },
    "es": {"neutral": ["neutr"], "entailment": ["impli"], "contradiction": ["contrad"]},
    "jp": {"neutral": ["中立"], "entailment": ["含意"], "contradiction": ["矛盾"]},
    "nl": {
        "neutral": ["neutr"],
        "entailment": ["impli"],
        "contradiction": ["tegensp", "contrad"],
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
        valid_responses: dict[str, str] = STRICT_VALID_RESPONSES[self.language]
        for label, expected in valid_responses.items():
            if self.response == expected:
                return LABEL_MAP[label]
        # if self.label != -1:
        #     print(f'Lenient accepted but strict rejected: "{self.response}"')
        return -1

    def parse_label_from_response(self) -> int:
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
        self.original_ids: list[int] = []

        self.model_name: str = model_name
        self.language: str = language
        self.split: str = split

        self.load_dataset()

    def load_dataset(self) -> None:
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
        return [response.label for response in self.responses]

    def get_strict_labels(self) -> list[int]:
        return [response.strict_label for response in self.responses]

    def __getitem__(self, i: int) -> Response:
        return self.responses[i]

    def __len__(self) -> int:
        return len(self.responses)


def get_unk_count(exp_result, split) -> int:
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
