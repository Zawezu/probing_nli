from collections import Counter
from hashlib import sha256

from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import json
import random
import csv
import nltk

from utils import (
    SICK_DIRTY_NL_FILE,
    SICK_FOLDER,
    SICK_DIRTY_FOLDERS,
    SICK_DIRTY_EN_FILE,
    SICK_DIRTY_ES_FILE,
    LABEL_MAP,
    LANGUAGES,
    SPLITS,
    MERGED_SICK_FILEPATH,
    SICK_DIRTY_JP_FILE,
)

random.seed(42)


class SICKDirtyDataset(Dataset):
    def __init__(self, language: str, split: str) -> None:
        """Load a single-language SICK split directly from its original source file.

        Supports 'en', 'es', 'jp', and 'nl'. Maps the dataset's 'trial' split to
        the 'val' split name used throughout this project.

        Args:
            language: Language code ('en', 'es', 'jp', or 'nl').
            split: One of 'train', 'test', or 'val'.
        """
        self.sentence_pairs: list[tuple[str, str]] = []
        self.labels: list[int] = []
        self.original_ids: list[int] = []

        self.language: str = language
        self.split: str = split

        self._load_dirty_dataset(split)

    def _load_dirty_dataset(self, split: str) -> None:
        """Parse the raw source file for self.language and populate sentence_pairs, labels, and original_ids."""
        # The original set has a trial instead of a validation set, so the _load_dirty_dataset function will use trial
        original_split: str = "trial" if split == "val" else split

        match self.language:
            case "en":
                filepath: str = (
                    f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS['en']}/{SICK_DIRTY_EN_FILE}"
                )

                with open(filepath, "r", encoding="utf-8") as f:
                    next(f)  # Skip first line, since it is the column names
                    for line in f:
                        line: str = line.strip()
                        if not line:  # Skip empty lines
                            continue

                        data: list[str] = [s.strip() for s in line.split("\t")]

                        if data[-1].lower() != original_split:
                            continue

                        pair_ID = int(data[0])
                        # pair_type = data[1]
                        sentence_A: str = data[2]
                        # sentence_A_expRule = data[3]
                        sentence_B: str = data[4]
                        # sentence_B_expRule = data[5]
                        # relatedness_score = float(data[6])
                        label: str = data[7].lower()
                        # entailment_AB = data[8]
                        # entailment_BA = data[9]
                        # sentence_A_original = data[10]
                        # sentence_B_original = data[11]
                        # sentence_A_dataset = data[12]
                        # sentence_B_datase = data[13]

                        self.sentence_pairs.append((sentence_A, sentence_B))
                        self.labels.append(LABEL_MAP[label])
                        self.original_ids.append(pair_ID)
            case "es":
                filepath = f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS['es']}/{SICK_DIRTY_ES_FILE[original_split]}"

                with open(filepath, "r", encoding="utf-8") as f:
                    next(f)  # Skip first line, since it is the column names
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue

                        data = [s.strip() for s in line.split("\t")]

                        pair_ID = int(data[0])
                        sentence_A = data[1]
                        sentence_B = data[2]

                        # SICK Spanish sentences have dots that English SICK sentences do not.
                        # We remove them for consistency
                        sentence_A = sentence_A.replace(".", "").strip()
                        sentence_B = sentence_B.replace(".", "").strip()

                        # relatedness_score = float(data[3])
                        label = data[4].lower()

                        self.sentence_pairs.append((sentence_A, sentence_B))
                        self.labels.append(LABEL_MAP[label])
                        self.original_ids.append(pair_ID)
            case "jp":
                filepath: str = (
                    f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS['jp']}/{SICK_DIRTY_JP_FILE}"
                )

                with open(filepath, "r", newline="", encoding="utf-8") as f:
                    tsv_reader: csv.DictReader[str] = csv.DictReader(f, delimiter="\t")
                    for row in tsv_reader:
                        if row["data"] != original_split:
                            continue

                        self.sentence_pairs.append(
                            (row["sentence_A_Ja"], row["sentence_B_Ja"])
                        )
                        self.labels.append(LABEL_MAP[row["entailment_label_Ja"]])
                        self.original_ids.append(int(row["pair_ID"]))
            case "nl":
                filepath: str = (
                    f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS['nl']}/{SICK_DIRTY_NL_FILE}"
                )

                with open(filepath, "r", encoding="utf-8") as f:
                    next(f)  # Skip first line, since it is the column names
                    for line in f:
                        line: str = line.strip()
                        if not line:  # Skip empty lines
                            continue

                        data: list[str] = [s.strip() for s in line.split("\t")]

                        if data[-1].lower() != original_split:
                            continue

                        pair_ID = int(data[0])
                        sentence_A: str = data[1]
                        sentence_B: str = data[2]
                        label: str = data[3].lower()

                        self.sentence_pairs.append((sentence_A, sentence_B))
                        self.labels.append(LABEL_MAP[label])
                        self.original_ids.append(pair_ID)

    def __getitem__(self, i: int) -> tuple[tuple[str, str], int, int]:
        return self.sentence_pairs[i], self.labels[i], self.original_ids[i]

    def __len__(self) -> int:
        return len(self.sentence_pairs)


class SICKMergedDataset(Dataset):
    def __init__(
        self,
        language: str,
        split: str,
        control_type="noun_based_control_label",
        force_original_labels: bool = False,
    ) -> None:
        """Load a language/split slice from the pre-built merged SICK JSON.

        The merged file contains all languages and all control label columns. Each
        sample's label dict holds both the 'standard' NLI label and a control label
        determined by `control_type`.

        Args:
            language: Language code ('en', 'es', 'jp', or 'nl').
            split: One of 'train', 'test', or 'val'.
            control_type: Column name for the control label (default: 'noun_based_control_label').
            force_original_labels: If True and language is 'jp', use 'standard_label'
                instead of 'standard_japanese_label'.
        """
        self.sentence_pairs: list[tuple[str, str]] = []
        self.labels: list[dict[str, int]] = []
        self.original_ids: list[int] = []

        self.language: str = language
        self.split: str = split
        self.control_type: str = control_type
        self.force_original_labels = force_original_labels

        self.load_dataset()

    def load_dataset(self) -> None:
        """Read MERGED_SICK_FILEPATH and populate sentence_pairs, labels, and original_ids."""
        with open(MERGED_SICK_FILEPATH, "r", encoding="utf-8") as file:
            merged_dataset_dict: dict[str, dict[str, str | int]] = json.load(file)

        for id, values in merged_dataset_dict.items():
            if values["split"] == self.split:
                self.sentence_pairs.append(
                    (
                        str(values[f"sentence_a_{self.language}"]),
                        str(values[f"sentence_b_{self.language}"]),
                    )
                )

                # Japanese uses slightly different labels
                if self.language == "jp" and not self.force_original_labels:
                    standard_label_key = "standard_japanese_label"
                else:
                    standard_label_key = "standard_label"

                self.labels.append(
                    {
                        "standard": int(values[standard_label_key]),
                        "control": int(values[self.control_type]),
                    }
                )
                self.original_ids.append(int(id))

    def get_labels(
        self, start: int = 0, end: int | None = None, probing_task: str = "standard"
    ) -> list[int]:
        """Return integer labels for a slice of the dataset.

        Args:
            start: First index (inclusive).
            end: Last index (exclusive); defaults to the end of the dataset.
            probing_task: Which label column to extract ('standard' or 'control').
        """
        if end is None:
            end = len(self.labels)

        labels_selected: list[dict[str, int]] = self.labels[start:end]
        return [x[probing_task] for x in labels_selected]

    def __getitem__(self, i: int) -> tuple[tuple[str, str], dict[str, int], int]:
        return (
            self.sentence_pairs[i],
            self.labels[i],
            self.original_ids[i],
        )

    def __len__(self) -> int:
        return len(self.sentence_pairs)


def calculate_majority_class_baseline_f1(
    probing_task: str, language: str, control_type: str = "noun_based_control_label"
) -> float:
    """Compute the macro-F1 a majority-class classifier achieves on the test split.

    The majority class is determined from the training split.
    """
    train_dataset = SICKMergedDataset(language, "train", control_type)
    majority_class: int = Counter(
        train_dataset.get_labels(probing_task=probing_task)
    ).most_common(1)[0][0]

    test_dataset = SICKMergedDataset(language, "test", control_type)
    test_labels: list[int] = test_dataset.get_labels(probing_task=probing_task)

    predictions: list[int] = [majority_class] * len(test_labels)
    return float(f1_score(test_labels, predictions, average="macro"))


def get_dataset_and_dataloader(
    language, split, batch_size=1, force_original_labels=False
) -> tuple[SICKMergedDataset, DataLoader[tuple[tuple[str, str], dict[str, int], int]]]:
    """Construct a SICKMergedDataset and a non-shuffled DataLoader for it."""
    dataset: SICKMergedDataset = SICKMergedDataset(
        language, split, force_original_labels=force_original_labels
    )

    dataloader: DataLoader[tuple[tuple[str, str], dict[str, int], int]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataset, dataloader


def add_to_dict(dictionary, key, value) -> None:
    """Insert key/value into dictionary only if key is absent; assert consistency if already present."""
    try:
        assert dictionary[key] == value
    except KeyError:
        dictionary[key] = value


def create_disjunct_control_labels(dataset_dict, unique_labels, amount_per_label):
    """Assign a random control label to each entry that is guaranteed to differ from its standard label.

    Uses class-conditional sampling: the probability of drawing each label is
    proportional to its overall frequency in the dataset, but the entry's own
    original label is excluded (weight set to 0).
    """
    print("Creating disjunct control labels")
    # Create dictionary with the label_ratios when we exclude each of the labels
    total_amount: int = sum(amount_per_label)
    label_ratios: dict[str, dict[int, float]] = {}
    for label_excluded, amount_excluded in enumerate(amount_per_label):
        label_ratios[f"{label_excluded}_out"] = {}
        # Calculate the ratio of each label if we exclude a particular label
        for label, amount in enumerate(amount_per_label):
            label_ratios[f"{label_excluded}_out"][label] = amount / (
                total_amount - amount_excluded
            )
        # Forcefully exclude the label by setting its ratio to 0
        label_ratios[f"{label_excluded}_out"][label_excluded] = 0
    print(f"Label ratios excluding each of the labels: {label_ratios}")

    # print(label_ratios)

    for id, values in dataset_dict.items():
        original_label: int = values["standard_label"]
        label_ratios_used: list[float] = list(
            label_ratios[f"{original_label}_out"].values()
        )
        # print(f"label_ratios_used={label_ratios_used}")
        disjunct_control_label: int = random.choices(
            unique_labels, weights=label_ratios_used, k=1
        )[0]
        dataset_dict[id]["disjunct_control_label"] = disjunct_control_label


def create_non_disjunct_control_labels(dataset_dict, unique_labels, amount_per_label):
    """Assign random control labels sampled from the same class distribution as the standard labels.

    Unlike the disjunct variant, a control label may coincide with the standard label.
    Labels are drawn without replacement across the full dataset to preserve class ratios.
    """
    print("Creating non-disjunct control labels")
    print(f"Amount per label: {amount_per_label}")
    label_ratios: list[float] = [
        amount / sum(amount_per_label) for amount in amount_per_label
    ]
    print(f"Label ratios: {label_ratios}")
    # Create list with the correct amounts of each label
    control_labels: list[int] = random.choices(
        unique_labels, label_ratios, k=sum(amount_per_label)
    )

    # Add the control label to each row in the dataset
    for id in dataset_dict.keys():
        dataset_dict[id]["random_choice_control_label"] = control_labels.pop(0)


def create_id_hash_control_labels(dataset_dict, unique_labels) -> None:
    """Assign a deterministic control label to each entry derived from the SHA-256 hash of its ID.

    The hash is reduced modulo the number of unique labels, giving a reproducible but
    pseudo-random assignment that is independent of the standard label.
    """
    print("Creating pair id-based control labels")

    n: int = len(unique_labels)

    for id in dataset_dict.keys():
        id_bytes = str(id).encode("utf-8")
        sha256_hash = sha256(id_bytes).hexdigest()
        id_hash_control_label: int = int(sha256_hash, 16) % n
        dataset_dict[id]["id_hash_control_label"] = id_hash_control_label


def create_noun_based_control_labels(dataset_dict, unique_labels) -> None:
    """Assign a control label based on the first noun in sentence_a_en.

    All pairs sharing the same first noun receive the same randomly chosen label.
    Pairs with no noun are grouped under the '__no_noun__' key.
    """
    print("Creating noun-based control labels")

    noun_tags = {"NN", "NNS", "NNP", "NNPS"}

    # Map each noun to the ids of sentence pairs whose first noun is that noun
    noun_to_ids: dict[str, list[str]] = {}
    for id, values in dataset_dict.items():
        sentence_a: str = values["sentence_a_en"]
        tokens = nltk.word_tokenize(sentence_a)
        pos_tags = nltk.pos_tag(tokens)

        noun = next(
            (token.lower() for token, tag in pos_tags if tag in noun_tags),
            "__no_noun__",
        )

        if noun not in noun_to_ids:
            noun_to_ids[noun] = []
        noun_to_ids[noun].append(id)

    # Assign a random label to each distinct noun
    noun_to_label: dict[str, int] = {
        noun: random.choice(unique_labels) for noun in noun_to_ids
    }

    # Write the label back to each sentence pair
    for noun, ids in noun_to_ids.items():
        label = noun_to_label[noun]
        for id in ids:
            dataset_dict[id]["noun_based_control_label"] = label


def create_control_labels(
    dataset_dict, disjunct: bool, predetermined_label_ratio: list[float] | None = None
) -> None:
    """Create all control label columns and write them into dataset_dict in place.

    When `predetermined_label_ratio` is None the class distribution is derived from
    the actual standard labels. Pass `disjunct=True` to guarantee each control label
    differs from the corresponding standard label, or False to allow coincidences.

    If `predetermined_label_ratio` is given (must sum to 1.0), only disjunct labels
    are created using the specified ratios instead of the empirical ones.
    """
    print("Creating all control labels")
    unique_labels: list[int] = list(LABEL_MAP.values())
    # If we don't specify the label_ratio, use the same ratio as in the real labels
    if predetermined_label_ratio is None:
        # Calculate the real ratio
        amount_per_label: list[int] = [0 for _ in unique_labels]
        for entry in dataset_dict.values():
            standard_label: int = entry["standard_label"]
            amount_per_label[standard_label] += 1

        if disjunct:
            create_disjunct_control_labels(
                dataset_dict, unique_labels, amount_per_label
            )
        else:
            create_non_disjunct_control_labels(
                dataset_dict, unique_labels, amount_per_label
            )

        create_id_hash_control_labels(dataset_dict, unique_labels)
        create_noun_based_control_labels(dataset_dict, unique_labels)
    else:
        assert (
            round(sum(list(predetermined_label_ratio)), 3) == 1.00
        ), "predetermined_label_ratio must sum up to 1"
        total_amount_of_labels = len(dataset_dict.values())
        print(f"total_amount_of_labels: {total_amount_of_labels}")
        amount_per_label = [
            round(ratio * total_amount_of_labels) for ratio in predetermined_label_ratio
        ]
        create_disjunct_control_labels(dataset_dict, unique_labels, amount_per_label)


def create_merged_json(save=False) -> None:
    """Build the merged SICK JSON that combines all languages, splits, and control labels.

    Iterates over all languages and splits, aligning sentence pairs by their original
    pair_ID. Japanese entries that have no English counterpart are skipped. All control
    label variants are added via `create_control_labels`. When `save=True` the result
    is written to MERGED_SICK_FILEPATH.
    """
    merged_dataset_dict: dict[str, dict[str, str | int]] = {}

    seen_ids = set()

    for language in LANGUAGES:
        for split in SPLITS:
            dataset: SICKDirtyDataset = SICKDirtyDataset(language, split)

            dataloader: DataLoader[tuple[tuple[str, str], int, int]] = DataLoader(
                dataset, batch_size=1, shuffle=False
            )

            for (sentence_a, sentence_b), label, original_id in dataloader:
                # print(sentence_a, sentence_b, label, original_id, split)

                id = str(original_id.item())

                if id not in seen_ids:
                    #  There are some pairs that are only in Japanese. We skip these
                    if language == "jp":
                        continue

                    merged_dataset_dict[id] = {}
                    seen_ids.add(id)

                add_to_dict(
                    merged_dataset_dict[id], f"sentence_a_{language}", sentence_a[0]
                )
                add_to_dict(
                    merged_dataset_dict[id], f"sentence_b_{language}", sentence_b[0]
                )
                if language == "jp":
                    add_to_dict(
                        merged_dataset_dict[id], "standard_japanese_label", label.item()
                    )
                else:
                    add_to_dict(merged_dataset_dict[id], "standard_label", label.item())
                add_to_dict(merged_dataset_dict[id], "split", split)

    create_control_labels(merged_dataset_dict, disjunct=False)
    create_control_labels(merged_dataset_dict, disjunct=True)

    # print(merged_dataset_dict)
    if save:
        print(f"Saving merged SICK dataset to {MERGED_SICK_FILEPATH}")
        with open(MERGED_SICK_FILEPATH, "w", encoding="utf-8") as f:
            f.write(json.dumps(merged_dataset_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # create_merged_json(save=True)

    # Sanity tests
    # for language in LANGUAGES:
    #     print(language)
    #     split= "train"

    #     dataset, dataloader = get_dataset_and_dataloader(language, split)

    #     print(dataset.labels[:20])

    dataset_en, dataloader_en = get_dataset_and_dataloader("en", "train")
    dataset_jp, dataloader_jp = get_dataset_and_dataloader(
        "jp", "train", force_original_labels=True
    )

    diffs_in_standard = 0
    diffs_in_control = 0

    for label_en, label_jp in zip(dataset_en.labels, dataset_jp.labels):
        if label_en["standard"] != label_jp["standard"]:
            print("Difference in standard!")
            diffs_in_standard += 1

        if label_en["control"] != label_jp["control"]:
            print("Difference in control!")
            diffs_in_control += 1

    print(f"Differences in standard: {diffs_in_standard}")
    print(f"Differences in control: {diffs_in_control}")
    # for probing_task in ["standard", "control"]:
    #     for language in ["en", "jp"]:
    #         print(
    #             f"f1 for majority class of {probing_task} {language}: {calculate_majority_class_baseline_f1('standard', 'en')}"
    #         )
