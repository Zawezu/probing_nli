from torch.utils.data import Dataset, DataLoader
import json
import random

from common_constants import (
    SICK_FOLDER,
    SICK_DIRTY_FOLDERS,
    SICK_DIRTY_EN_FILE,
    SICK_DIRTY_ES_FILE,
    LABEL_MAP,
    LANGUAGES,
    SPLITS,
    MERGED_SICK_FILEPATH,
)

random.seed(42)


class SICKDirtyDataset(Dataset):
    def __init__(self, language: str, split: str) -> None:
        self.sentence_pairs: list[tuple[str, str]] = []
        self.labels: list[int] = []
        self.original_ids: list[int] = []

        self.language: str = language

        self._load_dirty_dataset(split)

    def _load_dirty_dataset(self, split: str) -> None:
        # The original set has a trial instead of a validation set, so the _load_dirty_dataset function will use trial
        original_split: str = "trial" if split == "val" else split

        match self.language:
            case "en":
                filepath: str = (
                    f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS["en"]}/{SICK_DIRTY_EN_FILE}"
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
                filepath = f"./{SICK_FOLDER}/{SICK_DIRTY_FOLDERS["es"]}/{SICK_DIRTY_ES_FILE[original_split]}"

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
                        # relatedness_score = float(data[3])
                        label = data[4].lower()

                        self.sentence_pairs.append((sentence_A, sentence_B))
                        self.labels.append(LABEL_MAP[label])
                        self.original_ids.append(pair_ID)

    def __getitem__(self, i: int) -> tuple[tuple[str, str], int, int]:
        return self.sentence_pairs[i], self.labels[i], self.original_ids[i]

    def __len__(self) -> int:
        return len(self.sentence_pairs)


class SICKMergedDataset(Dataset):
    def __init__(self, language, split) -> None:
        self.sentence_pairs: list[tuple[str, str]] = []
        self.labels: list[dict[str, int]] = []
        self.original_ids: list[int] = []

        self.load_dataset(language, split)

    def load_dataset(self, language: str, split: str) -> None:
        with open(MERGED_SICK_FILEPATH, "r", encoding="utf-8") as file:
            merged_dataset_dict: dict[str, dict[str, str | int]] = json.load(file)

        for id, values in merged_dataset_dict.items():
            if values["split"] == split:
                match language:
                    case "en":
                        self.sentence_pairs.append(
                            (str(values["sentence_a_en"]), str(values["sentence_b_en"]))
                        )
                    case "es":
                        self.sentence_pairs.append(
                            (str(values["sentence_a_es"]), str(values["sentence_b_es"]))
                        )

                self.labels.append(
                    {
                        "standard": int(values["standard_label"]),
                        "control": int(values["control_label"]),
                        "disjunct_control": int(values["disjunct_control_label"]),
                    }
                )
                self.original_ids.append(int(id))

    def get_labels_in_range(self, start: int, end: int, probing_task: str) -> list[int]:
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


def get_dataset_and_dataloader(
    language, split, batch_size=1
) -> tuple[SICKMergedDataset, DataLoader[tuple[tuple[str, str], dict[str, int], int]]]:
    dataset: SICKMergedDataset = SICKMergedDataset(language, split)

    dataloader: DataLoader[tuple[tuple[str, str], dict[str, int], int]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataset, dataloader


def add_to_dict(dictionary, key, value) -> None:
    try:
        assert dictionary[key] == value
    except KeyError:
        dictionary[key] = value


def create_disjunct_labels(dataset_dict, unique_labels, amount_per_label):
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


def create_non_disjunct_labels(dataset_dict, unique_labels, amount_per_label):
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
        dataset_dict[id]["control_label"] = control_labels.pop(0)


def create_control_labels(
    dataset_dict, disjunct: bool, predetermined_label_ratio: list[float] | None = None
) -> None:
    """
    disjunt: determines whether the label at some line is necessarily different from the original label
    """
    print("Creating all control labels")
    unique_labels: list[int] = list(LABEL_MAP.values())
    # If we don't specify the label_ratio, use the same ratio as in the real labels
    if predetermined_label_ratio is None:
        # Calculate the real ratio
        amount_per_label: list[int] = [0 for _ in unique_labels]
        for tup in dataset_dict.values():
            standard_label: int = tup["standard_label"]
            amount_per_label[standard_label] += 1

        if disjunct:
            create_disjunct_labels(dataset_dict, unique_labels, amount_per_label)
        else:
            create_non_disjunct_labels(dataset_dict, unique_labels, amount_per_label)
    else:
        assert (
            round(sum(list(predetermined_label_ratio)), 3) == 1.00
        ), "predetermined_label_ratio must sum up to 1"
        total_amount_of_labels = len(dataset_dict.values())
        print(f"total_amount_of_labels: {total_amount_of_labels}")
        amount_per_label = [
            round(ratio * total_amount_of_labels) for ratio in predetermined_label_ratio
        ]
        create_disjunct_labels(dataset_dict, unique_labels, amount_per_label)


def create_merged_json() -> None:
    merged_dataset_dict: dict[str, dict[str, str | int]] = {}

    for language in LANGUAGES:
        for split in SPLITS:
            dataset: SICKDirtyDataset = SICKDirtyDataset(language, split)

            dataloader: DataLoader[tuple[tuple[str, str], int, int]] = DataLoader(
                dataset, batch_size=1, shuffle=False
            )

            for (sentence_a, sentence_b), label, original_id in dataloader:
                # print(sentence_a, sentence_b, label, original_id, split)

                id = str(original_id.item())

                try:
                    merged_dataset_dict[id]
                except KeyError:
                    merged_dataset_dict[id] = {}

                add_to_dict(
                    merged_dataset_dict[id], f"sentence_a_{language}", sentence_a[0]
                )
                add_to_dict(
                    merged_dataset_dict[id], f"sentence_b_{language}", sentence_b[0]
                )
                add_to_dict(merged_dataset_dict[id], "standard_label", label.item())
                add_to_dict(merged_dataset_dict[id], "split", split)

    create_control_labels(merged_dataset_dict, disjunct=False)
    create_control_labels(merged_dataset_dict, disjunct=True)

    # print(merged_dataset_dict)

    with open(MERGED_SICK_FILEPATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged_dataset_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    create_merged_json()

    # for language in LANGUAGES:
    #     print(language)
    #     split= "train"

    #     dataset, dataloader = get_dataset_and_dataloader(language, split)

    #     print(dataset.sentence_pairs[:10])
