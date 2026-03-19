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

    def __getitem__(self, index: int) -> tuple[tuple[str, str], int, int]:
        return self.sentence_pairs[index], self.labels[index], self.original_ids[index]

    def __len__(self) -> int:
        return len(self.sentence_pairs)


class SICKMergedDataset(Dataset):
    def __init__(self, language, split) -> None:
        self.sentence_pairs: list[tuple[str, str]] = []
        self.standard_labels: list[int] = []
        self.control_labels: list[int] = []
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

                self.standard_labels.append(int(values["standard_label"]))
                self.control_labels.append(int(values["control_label"]))
                self.original_ids.append(int(id))

    def __getitem__(self, index: int) -> tuple[tuple[str, str], int, int, int]:
        return (
            self.sentence_pairs[index],
            self.standard_labels[index],
            self.control_labels[index],
            self.original_ids[index],
        )

    def __len__(self) -> int:
        return len(self.sentence_pairs)


def get_dataset_and_dataloader(
    language, split, batch_size=1
) -> tuple[SICKMergedDataset, DataLoader[tuple[tuple[str, str], int, int, int]]]:
    dataset: SICKMergedDataset = SICKMergedDataset(language, split)

    dataloader: DataLoader[tuple[tuple[str, str], int, int, int]] = DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataset, dataloader


def add_to_dict(dictionary, key, value) -> None:
    try:
        assert dictionary[key] == value
    except KeyError:
        dictionary[key] = value


def create_control_labels(dataset_dict, disjunct: bool, label_ratio=None) -> None:
    """
    disjunt: determines whether the label at some line is necessarily different from the original label
    """
    unique_labels: list[int] = list(LABEL_MAP.values())
    # If we don't specify the label_ratio, use the same ratio as in the real labels
    if label_ratio is None:
        amount_per_label: dict[int, int] = {label: 0 for label in unique_labels}
        for tup in dataset_dict.values():
            label: int = tup["label"]
            amount_per_label[label] += 1
    else:
        amount_per_label = label_ratio

    if disjunct:
        assert (
            label_ratio is None
        ), "label_ratio should not be specified if disjunct is true"

        # Create dictionary with the probabilities when we exclude each of the labels
        total_amount: int = sum(amount_per_label.values())
        probabilities: dict[str, dict[int, float]] = {}
        for label_excluded, amount_excluded in amount_per_label.items():
            probabilities[f"{label_excluded}_out"] = {}
            # Calculate the probability of each label if we exclude a particular label
            for label, amount in amount_per_label.items():
                probabilities[f"{label_excluded}_out"][label] = amount / (
                    total_amount - amount_excluded
                )
            # Forcefully exclude the label by setting its probability to 0
            probabilities[f"{label_excluded}_out"][label_excluded] = 0

        # print(probabilities)

        for id, values in dataset_dict.items():
            original_label: int = values["label"]
            probabilities_used: list[float] = list(
                probabilities[f"{original_label}_out"].values()
            )
            # print(f"probabilities_used={probabilities_used}")
            disjunct_control_label: int = random.choices(
                unique_labels, weights=probabilities_used, k=1
            )[0]
            dataset_dict[id]["disjunct_control_label"] = disjunct_control_label
    else:
        # Create list with the correct amounts of each label
        control_labels: list[int] = []
        for label, amount in amount_per_label.items():
            control_labels += [label] * amount

        # Shuffle the list
        random.shuffle(control_labels)

        # Add the control label to each row in the dataset
        for id in dataset_dict.keys():
            dataset_dict[id]["control_label"] = control_labels.pop(0)


def create_merged_dataset() -> None:
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
                add_to_dict(merged_dataset_dict[id], "label", label.item())
                add_to_dict(merged_dataset_dict[id], "split", split)

    create_control_labels(merged_dataset_dict, disjunct=False)
    create_control_labels(merged_dataset_dict, disjunct=True)

    # print(merged_dataset_dict)

    with open(MERGED_SICK_FILEPATH, "w", encoding="utf-8") as f:
        f.write(json.dumps(merged_dataset_dict, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    create_merged_dataset()

    # for language in LANGUAGES:
    #     print(language)
    #     split= "train"

    #     dataset, dataloader = get_dataset_and_dataloader(language, split)

    #     print(dataset.sentence_pairs[:10])
