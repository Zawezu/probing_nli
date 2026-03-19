from typing import LiteralString

# SICK constants
SICK_FOLDER = "data/sick"

SICK_DIRTY_FOLDERS: dict[str, str] = {"en": "sick_en", "es": "sick_es"}
SICK_DIRTY_EN_FILE = "SICK_annotated.txt"
SICK_DIRTY_ES_FILE: dict[str, str] = {
    "train": "SICK_train.txt",
    "test": "SICK_test.txt",
    "trial": "SICK_trial.txt",
}

MERGED_SICK_FILEPATH: LiteralString = f"{SICK_FOLDER}/SICK_merged.json"

LABEL_MAP: dict[str, int] = {"entailment": 0, "neutral": 1, "contradiction": 2}

# Model constants
MODEL_FOLDER = "models"
MODEL_FILEPATHS: dict[str, str] = {"olmo_model": f"./{MODEL_FOLDER}/olmo_model"}

# Activations constants
ACTIVATIONS_PATH = "./data/activations/"

# Other constants
LANGUAGES: list[str] = ["en", "es"]
SPLITS: list[str] = ["train", "test", "val"]
