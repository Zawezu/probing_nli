from typing import LiteralString
# import os

# SICK constants
SICK_FOLDER = "./data/sick"

SICK_DIRTY_FOLDERS: dict[str, str] = {"en": "sick_en", "es": "sick_es", "jp": "jsick"}
SICK_DIRTY_EN_FILE = "SICK_annotated.txt"
SICK_DIRTY_ES_FILE: dict[str, str] = {
    "train": "SICK_train.txt",
    "test": "SICK_test.txt",
    "trial": "SICK_trial.txt",
}
SICK_DIRTY_JP_FILE = "jsick.tsv"

MERGED_SICK_FILEPATH: LiteralString = f"{SICK_FOLDER}/SICK_merged.json"

LABEL_MAP: dict[str, int] = {"entailment": 0, "neutral": 1, "contradiction": 2}

# Model constants
MODELS_FOLDER = "models"
MODEL_NAMES: list[str] = ["olmo_model", "tiny_aya_global"]
MODEL_IDS: dict[str, str] = {
    "olmo_model": "allenai/Olmo-3-1025-7B",
    "tiny_aya_global": "CohereLabs/tiny-aya-global",
}

# Activations constants
ACTIVATIONS_FOLDER = "./data/activations"

# Other constants
LANGUAGES: list[str] = ["en", "es", "jp"]
LANGUAGE_FULL_NAME_MAP: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "jp": "Japanese",
    "en→es": "trained in English, tested in Spanish",
    "en→jp": "trained in English, tested in Japanese",
    "es→en": "trained in Spanish, tested in English",
    "es→jp": "trained in Spanish, tested in Japanese",
    "jp→en": "trained in Japanese, tested in English",
    "jp→es": "trained in Japanese, tested in Spanish",
}
SPLITS: list[str] = ["train", "test", "val"]

# Experiment constants
EXPERIMENT_RESULTS_FOLDER = "./data/experiment_results"
PLOTS_FOLDER = "./plots"
PROBES_FOLDER = "./data/probes"

PROBING_TASKS: list[str] = ["standard", "control", "disjunct_control"]

# os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
