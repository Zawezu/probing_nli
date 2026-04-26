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
REVERSE_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Model constants
MODELS_FOLDER = "models"
MODEL_NAMES: list[str] = ["olmo_model", "tiny_aya_global"]
MODEL_IDS: dict[str, str] = {
    # "olmo_model": "allenai/Olmo-3-1025-7B", # This model is not instruction-tuned, so I no longer use it
    "olmo_model": "allenai/Olmo-3-7B-Instruct",  # This is the instruction-tuned version of the same Olmo model
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

# Hyperparameter constants

HYPERPARAMETERS_FILEPATH = "./data/hyperparameters/hyperparameters.json"

# os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

# Prompt constants

CHAT_TEMPLATES = {
    "olmo_model": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    ),
    "tiny_aya_global": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
        "{% elif message['role'] == 'user' %}<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
        "{% elif message['role'] == 'assistant' %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ message['content'] }}<|END_OF_TURN_TOKEN|>"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"
    ),
}

SYSTEM_PROMPTS = {
    "en": "You are a textual entailment classifier. Always respond with exactly one word: entailment, contradiction, or neutral.",
    "es": "Eres un clasificador de implicación textual. Responde siempre con una sola palabra: implicación, contradicción o neutral.",
    "jp": "あなたはテキスト含意分類器です。常に一言で答えてください：含意、矛盾、または中立。",
}
FEW_SHOT_EXAMPLES = {
    "en": (
        "Premise: A dog is running.\nHypothesis: An animal is moving.",
        "entailment",
    ),
    "es": (
        "Premisa: Un perro está corriendo.\nHipótesis: Un animal se está moviendo.",
        "implicación",
    ),
    "jp": (
        "前提：犬が走っている。\n仮説：動物が動いている。",
        "含意",
    ),
}


def get_number_of_layers_from_file(model_name) -> int:
    with open(get_n_layers_txt_filepath(model_name), "r") as file:
        return int(file.readline())


def get_n_layers_txt_filepath(model_name) -> str:
    return f"{ACTIVATIONS_FOLDER}/{model_name}/n_layers.txt"
