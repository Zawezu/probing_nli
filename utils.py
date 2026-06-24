from itertools import permutations, combinations
from typing import LiteralString
# import os

# SICK constants
SICK_FOLDER = "./data/sick"

SICK_DIRTY_FOLDERS: dict[str, str] = {
    "en": "sick_en",
    "es": "sick_es",
    "nl": "sick_nl",
    "jp": "jsick",
}
SICK_DIRTY_EN_FILE = "SICK_annotated.txt"
SICK_DIRTY_ES_FILE: dict[str, str] = {
    "train": "SICK_train.txt",
    "test": "SICK_test.txt",
    "trial": "SICK_trial.txt",
}
SICK_DIRTY_JP_FILE = "jsick.tsv"
SICK_DIRTY_NL_FILE = "SICK_NL.txt"

MERGED_SICK_FILEPATH: LiteralString = f"{SICK_FOLDER}/SICK_merged.json"

LABEL_MAP: dict[str, int] = {"entailment": 0, "neutral": 1, "contradiction": 2}
REVERSE_LABEL_MAP: dict[int, str] = {v: k for k, v in LABEL_MAP.items()}

# Model constants
MODELS_FOLDER = "models"
MODEL_NAMES: list[str] = ["olmo_model", "tiny_aya_global"]
MODEL_THESIS_NAMES: dict[str, str] = {"olmo_model": "olmo", "tiny_aya_global": "aya"}
MODEL_IDS: dict[str, str] = {
    # "olmo_model": "allenai/Olmo-3-1025-7B", # This model is not instruction-tuned, so I no longer use it
    "olmo_model": "allenai/Olmo-3-7B-Instruct",  # This is the instruction-tuned version of the same Olmo model
    "tiny_aya_global": "CohereLabs/tiny-aya-global",
}

# Activations constants
ACTIVATIONS_FOLDER = "./data/activations"

# Other constants
LANGUAGES: list[str] = ["en", "es", "nl", "jp"]
LANGUAGE_FULL_NAME_MAP: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "nl": "Dutch",
    "jp": "Japanese",
}
SPLITS: list[str] = ["train", "test", "val"]


def _verbose_lang_part(part: str) -> str:
    """Return the full name for a single language code, stripping _original_labels suffix if present."""
    _suffix = "_original_labels"
    if part.endswith(_suffix):
        name = LANGUAGE_FULL_NAME_MAP[part[: -len(_suffix)]]
        return f"{name} (ol)"
    return LANGUAGE_FULL_NAME_MAP[part]


def get_verbose_version_of_language_string(language: str):
    """Convert a language code or cross-language string to a human-readable description.

    For a plain code (e.g. 'en') returns the full name ('English').
    For a cross-language string like 'en→jp' returns e.g. 'trained in English, tested in
    Japanese'. When both sides refer to the same language (after stripping
    '_original_labels') returns e.g. 'trained and tested in Japanese'.
    """
    if "→" in language:
        parts: list[str] = language.split("→")
        # Strip _original_labels for same-language comparison
        _suffix = "_original_labels"
        bare_a = parts[0][: -len(_suffix)] if parts[0].endswith(_suffix) else parts[0]
        bare_b = parts[1][: -len(_suffix)] if parts[1].endswith(_suffix) else parts[1]
        name_a = _verbose_lang_part(parts[0])
        name_b = _verbose_lang_part(parts[1])
        if bare_a == bare_b:
            return f"trained and tested in {name_a}"
        else:
            return f"trained in {name_a}, tested in {name_b}"
    else:
        return _verbose_lang_part(language)


def _short_lang_part(part: str) -> str:
    """Return the short code for a single language component, keeping the
    '_original_labels' marker as a parenthetical (e.g. 'jp (ol)')."""
    _suffix = "_original_labels"
    if part.endswith(_suffix):
        return f"{part[: -len(_suffix)]} (ol)"
    return part


def get_short_version_of_language_string(language: str):
    """Like get_verbose_version_of_language_string but uses short language codes.

    For a plain code (e.g. 'en') returns the code itself ('en').
    For a cross-language string like 'en→jp' returns 'trained in en, tested in jp'.
    When both sides refer to the same language (after stripping '_original_labels')
    returns 'trained and tested in jp'. The grammar matches the verbose version so the
    same legend-parsing regexes apply.
    """
    if "→" in language:
        parts: list[str] = language.split("→")
        _suffix = "_original_labels"
        bare_a = parts[0][: -len(_suffix)] if parts[0].endswith(_suffix) else parts[0]
        bare_b = parts[1][: -len(_suffix)] if parts[1].endswith(_suffix) else parts[1]
        name_a = _short_lang_part(parts[0])
        name_b = _short_lang_part(parts[1])
        if bare_a == bare_b:
            return f"trained and tested in {name_a}"
        else:
            return f"trained in {name_a}, tested in {name_b}"
    else:
        return _short_lang_part(language)


# Experiment constants
EXPERIMENT_RESULTS_FOLDER = "./data/experiment_results"
PLOTS_FOLDER = "./plots"
PROBES_FOLDER = "./data/probes"

PROBE_TYPE_SUBFOLDERS: dict[str, str] = {
    "lr": "logistic_regression",
    "mm": "mass_mean",
}

PROBE_TYPE_FULL_NAME_MAP: dict[str, str] = {
    "lr": "logistic regression",
    "mm": "mass mean",
}

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
    "nl": "Jij bent een classificator van tekstuele implicaties. Reageer altijd met precies één woord: implicatie, tegenspraak of neutraal.",
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
    "nl": (
        "Premisse: Een hond rent.\nHypothese: Een dier beweegt.",
        "implicatie",
    ),
    "jp": (
        "前提：犬が走っている。\n仮説：動物が動いている。",
        "含意",
    ),
}

# Significance results constants

SIGNIFICANCE_RESULTS_FOLDER = "./data/significance_results"

# Miscellaneous functions


def get_number_of_layers_from_file(model_name) -> int:
    """Read the layer count written by ActivationRecorder.load_model from a text file."""
    with open(get_n_layers_txt_filepath(model_name), "r") as file:
        return int(file.readline())


def get_n_layers_txt_filepath(model_name) -> str:
    """Return the path to the text file that caches the model's layer count."""
    return f"{ACTIVATIONS_FOLDER}/{model_name}/n_layers.txt"


def get_language_merged_string(language_pair: tuple[str, str]) -> str:
    """Encode a (train_lang, test_lang) pair as a single arrow-separated string.

    ExperimentResult stores cross-lingual results under a single language key;
    this function produces that key (e.g. ('en', 'jp') -> 'en→jp').
    """
    return f"{language_pair[0]}→{language_pair[1]}"


def get_all_language_merged_strings(language_pairs: list[tuple[str, str]]) -> list[str]:
    """Apply `get_language_merged_string` to every pair and return the resulting list."""
    merged_string_list: list[str] = []
    for pair in language_pairs:
        merged_string_list.append(get_language_merged_string(pair))
    return merged_string_list


def get_language_pair_permutations(languages: list[str]) -> list[tuple[str, str]]:
    """
    Gets a list of all permuations (order matters) of language pairs given a list of languages
    """
    return list(permutations(languages, 2))


def get_language_pair_combinations(languages: list[str]) -> list[tuple[str, str]]:
    """
    Gets a list of all combinations (order doesn't matter) of language pairs given a list of languages
    """
    return list(combinations(languages, 2))
