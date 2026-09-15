import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import MODELS_FOLDER, MODEL_IDS, MODEL_NAMES
from pathlib import Path

# Some models (e.g. CohereLabs/aya-expanse-8b) live in gated repositories and can only
# be downloaded with an access token from an account that has accepted their terms.
# The token is read from this file if it exists; otherwise the HF_TOKEN environment
# variable is used, which is what `huggingface-cli login` sets up.
HUGGINGFACE_KEY_FILEPATH = "huggingface_key.txt"


def get_huggingface_token() -> str | None:
    """Return the HuggingFace access token, or None if none is configured.

    Reads HUGGINGFACE_KEY_FILEPATH first (it is gitignored, so the key never gets
    committed) and falls back to the HF_TOKEN environment variable.
    """
    key_filepath = Path(HUGGINGFACE_KEY_FILEPATH)
    if key_filepath.exists():
        token: str = key_filepath.read_text(encoding="utf-8").strip()
        if token:
            return token

    return os.environ.get("HF_TOKEN")


def load_and_save_model(model_name: str) -> None:
    """Download a model from HuggingFace Hub and save it to the local models directory.

    Args:
        model_name: Key in MODEL_IDS / MODEL_NAMES (e.g. 'olmo_model').
    """
    model_id: str = MODEL_IDS[model_name]
    token: str | None = get_huggingface_token()

    # dtype="auto" keeps the checkpoint in its published precision (bfloat16 for all
    # the models used here) instead of upcasting it to float32, which matters for the
    # 7-8B models both while downloading and on disk.
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    filepath: str = f"{MODELS_FOLDER}/{model_name}"

    # Create directory if it doesn't exist
    Path(filepath).mkdir(parents=True, exist_ok=True)

    model.save_pretrained(filepath)
    tokenizer.save_pretrained(filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="model names", nargs="*", default=MODEL_NAMES)
    args = parser.parse_args()

    for model_name in args.m:
        print(f"Downloading {model_name} ({MODEL_IDS[model_name]})")
        load_and_save_model(model_name)
