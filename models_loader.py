from transformers import AutoModelForCausalLM, AutoTokenizer
from common_constants import MODELS_FOLDER, MODEL_IDS


def load_and_save_model(model_name: str) -> None:
    model_id: str = MODEL_IDS[model_name]

    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    filepath: str = f"{MODELS_FOLDER}/{model_name}"

    model.save_pretrained(filepath)
    tokenizer.save_pretrained(filepath)


if __name__ == "__main__":
    # load_and_save_model("olmo_model")

    load_and_save_model("tiny_aya_global")
