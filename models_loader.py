from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_FOLDER = "models"

olmo = AutoModelForCausalLM.from_pretrained("allenai/Olmo-3-1025-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/Olmo-3-1025-7B")

olmo_filepath = f"./{MODEL_FOLDER}/olmo_model"

olmo.save_pretrained(olmo_filepath)
tokenizer.save_pretrained(olmo_filepath)