import sick_loader
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_FOLDER = "models"

if __name__ == "__main__":
    sick_en_dataset_train = sick_loader.SICKDataset("en", "train")
    # sick_en_dataset_test = sick_loader.SICKDataset("en", "test")
    # sick_en_dataset_val = sick_loader.SICKDataset("en", "val")

    olmo_filepath = f"./{MODEL_FOLDER}/olmo_model"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(olmo_filepath, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(olmo_filepath, local_files_only=True).to(device)

    sentence_pair, label = sick_en_dataset_train[1]
    prompt = f"""
Choose entailmnent, neutral, or contradiction. Answer only in one word:
Sentence 1: {sentence_pair[0]}
Sentence 2: {sentence_pair[1]}
Final answer:
            """
    print(f"Prompt: {prompt}\n--------------\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(**inputs, max_new_tokens=50)
    print(f"Model answer: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")