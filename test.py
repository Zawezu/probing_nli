# from utils import MODELS_FOLDER

# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch as t

# device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")


# def run_nli_inference():
#     """Run NLI task inference with any model."""
#     sentence_tuple_batch: list[tuple[str, str]] = [
#         (
#             "Three boys are jumping in the leaves",
#             "Three kids are jumping in the leaves",
#         ),
#         (
#             "Two women are sparring in a kickboxing match",
#             "Two people are kickboxing and spectators are not watching",
#         ),
#     ]

#     prompts: list[str] = [
#         f"Premise: {sent_a} Hypothesis: {sent_b} Label:"
#         for sent_a, sent_b in zip(sentence_tuple_batch[0], sentence_tuple_batch[1])
#     ]

#     # Tokenize entire batch at once
#     tokens = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

#     with t.no_grad():
#         response = hf_model(**tokens)

#     # Decode the first item in the batch
#     for tokens_to_decode in response["input_ids"]:
#         gen_text = tokenizer.decode(tokens_to_decode)
#         print("Decoded text:")
#         print(gen_text)


# if __name__ == "__main__":
#     model_name: str = "olmo_model"
#     # model_name: str = "tiny_aya_global"
#     model_filepath: str = f"{MODELS_FOLDER}/{model_name}"

#     tokenizer = AutoTokenizer.from_pretrained(model_filepath, local_files_only=True)
#     hf_model = AutoModelForCausalLM.from_pretrained(
#         model_filepath, local_files_only=True
#     ).to(device)  # type: ignore

#     run_nli_inference()
import pickle
from utils import LANGUAGES, MODEL_NAMES

for language in LANGUAGES:
    for model_name in MODEL_NAMES:
        result = pickle.load(
            open(
                f"./data/experiment_results/experiment_1/{language},standard,lr,{model_name}.pkl",
                "rb",
            )
        )
        print("idxs_per_cm_cell" in result.metrics["test"].keys())

        result = pickle.load(
            open(
                f"./data/experiment_results/experiment_1/{language},control,lr,{model_name}.pkl",
                "rb",
            )
        )
        print("idxs_per_cm_cell" in result.metrics["test"].keys())
