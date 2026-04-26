from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
import json
import sys
from icecream import ic

from sick import SICKMergedDataset
from utils import (
    ACTIVATIONS_FOLDER,
    MODELS_FOLDER,
    CHAT_TEMPLATES,
    SYSTEM_PROMPTS,
    FEW_SHOT_EXAMPLES,
    SPLITS,
    get_n_layers_txt_filepath,
    get_number_of_layers_from_file,
)

device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")


class SpecialCase:
    def __init__(self, language: str, split: str, start_from_batch: int | None) -> None:
        self.language: str = language
        self.split: str = split
        self.start_from_batch: int | None = start_from_batch

    def __str__(self) -> str:
        return f"SpecialCase(language={self.language}, split={self.split}, start_from_batch={self.start_from_batch})"


class ActivationRecorder:
    def __init__(self, model_name, tokenizer=None, hf_model=None) -> None:
        self.activations: dict[str, t.Tensor] = {}
        self.model_name: str = model_name

        self.tokenizer = tokenizer
        self.hf_model = hf_model

    def get_activation(self, name):
        def hook(model, input, output) -> None:
            # We only record the activations if the dictionary does not have an entry for them
            # This ensures that the only activations recorded are those for the first forward pass
            if name not in self.activations.keys():
                # print("Recording activations")
                # 'output' is a tuple for some models; we want the first element (the tensor)
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach()
                else:
                    self.activations[name] = output.detach()
            # else:
            #     print("Activations already recorded")

        return hook

    def _save_batch(
        self,
        acts_by_layer: dict[int, t.Tensor],
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        """write a single batch of activations for every layer."""
        for layer_num, acts in acts_by_layer.items():
            data_to_save: dict[str, t.Tensor | dict[str, int | str]] = {
                "activations": acts,
                "metadata": {
                    "layer": layer_num,
                    "model": self.model_name,
                    "batch_id": batch_id,
                },
            }
            save_path: str = get_activations_filepath(
                self.model_name, language, split, layer_num, batch_id
            )

            # If the folder doesn't exist, create it
            path_obj = Path(save_path)
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            t.save(data_to_save, save_path)
            # print(
            #     f"Saved {len(acts)} samples for layer {layer_num} (batch {batch_id}) to {save_path}"
            # )

        print(
            f"Saved samples for batch {batch_id} to {get_activations_filepath(self.model_name, language, split, 'n', batch_id)}"
        )

    def _save_batch_responses(
        self,
        messages_batch: list[list[dict]],
        responses: list[str],
        original_ids: list[int],
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        assert (
            len(messages_batch) == len(responses) == len(original_ids)
        ), "Cannot save batch responses. The lengths of the lists do not match."

        responses_filepath = get_responses_filepath(
            self.model_name, language, split, batch_id
        )

        path_obj = Path(responses_filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        records = [
            {"message": messages, "response": response, "original_id": original_id}
            for messages, response, original_id in zip(
                messages_batch, responses, original_ids
            )
        ]

        with open(responses_filepath, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(
            f"Saved {len(records)} responses for batch {batch_id} to {responses_filepath}"
        )

    def generate_activations(
        self,
        language: str,
        split: str,
        save_to_disk=True,
        amount_of_batches_to_generate=None,
        batch_size: int = 128,
        start_from_batch: int = 0,
    ) -> None:
        """
        Iterate through `dataloader` in batches, processing all sentences in each batch together.
        `start_from_batch` can be used to skip the first N batches so that you can resume
        after a crash.

        `amount_of_batches_to_generate` limits the number of batches produced *after* the start
        index.
        """
        if self.tokenizer is None or self.hf_model is None:
            print(
                f"The tokenizer or model were not loaded for the ActivationLoader of {self.model_name}. Loading now."
            )
            self.load_model()

        assert self.tokenizer is not None and self.hf_model is not None

        dataset: SICKMergedDataset = SICKMergedDataset(language, split)
        dataloader = DataLoader(dataset, batch_size, shuffle=False)

        n_layers: int = len(self.hf_model.model.layers)

        # register hooks
        hook_handles = []
        for layer_num in range(n_layers):
            handle = self.hf_model.model.layers[layer_num].register_forward_hook(
                self.get_activation(f"layer_{layer_num}")
            )
            hook_handles.append(handle)

        batch_id: int = start_from_batch

        len_dataloader: int = len(dataloader)
        print(len(dataloader))

        # Olmo needs more tokens to produce Japanese responses (although it still does very poorly with 16)
        if self.model_name == "olmo_model" and language == "jp":
            max_new_tokens: int = 16
        # In all other cases, 4 new tokens is enough to generate a meaningful response most of the time
        else:
            max_new_tokens = 4

        try:
            for batch_num, (
                sentence_tuple_batch,
                _,
                original_ids,
            ) in tqdm(
                enumerate(dataloader),
                desc="Extracting all layers per batch",
                total=len_dataloader,
            ):
                if batch_num < start_from_batch:  # skip until we reach the resume point
                    continue

                batch_acts_by_layer: dict[int, t.Tensor] = {}

                messages_batch = self.generate_messages_batch(
                    sentence_tuple_batch, language
                )

                tokens = self.tokenizer.apply_chat_template(
                    messages_batch,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    padding=True,
                ).to(device)

                with t.no_grad():
                    generated_ids = self.hf_model.generate(
                        input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        max_new_tokens=max_new_tokens,
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                # Extract activations for all samples in the batch from all layers
                for layer_num in range(n_layers):
                    acts: t.Tensor = self.activations[f"layer_{layer_num}"][
                        :, -1, :
                    ].cpu()

                    batch_acts_by_layer[layer_num] = acts

                # Clear GPU memory
                self.activations.clear()
                t.cuda.empty_cache()

                input_length = tokens.input_ids.shape[1]
                trimmed_responses = [
                    self.tokenizer.decode(
                        ids[input_length:],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    .encode("utf-8", errors="replace")
                    .decode("utf-8")
                    for ids in generated_ids
                ]

                # print(f"batch_acts_by_layer[0]:\n{batch_acts_by_layer[0]}")
                # print(f"batch_acts_by_layer[1]:\n{batch_acts_by_layer[1]}")

                # Save batch and reset
                if save_to_disk:
                    self._save_batch(
                        batch_acts_by_layer,
                        language,
                        split,
                        batch_id,
                    )
                    self._save_batch_responses(
                        messages_batch,
                        trimmed_responses,
                        original_ids,
                        language,
                        split,
                        batch_id,
                    )
                batch_id += 1

                if (
                    amount_of_batches_to_generate
                    and batch_id - start_from_batch >= amount_of_batches_to_generate
                ):
                    break
        finally:
            for handle in hook_handles:
                handle.remove()

    def load_model(self) -> None:
        model_filepath: str = f"{MODELS_FOLDER}/{self.model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_filepath, local_files_only=True
        )

        self.tokenizer.chat_template = CHAT_TEMPLATES[self.model_name]

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            model_filepath, local_files_only=True
        ).to(device)  # type: ignore

        n_layers_txt_filepath: str = get_n_layers_txt_filepath(self.model_name)
        # Create parent directory if it doesn't exist
        path_obj = Path(n_layers_txt_filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(n_layers_txt_filepath, "w") as file:
            file.write(str(len(self.hf_model.model.layers)))

    def get_number_of_layers(self) -> int:
        if self.hf_model is not None:
            return len(self.hf_model.model.layers)
        else:
            print(
                f"Model not loaded. Getting the number of layers from {get_n_layers_txt_filepath(self.model_name)}"
            )
            try:
                return get_number_of_layers_from_file(self.model_name)
            except FileNotFoundError:
                print("Could not find . Loading model")
                self.load_model()
                return self.get_number_of_layers()

    @staticmethod
    def generate_prompt(sent_a, sent_b, language) -> str:
        match language:
            case "en":
                return f"Premise: {sent_a}\nHypothesis: {sent_b}\nClassification: "
            case "es":
                return f"Premisa: {sent_a}\nHipótesis: {sent_b}\nClasificación: "
            case "jp":
                return f"前提：{sent_a}\n仮説：{sent_b}\n分類："
            case _:
                raise KeyError(f"Language {language} is not supported")

    @staticmethod
    def generate_messages_batch(
        sentence_tuple_batch, language: str, few_shot: bool = False
    ) -> list[list[dict]]:
        system_prompt = SYSTEM_PROMPTS[language]

        prompts: list[str] = [
            ActivationRecorder.generate_prompt(sent_a, sent_b, language)
            for sent_a, sent_b in zip(sentence_tuple_batch[0], sentence_tuple_batch[1])
        ]

        # If few_shot, give an example of a NLI answer. This is off by default, since few-shot
        # may interfere with the probing results.
        if few_shot:
            few_shot_user, few_shot_assistant = FEW_SHOT_EXAMPLES[language]
            return [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": few_shot_user},
                    {"role": "assistant", "content": few_shot_assistant},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]
        else:
            return [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": p},
                ]
                for p in prompts
            ]

    def generate_all_activations(
        self,
        languages_to_generate: list,
        splits_to_generate: list,
        amount_of_batches_to_generate: int | None,
        save_to_disk: bool,
        batch_size: int,
        special_cases: list[SpecialCase] = [],
    ) -> None:
        for language in languages_to_generate:
            for split in splits_to_generate:
                print(
                    f"{'-'*20}\nGenerating language {language}, split {split}\n{'-'*20}"
                )
                start_from_batch: int = 0

                skip_split = False
                # Handle special cases where we want to start generating from another batch or skip a split.
                for special_case in special_cases:
                    if (
                        special_case.language == language
                        and special_case.split == split
                    ):
                        if special_case.start_from_batch is None:
                            skip_split = True
                            print(
                                f"Handling special case: {special_case}. Skipping split"
                            )
                        else:
                            start_from_batch: int = special_case.start_from_batch
                            print(
                                f"Handling special case: {special_case}. Starting from batch {special_case.start_from_batch}"
                            )

                if not skip_split:
                    self.generate_activations(
                        language,
                        split,
                        save_to_disk=save_to_disk,
                        amount_of_batches_to_generate=amount_of_batches_to_generate,
                        batch_size=batch_size,
                        start_from_batch=start_from_batch,
                    )


class ActivationDataset(Dataset):
    def __init__(
        self,
        language: str,
        split: str,
        layer_num: int,
        probing_task: str,
        model_name: str,
    ) -> None:
        self.language: str = language
        self.split: str = split
        self.layer_num: int = layer_num
        self.probing_task: str = probing_task
        self.model_name: str = model_name
        self.original_dataset: SICKMergedDataset = SICKMergedDataset(language, split)

        self.activations, self.labels = self.load_activations_from_merged()

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.activations[i], self.labels[i]

    def __len__(self) -> int:
        return len(self.activations)

    def load_activations_from_merged(self) -> tuple[Tensor, Tensor]:
        """
        Load activations and labels from a merged activation file.
        """
        original_dataset: SICKMergedDataset = SICKMergedDataset(
            self.language, self.split
        )

        # Load merged activation file
        merged_filepath: str = f"{get_activations_filepath(self.model_name, self.language, self.split, self.layer_num, None)}_merged.pt"

        if not Path(merged_filepath).exists():
            raise FileNotFoundError(
                f"Merged activation file not found at {merged_filepath}. "
                f"Make sure activations have been merged for {self.model_name}, "
                f"language={self.language}, split={self.split}, layer={self.layer_num}"
            )

        data = t.load(merged_filepath, weights_only=True)
        activations: Tensor = data["activations"]

        # Get labels for all samples
        num_samples = len(activations)
        labels: Tensor = t.IntTensor(
            original_dataset.get_labels(0, num_samples, self.probing_task)
        )

        return activations, labels


def get_activations_filepath(
    model_name: str,
    language: str,
    split: str,
    layer_num: int | str | None,
    batch_id: int | str | None,
) -> str:
    if layer_num is not None and batch_id is not None:
        # if type(batch_id) == int:
        if isinstance(batch_id, int):
            return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}_batch{batch_id}.pt"
        # elif type(batch_id) == str:
        elif isinstance(batch_id, str):
            return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}_{batch_id}.pt"
    elif layer_num is not None and batch_id is None:
        return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}/layer{layer_num}"
    elif layer_num is None and batch_id is not None:
        raise KeyError("If layer_num is None, batch_id cannot be None")
    else:
        return f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}"


def get_responses_filepath(
    model_name: str,
    language: str,
    split: str,
    batch_id: int | str,
) -> str:
    return f"./data/responses/{model_name}/{language}/{split}/responses_batch{batch_id}.json"


def delete_individual_file(filepath, ignore_substring, actually_delete) -> None:
    # print(filepath.name)
    if not (ignore_substring and ignore_substring in filepath.name):
        if actually_delete:
            filepath.unlink()
            print(f"Deleted {filepath}")
        else:
            print(f"Would delete {filepath}")
    # else:
    #     print(f"Ignoring {filepath} because {ignore_substring} is in its filename")


def delete_activations_file(
    model_name: str,
    language: str,
    split: str,
    layer_num: int | str | None = None,
    batch_id: int | None = None,
    ignore_substring: str = "",
    actually_delete: bool = False,
) -> None:
    """
    Delete activation pt file(s).

    If both layer_num and batch_id are provided, deletes the specific file.
    If batch_id is None and layer_num is provided, deletes all batches for that layer.
    If layer_num is None and batch_id is provided, deletes all layers for that batch.
    If both are None, deletes all activations for this model/language/split.
    """
    directory = Path(get_activations_filepath(model_name, language, split, None, None))

    if not directory.exists():
        print(f"Directory not found: {directory}")
        return

    deleted_count = 0
    if layer_num is not None and batch_id is not None:
        # Delete specific file
        filepath = Path(
            get_activations_filepath(model_name, language, split, layer_num, batch_id)
        )
        delete_individual_file(filepath, ignore_substring, actually_delete)
        deleted_count = 1
    elif layer_num is not None and batch_id is None:
        # Delete all batches for this layer
        pattern = f"layer{layer_num}_*.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1
    elif layer_num is None and batch_id is not None:
        # Delete all layers for this batch
        pattern = f"layer*_batch{batch_id}.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1
    else:
        # Delete all activations for this model/language/split
        pattern = "layer*_*.pt"
        for filepath in directory.glob(pattern):
            delete_individual_file(filepath, ignore_substring, actually_delete)
            deleted_count += 1

    if deleted_count == 0:
        print(f"No files found matching the criteria in {directory}")


def merge_activation_batches(
    model_name: str, language: str, split: str, layer_num: int | str
) -> str:
    # Get the directory containing batch files
    directory = Path(f"{ACTIVATIONS_FOLDER}/{model_name}/{language}/{split}")

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    # Find all batch files for this layer, sorted numerically by batch number
    pattern: str = f"layer{layer_num}_batch*.pt"
    batch_files: list[Path] = sorted(
        directory.glob(pattern), key=lambda p: int(p.stem.split("_batch")[1])
    )

    if not batch_files:
        raise FileNotFoundError(
            f"No batch files found for {model_name}/{language}/{split}/layer{layer_num}"
        )

    # Load and concatenate all activations
    all_activations: list[Tensor] = []
    for batch_file in batch_files:
        data = t.load(batch_file)
        all_activations.append(data["activations"])

    # Concatenate all activations along the batch dimension
    merged_activations: Tensor = t.cat(all_activations, dim=0)

    # Save merged file
    merged_filepath: str = f"{get_activations_filepath(model_name, language, split, layer_num, None)}_merged.pt"

    data_to_save = {
        "activations": merged_activations,
        "metadata": {
            "layer": layer_num,
            "model": model_name,
            "merged": True,
            "num_batches": len(batch_files),
        },
    }

    t.save(data_to_save, merged_filepath)
    print(
        f"Merged {len(batch_files)} batch files for {model_name}, {language}, {split}, layer{layer_num}"
    )
    print(f"Merged activations shape: {merged_activations.shape}")
    print(f"Saved to {merged_filepath}")

    return merged_filepath


def merge_response_batches(model_name: str, language: str, split: str) -> str:
    directory = Path(f"./data/responses/{model_name}/{language}/{split}")

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    pattern = "responses_batch*.json"
    response_files = sorted(
        directory.glob(pattern), key=lambda p: int(p.stem.split("_batch")[1])
    )

    if not response_files:
        raise FileNotFoundError(
            f"No response files found for {model_name}/{language}/{split}"
        )

    all_records: list[dict] = []
    for response_file in response_files:
        with open(response_file, "r", encoding="utf-8") as f:
            all_records.extend(json.load(f))

    merged_filepath = (
        f"./data/responses/{model_name}/{language}/{split}/responses_merged.json"
    )

    with open(merged_filepath, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print(
        f"Merged {len(response_files)} response files for {model_name}, {language}, {split}"
    )
    print(f"Total responses: {len(all_records)}")
    print(f"Saved to {merged_filepath}")

    return merged_filepath


if __name__ == "__main__":
    save_to_disk: bool = True
    amount_of_batches_to_generate: int | None = None
    splits_to_generate: list[str] = SPLITS

    debug = False
    if debug:
        splits_to_generate = ["train"]
        amount_of_batches_to_generate = 2
        batch_size = 4
        print("Running in debug mode")

    ic(save_to_disk, amount_of_batches_to_generate, splits_to_generate)

    assert save_to_disk, "If generating, must also save to disk"

    # ------------------------------

    model_name: str = sys.argv[1]
    languages_to_generate_arg: str = sys.argv[2]
    batch_size = int(sys.argv[3])

    print(f"model_name = {model_name}")
    print(f"batch_size = {batch_size}")

    languages_to_generate: list[str] = languages_to_generate_arg.split(",")

    print(f"languages_to_generate = {languages_to_generate}")

    # ------------------------------

    activation_recorder: ActivationRecorder = ActivationRecorder(model_name)

    special_cases: list[SpecialCase] = []

    activation_recorder.generate_all_activations(
        languages_to_generate,
        splits_to_generate,
        amount_of_batches_to_generate,
        save_to_disk,
        batch_size,
        special_cases=special_cases,
    )
