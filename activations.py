from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm import tqdm
from pathlib import Path
import json

from sick import SICKMergedDataset
from common_constants import ACTIVATIONS_FOLDER, MODELS_FOLDER

device: t.device = t.device("cuda" if t.cuda.is_available() else "cpu")

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
        "Premise: A dog is running.\nHypothesis: An animal is moving.\nClassify: entailment, contradiction, or neutral.",
        "entailment",
    ),
    "es": (
        "Premisa: Un perro está corriendo.\nHipótesis: Un animal se está moviendo.\nClasifica: implicación, contradicción o neutral.",
        "implicación",
    ),
    "jp": (
        "前提：犬が走っている。\n仮説：動物が動いている。\n一言で分類してください：含意、矛盾、または中立。",
        "含意",
    ),
}


class SpecialCase:
    def __init__(self, language: str, split: str, start_from_batch: int | None) -> None:
        self.language: str = language
        self.split: str = split
        self.start_from_batch: int | None = start_from_batch

    def __str__(self) -> str:
        return f"SpecialCase(language={self.language}, split={self.split}, start_from_batch={self.start_from_batch})"


class ActivationSaver:
    def __init__(self, model_name, tokenizer=None, hf_model=None) -> None:
        self.activations: dict[str, t.Tensor] = {}
        self.model_name: str = model_name

        self.tokenizer = tokenizer
        self.hf_model = hf_model

    def get_activation(self, name):
        def hook(model, input, output) -> None:
            # 'output' is a tuple for some models; we want the first element (the tensor)
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach()
            else:
                self.activations[name] = output.detach()

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
        language: str,
        split: str,
        batch_id: int,
    ) -> None:
        responses_filepath = get_responses_filepath(
            self.model_name, language, split, batch_id
        )

        path_obj = Path(responses_filepath)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        records = [
            {"message": messages, "response": response}
            for messages, response in zip(messages_batch, responses)
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
        amount_to_generate=None,
        batch_size: int = 128,
        start_from_batch: int = 0,
    ) -> None:
        """
        Iterate through `dataloader` in batches, processing all sentences in each batch together.
        `start_from_batch` can be used to skip the first N batches so that you can resume
        after a crash.

        `amount_to_generate` limits the number of batches produced *after* the start
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
        try:
            for batch_num, (
                sentence_tuple_batch,
                _,
                _,
            ) in tqdm(
                enumerate(dataloader),
                desc="Extracting all layers",
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
                        max_new_tokens=5,
                        num_beams=1,
                        pad_token_id=self.tokenizer.eos_token_id,
                    )

                input_length = tokens.input_ids.shape[1]
                trimmed_responses = self.tokenizer.batch_decode(
                    generated_ids[:, input_length:],  # slice off the prompt tokens
                    skip_special_tokens=True,
                )

                # Extract activations for all samples in the batch
                for layer_num in range(n_layers):
                    acts: t.Tensor = self.activations[f"layer_{layer_num}"][
                        :, -1, :
                    ].cpu()

                    batch_acts_by_layer[layer_num] = acts

                # Clear GPU memory
                self.activations.clear()
                t.cuda.empty_cache()

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
                        language,
                        split,
                        batch_id,
                    )
                batch_id += 1

                if (
                    amount_to_generate
                    and batch_id - start_from_batch >= amount_to_generate
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
                # return (
                #     f"Premise: {sent_a}\nHypothesis: {sent_b}\n"
                #     f"Classify the relationship between the premise and hypothesis. "
                #     f"Reply with a single word: entailment, contradiction, or neutral."
                # )
                return (
                    f"Premise: {sent_a}\nHypothesis: {sent_b}\n"
                    f"Classify: entailment, contradiction, or neutral."  # shorter, no prose instruction
                )
            case "es":
                return (
                    f"Premisa: {sent_a}\nHipótesis: {sent_b}\n"
                    f"Clasifica la relación. Responde con una sola palabra: "
                    f"implicación, contradicción o neutral."
                )
            case "jp":
                return (
                    f"前提：{sent_a}\n仮説：{sent_b}\n"
                    f"関係を一言で分類してください：含意、矛盾、または中立。"
                )
            case _:
                raise KeyError(f"Language {language} is not supported")

    @staticmethod
    def generate_messages_batch(
        sentence_tuple_batch, language: str
    ) -> list[list[dict]]:
        system_prompt = SYSTEM_PROMPTS[language]
        few_shot_user, few_shot_assistant = FEW_SHOT_EXAMPLES[language]

        prompts: list[str] = [
            ActivationSaver.generate_prompt(sent_a, sent_b, language)
            for sent_a, sent_b in zip(sentence_tuple_batch[0], sentence_tuple_batch[1])
        ]

        return [
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": few_shot_user},
                {"role": "assistant", "content": few_shot_assistant},
                {"role": "user", "content": p},
            ]
            for p in prompts
        ]


def get_number_of_layers_from_file(model_name):
    with open(get_n_layers_txt_filepath(model_name), "r") as file:
        return int(file.readline())


def get_n_layers_txt_filepath(model_name) -> str:
    return f"{ACTIVATIONS_FOLDER}/{model_name}/n_layers.txt"


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
            original_dataset.get_labels_in_range(0, num_samples, self.probing_task)
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
    print(filepath.name)
    if ignore_substring and ignore_substring in filepath.name:
        print(f"Ignoring {filepath} because {ignore_substring} is in its filename")
    else:
        if actually_delete:
            filepath.unlink()
        print(f"Deleted {filepath}")


def delete_activations_file(
    model_name: str,
    language: str,
    split: str,
    layer_num: int | str | None = None,
    batch_id: int | None = None,
    ignore_substring: str = "",
    actually_delete: bool = True,
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
        pattern = f"layer{layer_num}_batch*.pt"
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
        pattern = "layer*_batch*.pt"
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


def generate_all_activations(
    activation_loader,
    languages_to_generate: list,
    splits_to_generate: list,
    amount_to_generate: int | None,
    save_to_disk: bool,
    batch_size: int,
    special_cases: list[SpecialCase] = [],
) -> None:
    for language in languages_to_generate:
        for split in splits_to_generate:
            print(f"{'-'*20}\nGenerating language {language}, split {split}\n{'-'*20}")
            start_from_batch: int = 0

            skip_split = False
            # Handle special cases where we want to start generating from another batch or skip a split.
            for special_case in special_cases:
                if special_case.language == language and special_case.split == split:
                    if special_case.start_from_batch is None:
                        skip_split = True
                        print(f"Handling special case: {special_case}. Skipping split")
                    else:
                        start_from_batch: int = special_case.start_from_batch
                        print(
                            f"Handling special case: {special_case}. Starting from batch {special_case.start_from_batch}"
                        )

            if not skip_split:
                activation_loader.generate_activations(
                    language,
                    split,
                    save_to_disk=save_to_disk,
                    amount_to_generate=amount_to_generate,
                    batch_size=batch_size,
                    start_from_batch=start_from_batch,
                )
