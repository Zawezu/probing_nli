import itertools
from sklearn.metrics import f1_score
from icecream import ic
import json
from pathlib import Path

from activations import get_number_of_layers_from_file, ActivationDataset
from utils import MODEL_NAMES, LANGUAGES, HYPERPARAMETERS_FILEPATH
from probes import LRProbe


def optimise_hyperparameters(
    activation_dataset_train,
    activation_dataset_val,
    param_grid: dict,
) -> dict:
    """
    Optimise hyperparameters for LRProbe using a validation set.
    Selects the combination maximising macro-averaged F1.

    Args:
        activation_dataset_train: ActivationDataset for training.
        activation_dataset_val: ActivationDataset for validation.
        param_grid: Dictionary of hyperparameter names to lists of values.
                    Every combination in the Cartesian product is evaluated.

    Returns:
        Dictionary of best hyperparameters found.
    """

    X_val = activation_dataset_val.activations.cpu().float().numpy()
    y_val = activation_dataset_val.labels.cpu().float().numpy()

    best_f1 = -1.0
    best_params = {k: v[0] for k, v in param_grid.items()}

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combo in itertools.product(*param_values):
        params = dict(zip(param_names, combo))

        probe = LRProbe.create_from_data(
            activation_dataset_train,
            C=params.get("C", 0.1),
            fit_intercept=params.get("fit_intercept", False),
        )

        val_preds = probe.pred(X_val)
        val_f1 = f1_score(y_val, val_preds, average="macro")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_params = params

    # print(f"Best hyperparameters: {best_params} (val macro-F1: {best_f1:.4f})")
    return best_params


def optimise_hyperparameters_all_layers(
    model_name: str,
    language: str,
    num_layers: int,
    param_grid: dict = {
        "C": [0.001, 0.01, 0.1, 1.0, 10.0],
        "fit_intercept": [True, False],
    },
) -> dict[int, dict]:
    """
    Optimise hyperparameters independently at each layer for a given model
    and language, using only the standard probing task for tuning.

    ActivationDatasets are constructed internally for the 'train' and 'val' splits
    at each layer. Only the 'standard' probing task is used for tuning.

    Args:
        model_name: Name of the LLM being probed (e.g., 'olmo_model').
        language: Language code (e.g., 'en', 'es').
        num_layers: Total number of layers to tune over.
        param_grid: Hyperparameter grid passed to optimise_hyperparameters.

    Returns:
        Dictionary mapping layer_num -> best hyperparameter dict.
    """
    best_params_per_layer: dict[int, dict] = {}

    for layer_num in range(num_layers):
        print(f"Tuning layer {layer_num} | model: {model_name} | language: {language}")

        train_dataset = ActivationDataset(
            language, "train", layer_num, "standard", model_name
        )

        val_dataset = ActivationDataset(
            language, "val", layer_num, "standard", model_name
        )

        best_params: dict = optimise_hyperparameters(
            activation_dataset_train=train_dataset,
            activation_dataset_val=val_dataset,
            param_grid=param_grid,
        )
        best_params_per_layer[layer_num] = best_params

    return best_params_per_layer


def save_hyperparameters(
    all_hyperparameters: dict,
    output_file: str = HYPERPARAMETERS_FILEPATH,
) -> str:
    """
    Save aggregated hyperparameters to a JSON file.

    Args:
        all_hyperparameters: Dictionary with structure:
            {model_name: {language: {layer_num: {hyperparams}}}}
        output_file: Path to save the JSON file

    Returns:
        Path to the saved file
    """
    output_path = Path(output_file)

    # Convert int keys to strings for JSON serialization
    serializable_hyperparams = {}
    for model_name, model_data in all_hyperparameters.items():
        serializable_hyperparams[model_name] = {}
        for language, lang_data in model_data.items():
            serializable_hyperparams[model_name][language] = {
                str(layer_num): params for layer_num, params in lang_data.items()
            }

    with open(output_path, "w") as f:
        json.dump(serializable_hyperparams, f, indent=2)

    print(f"Hyperparameters saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    model_names: list[str] = MODEL_NAMES
    languages: list[str] = LANGUAGES
    num_layers: int | None = None

    custom = False
    if custom:
        model_names = ["olmo_model"]
        languages = ["en"]
        num_layers = 1
        print("Using custom configuration")

    # exceptions: list[tuple[str, str]] = []
    exceptions = [("olmo_model", "en"), ("olmo_model", "es")]

    ic(custom, model_names, languages, num_layers, exceptions)

    # Aggregate hyperparameters for all model/language combinations
    all_hyperparameters: dict = {}

    for model_name in model_names:
        all_hyperparameters[model_name] = {}

        if num_layers is None:
            num_layers_for_this_model: int = get_number_of_layers_from_file(model_name)
        else:
            num_layers_for_this_model = num_layers

        for language in languages:
            if ((model_name, language)) not in exceptions:
                hyperparameters: dict = optimise_hyperparameters_all_layers(
                    model_name, language, num_layers_for_this_model
                )
                print(
                    f"Best hyperparameters for {model_name} in {language}:\n{hyperparameters}"
                )
                all_hyperparameters[model_name][language] = hyperparameters

    # Save all hyperparameters to JSON
    save_hyperparameters(all_hyperparameters)
