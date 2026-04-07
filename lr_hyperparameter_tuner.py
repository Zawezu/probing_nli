import itertools
from sklearn.metrics import f1_score
from icecream import ic

from activations import get_number_of_layers_from_file, ActivationDataset
from common_constants import MODEL_NAMES, LANGUAGES
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
        activation_dataset_train: ActivationDataset for training
        activation_dataset_val: ActivationDataset for validation
        param_grid: Dictionary of hyperparameter names to lists of values.
                    If None, uses a sensible default grid.

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

    print(f"Best hyperparameters: {best_params} (val macro-F1: {best_f1:.4f})")
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

    Args:
        model_name: Name of the LLM being probed (e.g., 'olmo_model')
        language: Language code (e.g., 'en', 'es')
        num_layers: Total number of layers to tune over
        get_activation_dataset_fn: Callable with signature
            (model_name, language, layer_num, split, probing_task)
            -> ActivationDataset
            where split is 'train' or 'val' and probing_task is e.g. 'standard'
        param_grid: Hyperparameter grid passed to optimise_hyperparameters.
                    If None, uses the default grid.

    Returns:
        Dictionary mapping layer_num -> best hyperparameter dict
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


if __name__ == "__main__":
    model_names: list[str] = MODEL_NAMES
    languages: list[str] = LANGUAGES
    num_layers: int | None = None

    custom = True
    if custom:
        model_names = ["olmo_model"]
        languages = ["en"]
        num_layers = 3
        print("Using custom configuration")

    ic(custom, model_names, languages, num_layers)

    for model_name in model_names:
        if num_layers is None:
            num_layers_for_this_model: int = get_number_of_layers_from_file(model_name)
        else:
            num_layers_for_this_model = num_layers

        for language in languages:
            hyperparameters: dict = optimise_hyperparameters_all_layers(
                model_name, language, num_layers_for_this_model
            )
            print(hyperparameters)
