import torch as t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from common_constants import PROBES_FOLDER
import pickle


mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}


class LRProbe:
    """Sklearn-based logistic regression probe without PyTorch."""

    def __init__(self, lr_model, scaler_mean, scaler_scale) -> None:
        """
        Initialise LRProbe.

        Args:
            lr_model: Fitted sklearn LogisticRegression model
            scaler_mean: Mean values from StandardScaler
            scaler_scale: Scale values from StandardScaler
        """
        self.lr_model = lr_model
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

    def _normalise(self, x):
        """normalise input using stored scaler parameters."""
        if isinstance(x, t.Tensor):
            x = x.float().cpu().numpy()
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def pred(self, x):
        """
        Get predicted class labels for input x.

        Args:
            x: Input data, can be numpy array or torch Tensor

        Returns:
            numpy array of predicted class labels
        """
        normalised = self._normalise(x)
        return self.lr_model.predict(normalised)

    @staticmethod
    def create_from_data(dataset, C, fit_intercept) -> "LRProbe":
        """
        Create LRProbe from an activation dataset.

        Args:
            dataset: ActivationDataset with activations and labels
            C: Inverse of regularisation strength for LogisticRegression
            device: Device parameter (kept for API compatibility)

        Returns:
            Fitted LRProbe instance
        """
        acts, labels = (dataset.activations, dataset.labels)
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(
            C=C,
            random_state=42,
            fit_intercept=fit_intercept,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",  # saga does not work well as a solver. It takes a very long time to fit and does not converge after 1000 iterations.
        )
        lr_model.fit(X_scaled, y)

        return LRProbe(lr_model, scaler.mean_, scaler.scale_)


def get_probe_filename(
    probe_type: str, language: str, layer_num: int, probing_task: str
) -> str:
    return f"{probe_type}_{language}_layer{layer_num}_{probing_task}.pkl"


def save_probe(
    model: LRProbe,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
) -> str:
    """
    Save an sklearn-based probe model to a file.

    Args:
        model: The LRProbe instance to save
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe (e.g., 'lr')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        The path to the saved file
    """
    save_dir: Path = Path(PROBES_FOLDER) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
    filepath: Path = save_dir / filename

    with open(filepath, "wb") as f:
        pickle.dump(model, f)

    print(f"Probe saved to {filepath}")

    return str(filepath)


def load_probe(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
) -> LRProbe:
    """
    Load an sklearn-based probe model from a file.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe (e.g., 'lr')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        The loaded LRProbe instance
    """
    filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
    filepath: Path = Path(PROBES_FOLDER) / model_name / filename

    with open(filepath, "rb") as f:
        model = pickle.load(f)

    print(f"Probe loaded from {filepath}")

    return model


def probe_exists(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
) -> bool:
    """
    Check if a probe file exists.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe (e.g., 'lr', 'mlp')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        True if the probe file exists, False otherwise
    """
    filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
    filepath: Path = Path(PROBES_FOLDER) / model_name / filename

    return filepath.exists()


def get_probe(
    language,
    layer_num,
    probing_task,
    probe_type,
    model_name,
    activation_dataset_train,
    force_probe_creation,
    device,
):
    if (not force_probe_creation) and (
        probe_exists(language, layer_num, probing_task, probe_type, model_name)
    ):
        print("Probe already exists. Loading from file...")
        probe = load_probe(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
        )
    else:
        # Create new probe
        print("Creating probe")
        match probe_type:
            case "lr":
                probe = LRProbe.create_from_data(activation_dataset_train)
            # MLP not currently implemented
            # case "mlp":
            #     probe = MLPProbe.create_from_data(activation_dataset_train, 128, device)
            case _:
                raise KeyError(f"Probe {probe_type} does not exist")
        # Save the probe
        save_probe(probe, language, layer_num, probing_task, probe_type, model_name)

    return probe
