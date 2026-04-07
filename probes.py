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


# class LRProbe(t.nn.Module):
#     def __init__(self, d_in, scaler_mean, scaler_scale, num_classes) -> None:
#         super().__init__()
#         self.net = t.nn.Sequential(
#             t.nn.Linear(d_in, num_classes, bias=False), t.nn.Softmax(dim=-1)
#         )
#         self.register_buffer("scaler_mean", scaler_mean)
#         self.register_buffer("scaler_scale", scaler_scale)

#     def _normalize(self, x):
#         if self.scaler_mean is not None and self.scaler_scale is not None:
#             return (x - self.scaler_mean) / self.scaler_scale
#         return x

#     def forward(self, x) -> t.Tensor:
#         return self.net(self._normalize(x))

#     def pred(self, x) -> t.Tensor:
#         logits: t.Tensor = self.forward(x)
#         return t.argmax(logits, dim=-1)

#     @staticmethod
#     def create_from_data(dataset, C=0.1, device="cpu") -> "LRProbe":
#         acts, labels = (dataset.activations, dataset.labels)
#         X = acts.cpu().float().numpy()
#         y = labels.cpu().float().numpy()
#         # print(f"y:\n{y}")

#         num_classes: int = t.unique(labels).size(dim=0)

#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)

#         lr_model = LogisticRegression(
#             C=C,
#             random_state=42,
#             fit_intercept=False,
#             max_iter=1000,
#             class_weight="balanced"
#         )
#         lr_model.fit(X_scaled, y)

#         scaler_mean: t.Tensor = t.tensor(scaler.mean_, dtype=t.float32)
#         scaler_scale: t.Tensor = t.tensor(scaler.scale_, dtype=t.float32)

#         model: LRProbe = LRProbe(
#             acts.shape[-1], scaler_mean, scaler_scale, num_classes
#         ).to(device)

#         model.net[0].weight.data = t.tensor(lr_model.coef_, dtype=t.float32).to(device)

#         return model


class LRProbe:
    """Sklearn-based logistic regression probe without PyTorch."""

    def __init__(self, lr_model, scaler_mean, scaler_scale) -> None:
        """
        Initialize LRProbe.

        Args:
            lr_model: Fitted sklearn LogisticRegression model
            scaler_mean: Mean values from StandardScaler
            scaler_scale: Scale values from StandardScaler
        """
        self.lr_model = lr_model
        self.scaler_mean = scaler_mean
        self.scaler_scale = scaler_scale

    def _normalize(self, x):
        """Normalize input using stored scaler parameters."""
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
        normalized = self._normalize(x)
        return self.lr_model.predict(normalized)

    @staticmethod
    def create_from_data(dataset, C=0.1) -> "LRProbe":
        """
        Create LRProbe from an activation dataset.

        Args:
            dataset: ActivationDataset with activations and labels
            C: Inverse of regularization strength for LogisticRegression
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
            fit_intercept=False,
            max_iter=1000,
            class_weight="balanced",
        )
        lr_model.fit(X_scaled, y)

        return LRProbe(lr_model, scaler.mean_, scaler.scale_)


# class MLPProbe(t.nn.Module):
#     def __init__(self, d_in, hidden_size, num_classes) -> None:
#         super().__init__()
#         # Define the layers
#         self.fc1 = nn.Linear(d_in, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

#     def pred(self, x: Tensor) -> t.Tensor:
#         logits: t.Tensor = self.forward(x)
#         return t.argmax(logits, dim=-1)

#     @staticmethod
#     def create_from_data(dataset: ActivationDataset, batch_size: int, device):
#         # TODO Add scaler like in the logistic regression code
#         acts, labels = dataset.activations, dataset.labels

#         d_in: int = acts[0].size(dim=0)
#         num_classes: int = t.unique(labels).size(dim=0)
#         hidden_size: int = 128

#         print(f"d_in: {d_in}")
#         print(f"num_classes: {num_classes}")

#         train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#         model: MLPProbe = MLPProbe(d_in, hidden_size, num_classes)
#         model.to(device)

#         optimizer = optim.Adam(
#             model.parameters(),
#             lr=mlp_training_parameters["learning_rate"],
#             weight_decay=mlp_training_parameters["weight_decay"],
#         )
#         loss_fn = nn.CrossEntropyLoss()
#         model.to(device)

#         train_num_epochs(
#             model,
#             train_loader,
#             optimizer,
#             loss_fn,
#             mlp_training_parameters["num_epochs"],
#             device,
#         )

#         return model


# Functions for training neural network probes
def train_one_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()

    total_loss = 0

    for acts, labels in train_loader:
        acts, labels = acts.to(device), labels.to(device)

        # Do a forward pass
        outputs = model(acts)
        loss = loss_fn(outputs, labels)

        # Do backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def train_num_epochs(model, train_loader, optimizer, loss_fn, num_epochs, device):
    for epoch in range(num_epochs):
        total_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def get_probe_filename(
    probe_type: str, language: str, layer_num: int, probing_task: str
) -> str:
    return f"{probe_type}_{language}_layer{layer_num}_{probing_task}.pkl"


# def save_probe(
#     model: t.nn.Module,
#     language: str,
#     layer_num: int,
#     probing_task: str,
#     probe_type: str,
#     model_name: str,
# ) -> str:
#     """
#     Save a probe model to a file.

#     Args:
#         model: The pytorch model to save
#         model_name: Name of the model (e.g., 'olmo_model')
#         language: Language code (e.g., 'en', 'es')
#         layer_num: Layer number
#         probing_task: Probing task name (e.g., 'standard')

#     Returns:
#         The path to the saved file
#     """
#     save_dir: Path = Path(PROBES_FOLDER) / model_name
#     save_dir.mkdir(parents=True, exist_ok=True)

#     filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
#     filepath: Path = save_dir / filename

#     t.save(model.state_dict(), filepath)
#     print(f"Probe saved to {filepath}")

#     return str(filepath)


# def load_probe(
#     model: t.nn.Module,
#     language: str,
#     layer_num: int,
#     probing_task: str,
#     probe_type: str,
#     model_name: str,
#     device: str = "cpu",
# ) -> LRProbe:
#     """
#     Load a probe model from a file.

#     Args:
#         model: The pytorch model instance to load state into
#         model_name: Name of the model (e.g., 'olmo_model')
#         language: Language code (e.g., 'en', 'es')
#         layer_num: Layer number
#         probing_task: Probing task name (e.g., 'standard')
#         device: Device to load model onto (default: 'cpu')

#     Returns:
#         The loaded model
#     """
#     filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
#     filepath: Path = Path(PROBES_FOLDER) / model_name / filename

#     state_dict = t.load(filepath, map_location=device, weights_only=True)
#     model.load_state_dict(state_dict)
#     model.to(device)

#     print(f"Probe loaded from {filepath}")

#     return model


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
        match probe_type:
            case "lr":
                probe = LRProbe.create_from_data(activation_dataset_train)
            # case "mlp":
            #     probe = MLPProbe.create_from_data(activation_dataset_train, 128, device)
            case _:
                raise KeyError(f"Probe {probe_type} does not exist")
        probe = load_probe(
            probe,
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            device="cpu",
        )
    else:
        # Create new probe
        print("Creating probe")
        match probe_type:
            case "lr":
                probe = LRProbe.create_from_data(activation_dataset_train)
            # case "mlp":
            #     probe = MLPProbe.create_from_data(activation_dataset_train, 128, device)
            case _:
                raise KeyError(f"Probe {probe_type} does not exist")
        # Save the probe
        save_probe(probe, language, layer_num, probing_task, probe_type, model_name)

    return probe
