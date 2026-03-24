import torch as t
import torch.nn as nn
from torch import Tensor
from activations_loader import ActivationDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
import torch.optim as optim
from pathlib import Path
from common_constants import PROBES_FOLDER


class LRProbe(t.nn.Module):
    def __init__(self, d_in, scaler_mean, scaler_scale, num_classes) -> None:
        super().__init__()
        self.net = t.nn.Sequential(
            t.nn.Linear(d_in, num_classes, bias=False), t.nn.Softmax(dim=-1)
        )
        self.register_buffer("scaler_mean", scaler_mean)
        self.register_buffer("scaler_scale", scaler_scale)

    def _normalize(self, x):
        if self.scaler_mean is not None and self.scaler_scale is not None:
            return (x - self.scaler_mean) / self.scaler_scale
        return x

    def forward(self, x) -> t.Tensor:
        return self.net(self._normalize(x))

    def pred(self, x) -> t.Tensor:
        logits: t.Tensor = self.forward(x)
        return t.argmax(logits, dim=-1)

    @staticmethod
    def create_from_data(dataset, C=0.1, device="cpu") -> "LRProbe":
        acts, labels = (dataset.activations, dataset.labels)
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()
        # print(f"y:\n{y}")

        num_classes: int = t.unique(labels).size(dim=0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lr_model = LogisticRegression(
            C=C,
            random_state=42,
            fit_intercept=False,
            max_iter=1000,
            # multi_class='auto'
        )
        lr_model.fit(X_scaled, y)

        scaler_mean: t.Tensor = t.tensor(scaler.mean_, dtype=t.float32)
        scaler_scale: t.Tensor = t.tensor(scaler.scale_, dtype=t.float32)

        model: LRProbe = LRProbe(
            acts.shape[-1], scaler_mean, scaler_scale, num_classes
        ).to(device)

        model.net[0].weight.data = t.tensor(lr_model.coef_, dtype=t.float32).to(device)

        return model


class MLPProbe(t.nn.Module):
    def __init__(self, d_in, hidden_size, num_classes) -> None:
        super().__init__()
        # Define the layers
        self.fc1 = nn.Linear(d_in, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def pred(self, x: Tensor) -> t.Tensor:
        logits: t.Tensor = self.forward(x)
        return t.argmax(logits, dim=-1)

    @staticmethod
    def create_from_data(
        dataset: ActivationDataset, batch_size: int, training_parameters, device
    ):
        # TODO Add scaler like in the logistic regression code
        acts, labels = dataset.activations, dataset.labels

        d_in: int = acts[0].size(dim=0)
        num_classes: int = t.unique(labels).size(dim=0)
        hidden_size: int = 128

        print(f"d_in: {d_in}")
        print(f"num_classes: {num_classes}")

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        model: MLPProbe = MLPProbe(d_in, hidden_size, num_classes)
        model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=training_parameters["learning_rate"],
            weight_decay=training_parameters["weight_decay"],
        )
        loss_fn = nn.CrossEntropyLoss()
        model.to(device)

        train_num_epochs(
            model,
            train_loader,
            optimizer,
            loss_fn,
            training_parameters["num_epochs"],
            device,
        )

        return model


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
    return f"{probe_type}_{language}_layer{layer_num}_{probing_task}.pt"


def save_probe(
    model: t.nn.Module,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
) -> str:
    """
    Save a probe model to a file.

    Args:
        model: The pytorch model to save
        model_name: Name of the model (e.g., 'olmo_model')
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')

    Returns:
        The path to the saved file
    """
    save_dir: Path = Path(PROBES_FOLDER) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
    filepath: Path = save_dir / filename

    t.save(model.state_dict(), filepath)
    print(f"Probe saved to {filepath}")

    return str(filepath)


def load_probe(
    model: t.nn.Module,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    device: str = "cpu",
) -> t.nn.Module:
    """
    Load a probe model from a file.

    Args:
        model: The pytorch model instance to load state into
        model_name: Name of the model (e.g., 'olmo_model')
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        device: Device to load model onto (default: 'cpu')

    Returns:
        The loaded model
    """
    filename: str = get_probe_filename(probe_type, language, layer_num, probing_task)
    filepath: Path = Path(PROBES_FOLDER) / model_name / filename

    state_dict = t.load(filepath, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)

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
