from typing import Any

import torch as t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import ledoit_wolf
from pathlib import Path
import pickle
import json
import numpy as np

from utils import PROBES_FOLDER, PROBE_TYPE_SUBFOLDERS

# Ignore convergence warnings
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


mlp_training_parameters: dict[str, float | int] = {
    "learning_rate": 0.001,
    "batch_size": 256,
    "weight_decay": 0,
    "epochs": 10,
}


def _get_probe_subfolder(probe_type: str) -> str:
    if probe_type not in PROBE_TYPE_SUBFOLDERS:
        raise KeyError(
            f"Unknown probe type '{probe_type}'. Valid types: {list(PROBE_TYPE_SUBFOLDERS)}"
        )
    return PROBE_TYPE_SUBFOLDERS[probe_type]


class LRProbe:
    """Sklearn-based logistic regression probe"""

    def __init__(
        self,
        lr_model,
        scaler_mean,
        scaler_scale,
        metadata: dict[str, Any] | None = None,
        optimal_shrinkage: float | None = None,
    ) -> None:
        """
        Initialise LRProbe.

        Args:
            lr_model: Fitted sklearn LogisticRegression model
            scaler_mean: Mean values from StandardScaler
            scaler_scale: Scale values from StandardScaler
            optimal_shrinkage: Ledoit-Wolf shrinkage coefficient computed at training time.
        """
        self.lr_model: LogisticRegression = lr_model
        self.scaler_mean: float = scaler_mean
        self.scaler_scale: float = scaler_scale
        self.metadata: dict[str, Any] | None = metadata
        self.optimal_shrinkage: float | None = optimal_shrinkage

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
            warm_start=True,  # This lets me retrain the model on another language without starting from scratch
        )
        lr_model.fit(X_scaled, y)

        metadata: dict[str, Any] = {
            "language": dataset.language,
            "split": dataset.split,
            "layer_num": dataset.layer_num,
            "probing_task": dataset.probing_task,
            "model_name": dataset.model_name,
        }

        _, optimal_shrinkage = ledoit_wolf(X)
        optimal_shrinkage = float(optimal_shrinkage)

        return LRProbe(
            lr_model, scaler.mean_, scaler.scale_, metadata, optimal_shrinkage
        )

    def refit(self, new_dataset, iterations) -> None:
        """
        Continue training the existing model on new data.
        """
        acts, labels = (new_dataset.activations, new_dataset.labels)
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        # Use the existing scaler to maintain feature consistency
        X_scaled = (X - self.scaler_mean) / self.scaler_scale

        # Update max_iter for this specific run
        self.lr_model.max_iter = iterations

        self.lr_model.fit(X_scaled, y)

    def get_vector(self, per_class: bool = False) -> np.ndarray:
        coef = self.lr_model.coef_  # shape (3, n)
        intercept = self.lr_model.intercept_  # shape (3,)

        if per_class:
            # Return shape (3, n+1) - concatenate coef and intercept for each class
            return np.concatenate([coef, intercept.reshape(-1, 1)], axis=1)
        else:
            # Flatten and add dummy dimension for consistency
            flattened_coef = coef.flatten()
            flattened_intercept = intercept.flatten()
            flattened = np.concatenate([flattened_coef, flattened_intercept])
            return flattened.reshape(1, -1)  # shape (1, 3n+3)

    def calculate_cosine_similarity(
        self, second_lr_probe: Any, per_class: bool = False
    ) -> dict[int, float]:
        """
        Calculate cosine similarity between this probe and another.

        Args:
            second_lr_probe: The other LRProbe to compare with
            per_class: If True, return similarity for each class separately.
                      If False, return similarity for the flattened vectors as class 0.

        Returns:
            Dictionary mapping class index to cosine similarity value
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)  # shape (3, n+1)
            vector_2 = second_lr_probe.get_vector(per_class=True)  # shape (3, n+1)
            similarities = {}
            for i in range(vector_1.shape[0]):
                sim = cosine_similarity(vector_1[i : i + 1], vector_2[i : i + 1])[0, 0]
                similarities[int(self.lr_model.classes_[i])] = sim
            return similarities
        else:
            vector_1 = self.get_vector(per_class=False)  # shape (1, 3n+3)
            vector_2 = second_lr_probe.get_vector(per_class=False)  # shape (1, 3n+3)
            sim = cosine_similarity(vector_1, vector_2)[0, 0]
            return {0: sim}

    def calculate_maha_cos_sim(
        self,
        second_lr_probe: Any,
        per_class: bool = False,
        shrinkage: float | None = None,
    ) -> dict[int, float]:
        """
        Calculate Mahalanobis cosine similarity between this probe and another.

        Uses a diagonal precision matrix whose diagonal entries are
        1 / (scale_A * scale_B) per feature dimension (geometric-mean variance
        of the two probes' scalers). Intercept dimensions are left unscaled.

        Args:
            second_lr_probe: The other LRProbe to compare with
            per_class: If True, return similarity for each class separately.
                      If False, return similarity for the flattened vectors as class 0.
            shrinkage: Ledoit-Wolf shrinkage coefficient in [0, 1]. When None (default),
                      uses the average of both probes' optimal_shrinkage values computed
                      at training time (falls back to 0.0 if unavailable). Pass an explicit
                      float to override.

        Returns:
            Dictionary mapping class index to Mahalanobis cosine similarity value
        """
        if shrinkage is None:
            if (
                self.optimal_shrinkage is not None
                and second_lr_probe.optimal_shrinkage is not None
            ):
                shrinkage = (
                    self.optimal_shrinkage + float(second_lr_probe.optimal_shrinkage)
                ) / 2.0
            else:
                shrinkage = 0.0
        shrinkage = float(shrinkage)

        sigma = self.scaler_scale * second_lr_probe.scaler_scale
        if shrinkage > 0.0:
            mu = np.mean(sigma)
            sigma = (1.0 - shrinkage) * sigma + shrinkage * mu
        precision = 1.0 / np.sqrt(sigma)

        if per_class:
            vector_1 = self.get_vector(
                per_class=True
            )  # shape (n_classes, n_features+1)
            vector_2 = second_lr_probe.get_vector(per_class=True)
            # Weight features by precision; leave intercept dimension at 1.0
            per_class_precision = np.concatenate([precision, [1.0]])
            similarities = {}
            for i in range(vector_1.shape[0]):
                u = vector_1[i] * per_class_precision
                v = vector_2[i] * per_class_precision
                sim = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
                similarities[int(self.lr_model.classes_[i])] = sim
            return similarities
        else:
            vector_1 = self.get_vector(
                per_class=False
            )  # shape (1, n_classes*n_features + n_classes)
            vector_2 = second_lr_probe.get_vector(per_class=False)
            # Layout: [class0_feats..., class1_feats..., ..., intercept0, intercept1, ...]
            n_model_classes = self.lr_model.coef_.shape[0]
            flat_precision = np.concatenate(
                [np.tile(precision, n_model_classes), np.ones(n_model_classes)]
            )
            u = vector_1[0] * flat_precision
            v = vector_2[0] * flat_precision
            sim = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
            return {0: sim}

    def calculate_l2_dist(
        self, second_lr_probe: "LRProbe", per_class: bool = False
    ) -> dict[int, float]:
        """
        Calculate L2 dist (Euclidean distance) between this probe and another.

        Args:
            second_lr_probe: The other LRProbe to compare with
            per_class: If True, return L2 dist for each class separately.
                      If False, return L2 dist for the flattened vectors as class 0.

        Returns:
            Dictionary mapping class index to L2 dist value
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)  # shape (3, n+1)
            vector_2 = second_lr_probe.get_vector(per_class=True)  # shape (3, n+1)
            l2_dists = {}
            for i in range(vector_1.shape[0]):
                dist = np.linalg.norm(vector_1[i] - vector_2[i])
                l2_dists[int(self.lr_model.classes_[i])] = dist
            return l2_dists
        else:
            vector_1 = self.get_vector(per_class=False)  # shape (1, 3n+3)
            vector_2 = second_lr_probe.get_vector(per_class=False)  # shape (1, 3n+3)
            dist = np.linalg.norm(vector_1 - vector_2)
            return {0: dist}  # type: ignore

    def __str__(self) -> str:
        try:
            return f"Probe {', '.join(self.metadata.values())}"
        except KeyError or ValueError or AttributeError:
            return "Probe (Metadata missing)"


class MMProbe:
    """
    Mass-Mean probe for three-class NLI.

    Trains three binary mass-mean classifiers:
      - Classifier 0: entailment (0) vs neutral (1)
      - Classifier 1: neutral (1) vs contradiction (2)
      - Classifier 2: entailment (0) vs contradiction (2)

    Each classifier computes the difference of class means as its direction vector.
    Final prediction uses confidence-weighted voting across the three classifiers.
    """

    # (positive_class, negative_class) for each binary classifier
    CLASSIFIER_PAIRS: list[tuple[int, int]] = [(0, 1), (1, 2), (0, 2)]

    def __init__(
        self,
        directions: list[np.ndarray],
        thresholds: list[float],
        feature_std: np.ndarray,
        metadata: dict[str, Any] | None = None,
        optimal_shrinkage: float | None = None,
    ) -> None:
        """
        Args:
            directions: List of 3 unit-norm direction vectors, one per binary classifier.
            thresholds: List of 3 decision thresholds (midpoint of class means projected onto direction).
            feature_std: Per-feature standard deviation of training data (used for Mahalanobis similarity).
            metadata: Optional dict with training metadata.
            optimal_shrinkage: Ledoit-Wolf shrinkage coefficient from training data.
        """
        self.directions = directions
        self.thresholds = thresholds
        self.feature_std = feature_std
        self.metadata = metadata
        self.optimal_shrinkage = optimal_shrinkage

    def pred(self, x) -> np.ndarray:
        """
        Predict class labels using confidence-weighted voting across the three binary classifiers.

        Signed scores from each classifier contribute to per-class confidence:
          class 0 score = s0 + s2  (wins clf0 and clf2)
          class 1 score = -s0 + s1 (loses clf0, wins clf1)
          class 2 score = -s1 - s2 (loses clf1 and clf2)

        Returns:
            1D numpy array of predicted class labels (0, 1, or 2).
        """
        if isinstance(x, t.Tensor):
            x = x.float().cpu().numpy()
        x_arr = np.atleast_2d(x)

        s0 = x_arr @ self.directions[0] - self.thresholds[0]  # ent vs neu
        s1 = x_arr @ self.directions[1] - self.thresholds[1]  # neu vs contra
        s2 = x_arr @ self.directions[2] - self.thresholds[2]  # ent vs contra

        class_scores = np.column_stack(
            [
                s0 + s2,  # class 0 (entailment)
                -s0 + s1,  # class 1 (neutral)
                -s1 - s2,  # class 2 (contradiction)
            ]
        )

        return np.argmax(class_scores, axis=1)

    @staticmethod
    def create_from_data(dataset) -> "MMProbe":
        """
        Create MMProbe from an activation dataset by computing mass-mean directions.

        Args:
            dataset: ActivationDataset with activations and labels.

        Returns:
            Fitted MMProbe instance.
        """
        acts, labels = dataset.activations, dataset.labels
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy().astype(int)

        directions: list[np.ndarray] = []
        thresholds: list[float] = []

        for pos_class, neg_class in MMProbe.CLASSIFIER_PAIRS:
            mask = (y == pos_class) | (y == neg_class)
            X_bin = X[mask]
            y_bin = y[mask]

            mean_pos = X_bin[y_bin == pos_class].mean(axis=0)
            mean_neg = X_bin[y_bin == neg_class].mean(axis=0)

            diff = mean_pos - mean_neg
            norm = np.linalg.norm(diff)
            direction = diff / norm if norm > 0 else diff
            threshold = float(0.5 * direction @ (mean_pos + mean_neg))

            directions.append(direction)
            thresholds.append(threshold)

        feature_std = X.std(axis=0)
        feature_std[feature_std == 0] = 1.0  # prevent division by zero in Mahalanobis

        _, optimal_shrinkage = ledoit_wolf(X)

        metadata: dict[str, Any] = {
            "language": dataset.language,
            "split": dataset.split,
            "layer_num": dataset.layer_num,
            "probing_task": dataset.probing_task,
            "model_name": dataset.model_name,
        }

        return MMProbe(
            directions, thresholds, feature_std, metadata, float(optimal_shrinkage)
        )

    def refit(self, new_dataset, iterations) -> None:
        raise NotImplementedError("refit is not implemented for MMProbe")

    def get_vector(self, per_class: bool = False) -> np.ndarray:
        """
        Return the probe weight vectors.

        Args:
            per_class: If True, return shape (3, n_features+1) — one row per binary classifier,
                       each row is [direction, threshold].
                       If False, return shape (1, 3*n_features+3) — all directions then all thresholds.
        """
        if per_class:
            return np.array(
                [np.append(d, th) for d, th in zip(self.directions, self.thresholds)]
            )  # shape (3, n_features+1)
        else:
            flattened = np.concatenate(
                list(self.directions) + [np.array(self.thresholds)]
            )
            return flattened.reshape(1, -1)  # shape (1, 3*n_features+3)

    def calculate_cosine_similarity(
        self, second_probe: "MMProbe", per_class: bool = False
    ) -> dict[int, float]:
        """
        Calculate cosine similarity between this probe and another MMProbe.

        Args:
            second_probe: The other MMProbe to compare with.
            per_class: If True, return similarity for each binary classifier (keys 0, 1, 2).
                       If False, return similarity for the flattened vectors (key 0).

        Returns:
            Dictionary mapping classifier index (or 0) to cosine similarity value.
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)  # (3, n+1)
            vector_2 = second_probe.get_vector(per_class=True)
            return {
                i: float(
                    cosine_similarity(vector_1[i : i + 1], vector_2[i : i + 1])[0, 0]
                )
                for i in range(3)
            }
        else:
            vector_1 = self.get_vector(per_class=False)
            vector_2 = second_probe.get_vector(per_class=False)
            return {0: float(cosine_similarity(vector_1, vector_2)[0, 0])}

    def calculate_maha_cos_sim(
        self,
        second_probe: "MMProbe",
        per_class: bool = False,
        shrinkage: float | None = None,
    ) -> dict[int, float]:
        """
        Calculate Mahalanobis cosine similarity between this probe and another MMProbe.

        Uses a diagonal precision matrix based on the geometric mean of per-feature variances.
        Threshold dimensions are left unscaled.

        Args:
            second_probe: The other MMProbe to compare with.
            per_class: If True, return similarity per binary classifier; if False, flattened.
            shrinkage: Ledoit-Wolf shrinkage in [0, 1]. Defaults to average of both probes'
                       optimal_shrinkage values, or 0.0 if unavailable.

        Returns:
            Dictionary mapping classifier index (or 0) to Mahalanobis cosine similarity.
        """
        if shrinkage is None:
            if (
                self.optimal_shrinkage is not None
                and second_probe.optimal_shrinkage is not None
            ):
                shrinkage = (
                    self.optimal_shrinkage + second_probe.optimal_shrinkage
                ) / 2.0
            else:
                shrinkage = 0.0

        sigma = self.feature_std * second_probe.feature_std
        if shrinkage > 0.0:
            mu = np.mean(sigma)
            sigma = (1.0 - shrinkage) * sigma + shrinkage * mu
        precision = 1.0 / np.sqrt(sigma)

        if per_class:
            vector_1 = self.get_vector(per_class=True)  # (3, n+1)
            vector_2 = second_probe.get_vector(per_class=True)
            # Weight feature dims by precision; leave threshold dim at 1.0
            per_clf_precision = np.append(precision, 1.0)
            similarities = {}
            for i in range(3):
                u = vector_1[i] * per_clf_precision
                v = vector_2[i] * per_clf_precision
                sim = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
                similarities[i] = float(sim)
            return similarities
        else:
            vector_1 = self.get_vector(per_class=False)  # (1, 3*n+3)
            vector_2 = second_probe.get_vector(per_class=False)
            # Layout: [dir0..., dir1..., dir2..., thresh0, thresh1, thresh2]
            flat_precision = np.concatenate([np.tile(precision, 3), np.ones(3)])
            u = vector_1[0] * flat_precision
            v = vector_2[0] * flat_precision
            sim = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]
            return {0: float(sim)}

    def calculate_l2_dist(
        self, second_probe: "MMProbe", per_class: bool = False
    ) -> dict[int, float]:
        """
        Calculate L2 (Euclidean) distance between this probe and another MMProbe.

        Args:
            second_probe: The other MMProbe to compare with.
            per_class: If True, return distance per binary classifier; if False, flattened.

        Returns:
            Dictionary mapping classifier index (or 0) to L2 distance.
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)  # (3, n+1)
            vector_2 = second_probe.get_vector(per_class=True)
            return {
                i: float(np.linalg.norm(vector_1[i] - vector_2[i])) for i in range(3)
            }
        else:
            vector_1 = self.get_vector(per_class=False)
            vector_2 = second_probe.get_vector(per_class=False)
            return {0: float(np.linalg.norm(vector_1 - vector_2))}

    def __str__(self) -> str:
        if self.metadata is None:
            return "MMProbe (Metadata missing)"
        try:
            return f"MMProbe {', '.join(str(v) for v in self.metadata.values())}"
        except (KeyError, ValueError, AttributeError):
            return "MMProbe (Metadata missing)"


AnyProbe = LRProbe | MMProbe


def get_probe_filename(
    probe_type: str,
    language: str,
    layer_num: int,
    probing_task: str,
    extra_iters: int = 0,
) -> str:
    return f"{probe_type}_{language}_layer{layer_num}_{probing_task}{f'_{extra_iters}_extra_iters' if extra_iters else ''}.pkl"


def save_probe(
    model: AnyProbe,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    extra_iters: int = 0,
) -> str:
    """
    Save a probe model to a file.

    Args:
        model: The probe instance to save (LRProbe or MMProbe)
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe ('lr' or 'mm')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        The path to the saved file
    """
    subfolder = _get_probe_subfolder(probe_type)
    save_dir: Path = Path(PROBES_FOLDER) / model_name / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    filename: str = get_probe_filename(
        probe_type, language, layer_num, probing_task, extra_iters
    )
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
    extra_iters: int = 0,
) -> AnyProbe:
    """
    Load a probe model from a file.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe ('lr' or 'mm')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        The loaded probe instance
    """
    subfolder = _get_probe_subfolder(probe_type)
    filename: str = get_probe_filename(
        probe_type, language, layer_num, probing_task, extra_iters
    )
    filepath: Path = Path(PROBES_FOLDER) / model_name / subfolder / filename

    with open(filepath, "rb") as f:
        probe = pickle.load(f)

    return probe


def probe_exists(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    extra_iters: int = 0,
) -> bool:
    """
    Check if a probe file exists.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe ('lr' or 'mm')
        model_name: Name of the model (e.g., 'olmo_model')

    Returns:
        True if the probe file exists, False otherwise
    """
    subfolder = _get_probe_subfolder(probe_type)
    filename: str = get_probe_filename(
        probe_type, language, layer_num, probing_task, extra_iters
    )
    filepath: Path = Path(PROBES_FOLDER) / model_name / subfolder / filename

    return filepath.exists()


def get_probe(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    activation_dataset_train=None,
    force_probe_creation: bool = False,
    hyperparameters_file: str | None = None,
) -> AnyProbe:
    if (not force_probe_creation) and (
        probe_exists(language, layer_num, probing_task, probe_type, model_name)
    ):
        probe = load_probe(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
        )
    else:
        print("Creating probe")
        match probe_type:
            case "lr":
                if activation_dataset_train is None:
                    raise ValueError(
                        "activation_dataset_train must be specified in order to create a probe"
                    )

                # For default we turn off the hyperparameters. This is because if the probe at each layer or language has different hyperparameters,
                # it messes up with the cosine similarity comparisons due to the probes working in fundamentally different ways
                if hyperparameters_file is None:
                    hyperparams = {"C": 0.01, "fit_intercept": True}
                else:
                    hyperparams: dict = load_hyperparameters(
                        model_name, language, layer_num, hyperparameters_file
                    )
                C: float = hyperparams["C"]
                fit_intercept = hyperparams["fit_intercept"]
                probe: AnyProbe = LRProbe.create_from_data(
                    activation_dataset_train, C, fit_intercept
                )
            case "mm":
                if activation_dataset_train is None:
                    raise ValueError(
                        "activation_dataset_train must be specified in order to create a probe"
                    )
                probe = MMProbe.create_from_data(activation_dataset_train)
            case _:
                raise KeyError(
                    f"Probe '{probe_type}' does not exist. Valid types: {list(PROBE_TYPE_SUBFOLDERS)}"
                )
        # Save the probe
        save_probe(probe, language, layer_num, probing_task, probe_type, model_name)

    return probe


def load_hyperparameters(
    model_name: str,
    language: str,
    layer_num: int,
    hyperparameters_file: str,
) -> dict:
    """
    Load hyperparameters for a specific model, language, and layer.

    Args:
        model_name: Name of the model (e.g., 'olmo_model')
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        hyperparameters_file: Path to the hyperparameters JSON file

    Returns:
        Dictionary with hyperparameters (e.g., {'C': 0.1, 'fit_intercept': True})

    Raises:
        FileNotFoundError: If hyperparameters file doesn't exist
        KeyError: If the specified model/language/layer combination doesn't exist
    """
    filepath = Path(hyperparameters_file)

    if not filepath.exists():
        raise FileNotFoundError(f"Hyperparameters file not found at {filepath}. ")

    with open(filepath, "r") as f:
        all_hyperparameters = json.load(f)

    layer_key = str(layer_num)

    if model_name not in all_hyperparameters:
        raise KeyError(f"Model '{model_name}' not found in hyperparameters")
    if language not in all_hyperparameters[model_name]:
        raise KeyError(
            f"Language '{language}' not found for model '{model_name}' in hyperparameters"
        )
    if layer_key not in all_hyperparameters[model_name][language]:
        raise KeyError(
            f"Layer {layer_num} not found for {model_name}/{language} in hyperparameters"
        )

    return all_hyperparameters[model_name][language][layer_key]
