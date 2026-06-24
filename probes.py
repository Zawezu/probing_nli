from typing import Any

import torch as t
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.covariance import ledoit_wolf
from pathlib import Path
import os
import pickle
import json
import tempfile
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
    """Return the subdirectory name for a given probe type (e.g. 'lr' -> 'logistic_regression')."""
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
        zeroed_dims: np.ndarray | None = None,
    ) -> None:
        """
        Initialise LRProbe.

        Args:
            lr_model: Fitted sklearn LogisticRegression model
            scaler_mean: Mean values from StandardScaler
            scaler_scale: Scale values from StandardScaler
            optimal_shrinkage: Ledoit-Wolf shrinkage coefficient computed at training time.
            zeroed_dims: Indices of activation dimensions zeroed out during training.
        """
        self.lr_model: LogisticRegression = lr_model
        self.scaler_mean: float = scaler_mean
        self.scaler_scale: float = scaler_scale
        self.metadata: dict[str, Any] | None = metadata
        self.optimal_shrinkage: float | None = optimal_shrinkage
        self.zeroed_dims: np.ndarray | None = zeroed_dims

    def _normalise(self, x):
        """normalise input using stored scaler parameters, then zero out stored dims."""
        if isinstance(x, t.Tensor):
            x = x.float().cpu().numpy()
        if self.scaler_mean is not None and self.scaler_scale is not None:
            result = (x - self.scaler_mean) / self.scaler_scale
        else:
            result = np.asarray(x, dtype=float)
        if self.zeroed_dims is not None:
            result = result.copy()
            if result.ndim == 2:
                result[:, self.zeroed_dims] = 0.0
            else:
                result[self.zeroed_dims] = 0.0
        return result

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
    def create_from_data(
        dataset, C, fit_intercept, zeroed_out_activation_dims: int = 0
    ) -> "LRProbe":
        """
        Create LRProbe from an activation dataset.

        Args:
            dataset: ActivationDataset with activations and labels.
            C: Inverse of regularisation strength for LogisticRegression.
            fit_intercept: Whether to fit a bias term in the logistic regression.
            zeroed_out_activation_dims: Number of highest-average-magnitude dims to zero out before training.

        Returns:
            Fitted LRProbe instance.
        """
        acts, labels = (dataset.activations, dataset.labels)
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        zeroed_dims: np.ndarray | None = None
        if zeroed_out_activation_dims > 0:
            avg_magnitudes = np.abs(X).mean(axis=0)
            zeroed_dims = np.argsort(avg_magnitudes)[-zeroed_out_activation_dims:]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if zeroed_dims is not None:
            X_scaled = X_scaled.copy()
            X_scaled[:, zeroed_dims] = 0.0

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
            lr_model,
            scaler.mean_,
            scaler.scale_,
            metadata,
            optimal_shrinkage,
            zeroed_dims,
        )

    def refit(self, new_dataset, iterations) -> None:
        """Continue training the existing model on new data using warm-start.

        The existing scaler parameters are reused so the feature space stays
        consistent between the initial fit and refitting.

        Args:
            new_dataset: ActivationDataset to retrain on.
            iterations: Maximum number of solver iterations for this refit step.
        """
        acts, labels = (new_dataset.activations, new_dataset.labels)
        X = acts.cpu().float().numpy()
        y = labels.cpu().float().numpy()

        # Use the existing scaler to maintain feature consistency
        X_scaled = (X - self.scaler_mean) / self.scaler_scale

        if self.zeroed_dims is not None:
            X_scaled = X_scaled.copy()
            X_scaled[:, self.zeroed_dims] = 0.0

        # Update max_iter for this specific run
        self.lr_model.max_iter = iterations

        self.lr_model.fit(X_scaled, y)

    def get_vector(self, per_class: bool = False) -> np.ndarray:
        """Return the probe's weight vectors, optionally concatenated with intercepts.

        Args:
            per_class: If True, return shape (n_classes, n_features+1) — one row per
                class with coefficients and intercept appended.
                If False, return shape (1, n_classes*n_features + n_classes) — all
                coefficients flattened followed by all intercepts.
        """
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
        self,
        second_lr_probe: "LRProbe",
        per_class: bool = False,
        normalise: bool = True,
    ) -> dict[int, float]:
        """Calculate L2 (Euclidean) distance between this probe and another LRProbe.

        Args:
            second_lr_probe: The other LRProbe to compare with.
            per_class: If True, return distance per class; if False, use flattened vectors.
            normalise: If True, unit-normalise both vectors before computing the distance.

        Returns:
            Dictionary mapping class index (or 0 for flattened) to L2 distance.
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)
            vector_2 = second_lr_probe.get_vector(per_class=True)
            l2_dists = {}
            for i in range(vector_1.shape[0]):
                v1, v2 = vector_1[i], vector_2[i]
                if normalise:
                    v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                    v2 = v2 / (np.linalg.norm(v2) + 1e-10)
                l2_dists[int(self.lr_model.classes_[i])] = np.linalg.norm(v1 - v2)
            return l2_dists
        else:
            vector_1 = self.get_vector(per_class=False)
            vector_2 = second_lr_probe.get_vector(per_class=False)
            v1, v2 = vector_1[0], vector_2[0]
            if normalise:
                v1 = v1 / (np.linalg.norm(v1) + 1e-10)
                v2 = v2 / (np.linalg.norm(v2) + 1e-10)
            return {0: float(np.linalg.norm(v1 - v2))}

    def __str__(self) -> str:
        try:
            return f"Probe {', '.join(self.metadata.values())}"
        except (KeyError, ValueError, AttributeError):
            return "Probe (Metadata missing)"


class MMProbe:
    """
    Mass-mean probe generalised to multiple classes via linear discriminant
    analysis (LDA).

    This is the natural multi-class extension of the binary mass-mean probe of
    Marks & Tegmark (2023). Instead of training several one-vs-one binary
    classifiers and combining their (scale-incomparable) signed margins, we fit a
    single LDA model sharing one pooled within-class covariance Σ:

        δ_k(x) = xᵀ Σ⁻¹ μ_k − ½ μ_kᵀ Σ⁻¹ μ_k        (+ log π_k)

    where μ_k is the mean of class k. The prediction is argmax_k δ_k(x). The Σ⁻¹
    factor is exactly the "mass-mean correction" of the IID mass-mean probe; here
    it is shared across all classes so that the K discriminant scores live in the
    same space and need no per-classifier re-normalisation.

    We use equal priors (the log π_k term is then constant and dropped). This
    matches both the original mass-mean probe — whose binary decision threshold
    sits at the midpoint of the two class means — and the
    `class_weight="balanced"` setting of `LRProbe`.

    The stored per-class weight vectors (`directions`) and `biases` are gauge-fixed
    to sum to zero across classes. argmax is invariant to adding a common vector to
    every class's score, so this centring leaves predictions unchanged while giving
    a canonical representation for cosine-similarity comparisons (mirroring the gauge
    freedom of multinomial logistic-regression logits).
    """

    def __init__(
        self,
        directions: list[np.ndarray],
        biases: np.ndarray,
        classes: np.ndarray,
        metadata: dict[str, Any] | None = None,
        zeroed_dims: np.ndarray | None = None,
        cov_inv: np.ndarray | None = None,
        means: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            directions: List of K per-class LDA weight vectors w_k = Σ⁻¹ μ_k,
                gauge-centred to sum to zero across classes.
            biases: Array of K per-class biases b_k = −½ μ_kᵀ Σ⁻¹ μ_k,
                gauge-centred to sum to zero across classes.
            classes: Sorted array of the K class labels, aligned with `directions`.
            metadata: Optional dict with training metadata.
            optimal_shrinkage: Ledoit-Wolf shrinkage coefficient from training data.
            zeroed_dims: Indices of activation dimensions zeroed out during training.
            cov_inv: Shared precision matrix Σ⁻¹ (d×d) used for Mahalanobis similarity.
            means: Per-class means μ_k (K×d), kept for reference/diagnostics.
        """
        self.directions = directions
        self.biases = np.asarray(biases, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.metadata = metadata
        self.zeroed_dims: np.ndarray | None = zeroed_dims
        self.cov_inv: np.ndarray | None = cov_inv
        self.means: np.ndarray | None = means

    def pred(self, x) -> np.ndarray:
        """
        Predict class labels via multi-class LDA: argmax_k δ_k(x), where the
        per-class discriminant is δ_k(x) = xᵀ w_k + b_k with w_k = Σ⁻¹ μ_k and
        b_k = −½ μ_kᵀ Σ⁻¹ μ_k.

        Returns:
            1D numpy array of predicted class labels.
        """
        if isinstance(x, t.Tensor):
            x = x.float().cpu().numpy()
        x_arr = np.atleast_2d(x).astype(np.float64)
        if self.zeroed_dims is not None:
            x_arr = x_arr.copy()
            x_arr[:, self.zeroed_dims] = 0.0

        W = np.asarray(self.directions, dtype=np.float64)  # (K, d)
        scores = x_arr @ W.T + self.biases  # (m, K)
        pred_idx = np.argmax(scores, axis=1)
        return self.classes_[pred_idx]

    @staticmethod
    def create_from_data(dataset, zeroed_out_activation_dims: int = 0) -> "MMProbe":
        """
        Create MMProbe from an activation dataset by computing mass-mean directions.

        Args:
            dataset: ActivationDataset with activations and labels.
            zeroed_out_activation_dims: Number of highest-average-magnitude dims to zero out before training.

        Returns:
            Fitted MMProbe instance.
        """
        acts, labels = dataset.activations, dataset.labels
        # float64 throughout: the d×d inverse and the quadratic bias terms are
        # sensitive to precision, especially for large-magnitude late-layer activations.
        X = acts.cpu().float().numpy().astype(np.float64)
        y = labels.cpu().float().numpy().astype(int)

        zeroed_dims: np.ndarray | None = None
        if zeroed_out_activation_dims > 0:
            avg_magnitudes = np.abs(X).mean(axis=0)
            zeroed_dims = np.argsort(avg_magnitudes)[-zeroed_out_activation_dims:]
            X = X.copy()
            X[:, zeroed_dims] = 0.0

        classes = np.unique(y)  # sorted class labels, e.g. [0, 1, 2]
        K = len(classes)

        # Per-class means μ_k and the pooled within-class centred data (each point
        # minus its own class mean). The covariance of the latter is the shared Σ.
        means = np.zeros((K, X.shape[1]), dtype=np.float64)
        X_centred = np.empty_like(X)
        for i, k in enumerate(classes):
            mask = y == k
            mu_k = X[mask].mean(axis=0)
            means[i] = mu_k
            X_centred[mask] = X[mask] - mu_k

        # Shared pooled within-class covariance Σ and its inverse (the mass-mean
        # correction). Ledoit-Wolf shrinkage keeps Σ well-conditioned; pinv is a
        # safety net in case it is still singular.
        cov, _ = ledoit_wolf(X_centred)
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)

        # LDA discriminant δ_k(x) = xᵀ(Σ⁻¹ μ_k) − ½ μ_kᵀ Σ⁻¹ μ_k (equal priors).
        # Σ⁻¹ is symmetric, so row k of (means @ Σ⁻¹) is (Σ⁻¹ μ_k)ᵀ.
        coefs = means @ cov_inv  # (K, d), w_k = Σ⁻¹ μ_k
        biases = -0.5 * np.einsum("kd,kd->k", coefs, means)  # −½ μ_kᵀ Σ⁻¹ μ_k

        # Gauge-fix: centre weights and biases across classes (argmax-invariant).
        # Must happen after `biases` is computed from the raw coefs above.
        coefs = coefs - coefs.mean(axis=0, keepdims=True)
        biases = biases - biases.mean()

        directions = [np.array(coefs[i]) for i in range(K)]

        metadata: dict[str, Any] = {
            "language": dataset.language,
            "split": dataset.split,
            "layer_num": dataset.layer_num,
            "probing_task": dataset.probing_task,
            "model_name": dataset.model_name,
        }

        return MMProbe(
            directions,
            biases,
            classes,
            metadata,
            zeroed_dims,
            cov_inv,
            means,
        )

    def refit(self, new_dataset, iterations) -> None:
        raise NotImplementedError("refit is not implemented for MMProbe")

    def get_vector(self, per_class: bool = False) -> np.ndarray:
        """
        Return the probe direction vectors.

        Args:
            per_class: If True, return shape (K, n_features) — one LDA weight vector per class.
                       If False, return shape (1, K*n_features) — all weight vectors flattened.
        """
        if per_class:
            return np.array(self.directions)  # shape (K, n_features)
        else:
            flattened = np.concatenate(self.directions)
            return flattened.reshape(1, -1)  # shape (1, K*n_features)

    def calculate_cosine_similarity(
        self, second_probe: "MMProbe", per_class: bool = False
    ) -> dict[int, float]:
        """
        Calculate cosine similarity between this probe and another MMProbe.

        Args:
            second_probe: The other MMProbe to compare with.
            per_class: If True, return similarity for each class (keys 0..K-1).
                       If False, return similarity for the flattened vectors (key 0).

        Returns:
            Dictionary mapping class index (or 0) to cosine similarity value.
        """
        if per_class:
            vector_1 = self.get_vector(per_class=True)  # (K, n_features)
            vector_2 = second_probe.get_vector(per_class=True)

            return {
                i: float(
                    cosine_similarity(vector_1[i : i + 1], vector_2[i : i + 1])[0, 0]
                )
                for i in range(vector_1.shape[0])
            }
        else:
            vector_1 = self.get_vector(per_class=False)
            vector_2 = second_probe.get_vector(per_class=False)
            return {0: float(cosine_similarity(vector_1, vector_2)[0, 0])}

    def calculate_maha_cos_sim(
        self,
        second_probe: "MMProbe",
        per_class: bool = False,
    ) -> dict[int, float]:
        """
        Calculate Mahalanobis cosine similarity between this probe and another MMProbe.

        Uses the shared LDA precision matrix M = (Σ⁻¹_self + Σ⁻¹_other) / 2, averaged
        across the two probes. For per_class=False the flattened-vector similarity uses
        a block-diagonal precision (the same shared M repeated, one block per class),
        which is equivalent to summing the per-class numerators and norms.

        Args:
            second_probe: The other MMProbe to compare with.
            per_class: If True, return similarity per class; if False, flattened.

        Returns:
            Dictionary mapping class index (or 0) to Mahalanobis cosine similarity.
        """
        # Shared precision averaged across both probes.
        M = (
            np.asarray(self.cov_inv, dtype=np.float64)
            + np.asarray(second_probe.cov_inv, dtype=np.float64)
        ) / 2.0
        if per_class:
            similarities = {}
            for i in range(len(self.directions)):
                u = self.directions[i]
                v = second_probe.directions[i]
                Mu = M @ u
                Mv = M @ v
                num = float(u @ Mv)
                denom = float(np.sqrt((u @ Mu) * (v @ Mv)))
                similarities[i] = num / denom if denom > 0 else 0.0
            return similarities
        else:
            num = 0.0
            u_norm_sq = 0.0
            v_norm_sq = 0.0
            for i in range(len(self.directions)):
                u = self.directions[i]
                v = second_probe.directions[i]
                Mu = M @ u
                Mv = M @ v
                num += float(u @ Mv)
                u_norm_sq += float(u @ Mu)
                v_norm_sq += float(v @ Mv)
            denom = np.sqrt(u_norm_sq * v_norm_sq)
            return {0: float(num / denom) if denom > 0 else 0.0}

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
    zeroed_out_activation_dims: int = 0,
    force_original_labels: bool = False,
) -> str:
    """Construct the filename for a saved probe, encoding all training options.

    Optional suffixes are appended in order: extra_iters, zeroed_out_activation_dims,
    and (for Japanese probes) orig_labels when force_original_labels is True.
    """
    name = f"{probe_type}_{language}_layer{layer_num}_{probing_task}"
    if extra_iters:
        name += f"_{extra_iters}_extra_iters"
    if zeroed_out_activation_dims:
        name += f"_{zeroed_out_activation_dims}_zeroed_act_dims"
    if force_original_labels and "jp" in language:
        name += "_orig_labels"
    return name + ".pkl"


def apply_zeroed_weight_dims(probe: "AnyProbe", zeroed_out_weight_dims: int) -> None:
    """Zero out the top-N highest-magnitude weight dimensions in a probe (per class/classifier)."""
    if zeroed_out_weight_dims <= 0:
        return
    if isinstance(probe, LRProbe):
        for i in range(probe.lr_model.coef_.shape[0]):
            top_dims = np.argsort(np.abs(probe.lr_model.coef_[i]))[
                -zeroed_out_weight_dims:
            ]
            probe.lr_model.coef_[i, top_dims] = 0.0
    elif isinstance(probe, MMProbe):
        for i in range(len(probe.directions)):
            top_dims = np.argsort(np.abs(probe.directions[i]))[-zeroed_out_weight_dims:]
            probe.directions[i][top_dims] = 0.0


def save_probe(
    model: AnyProbe,
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    extra_iters: int = 0,
    zeroed_out_activation_dims: int = 0,
    force_original_labels: bool = False,
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
        zeroed_out_activation_dims: Number of activation dims zeroed during training (affects filename).

    Returns:
        The path to the saved file
    """
    subfolder = _get_probe_subfolder(probe_type)
    save_dir: Path = Path(PROBES_FOLDER) / model_name / subfolder
    save_dir.mkdir(parents=True, exist_ok=True)

    filename: str = get_probe_filename(
        probe_type,
        language,
        layer_num,
        probing_task,
        extra_iters,
        zeroed_out_activation_dims,
        force_original_labels,
    )
    filepath: Path = save_dir / filename

    # Write atomically (temp file + os.replace) so that concurrent saves of the same
    # probe never produce a torn/half-written file. When experiments run in parallel,
    # several workers may save the same base probe (e.g. the language_a probe shared by
    # multiple language pairs); they write identical content, and the atomic rename
    # guarantees readers only ever see a complete pickle.
    fd, tmp_path = tempfile.mkstemp(dir=save_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(model, f)
        os.replace(tmp_path, filepath)
    except BaseException:
        # Clean up the temp file if anything went wrong before the rename.
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    print(f"Probe saved to {filepath}")

    return str(filepath)


def load_probe(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    extra_iters: int = 0,
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> AnyProbe:
    """
    Load a probe model from a file.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe ('lr' or 'mm')
        model_name: Name of the model (e.g., 'olmo_model')
        zeroed_out_activation_dims: Must match the value used when the probe was saved.
        zeroed_out_weight_dims: If > 0, zero out this many highest-magnitude weight dims per class after loading.
        force_original_labels: If True and language contains 'jp', loads the probe trained with original (non-Japanese) labels.

    Returns:
        The loaded probe instance
    """
    subfolder = _get_probe_subfolder(probe_type)
    filename: str = get_probe_filename(
        probe_type,
        language,
        layer_num,
        probing_task,
        extra_iters,
        zeroed_out_activation_dims,
        force_original_labels,
    )
    filepath: Path = Path(PROBES_FOLDER) / model_name / subfolder / filename

    with open(filepath, "rb") as f:
        probe = pickle.load(f)

    apply_zeroed_weight_dims(probe, zeroed_out_weight_dims)

    return probe


def probe_exists(
    language: str,
    layer_num: int,
    probing_task: str,
    probe_type: str,
    model_name: str,
    extra_iters: int = 0,
    zeroed_out_activation_dims: int = 0,
    force_original_labels: bool = False,
) -> bool:
    """
    Check if a probe file exists.

    Args:
        language: Language code (e.g., 'en', 'es')
        layer_num: Layer number
        probing_task: Probing task name (e.g., 'standard')
        probe_type: Type of probe ('lr' or 'mm')
        model_name: Name of the model (e.g., 'olmo_model')
        zeroed_out_activation_dims: Must match the value used when the probe was saved.
        force_original_labels: If True and language contains 'jp', checks for the probe trained with original labels.

    Returns:
        True if the probe file exists, False otherwise
    """
    subfolder = _get_probe_subfolder(probe_type)
    filename: str = get_probe_filename(
        probe_type,
        language,
        layer_num,
        probing_task,
        extra_iters,
        zeroed_out_activation_dims,
        force_original_labels,
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
    zeroed_out_activation_dims: int = 0,
    zeroed_out_weight_dims: int = 0,
    force_original_labels: bool = False,
) -> AnyProbe:
    """Load a probe from disk or create, save, and return a new one.

    If a saved probe matching all parameters exists and `force_probe_creation` is
    False, it is loaded directly. Otherwise a new probe is trained on
    `activation_dataset_train`, saved, and returned.

    For LR probes, hyperparameters are taken from `hyperparameters_file` when
    provided; otherwise defaults (C=0.01, fit_intercept=True) are used.

    `zeroed_out_weight_dims` is applied after loading or training and is not
    encoded in the filename, so it does not affect the cached probe on disk.
    """
    if (not force_probe_creation) and (
        probe_exists(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            force_original_labels=force_original_labels,
        )
    ):
        probe = load_probe(
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            zeroed_out_weight_dims=zeroed_out_weight_dims,
            force_original_labels=force_original_labels,
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
                    activation_dataset_train,
                    C,
                    fit_intercept,
                    zeroed_out_activation_dims,
                )
            case "mm":
                if activation_dataset_train is None:
                    raise ValueError(
                        "activation_dataset_train must be specified in order to create a probe"
                    )
                probe = MMProbe.create_from_data(
                    activation_dataset_train, zeroed_out_activation_dims
                )
            case _:
                raise KeyError(
                    f"Probe '{probe_type}' does not exist. Valid types: {list(PROBE_TYPE_SUBFOLDERS)}"
                )
        # Save the probe
        save_probe(
            probe,
            language,
            layer_num,
            probing_task,
            probe_type,
            model_name,
            zeroed_out_activation_dims=zeroed_out_activation_dims,
            force_original_labels=force_original_labels,
        )
        apply_zeroed_weight_dims(probe, zeroed_out_weight_dims)

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
