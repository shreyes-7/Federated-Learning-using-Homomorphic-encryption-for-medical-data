"""Model utilities for the federated learning demo."""

from __future__ import annotations

import numpy as np


class LogisticRegressionModel:
    """Simple NumPy logistic regression model with manual SGD updates."""

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def clone(self) -> "LogisticRegressionModel":
        clone = LogisticRegressionModel(self.n_features)
        clone.weights = self.weights.copy()
        clone.bias = float(self.bias)
        return clone

    def get_param_vector(self) -> np.ndarray:
        """Return flattened parameter vector: [weights..., bias]."""
        return np.concatenate([self.weights, np.array([self.bias], dtype=np.float64)])

    def set_param_vector(self, params: np.ndarray) -> None:
        """Load flattened parameters into model state."""
        self.weights = params[: self.n_features].astype(np.float64, copy=True)
        self.bias = float(params[self.n_features])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        logits = x @ self.weights + self.bias
        logits = np.clip(logits, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.int32)

    def train_local(self, x: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> None:
        """Train model parameters in-place using batch gradient descent."""
        n_samples = x.shape[0]
        for _ in range(epochs):
            probs = self.predict_proba(x)
            error = probs - y
            grad_w = (x.T @ error) / n_samples
            grad_b = float(np.mean(error))
            self.weights -= lr * grad_w
            self.bias -= lr * grad_b

    def evaluate_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(x)
        return float(np.mean(preds == y))
