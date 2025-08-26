from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    """Thin wrapper around IsolationForest for quick anomaly detection."""
    def __init__(self, n_estimators: int = 200, contamination: float = 0.05, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scores = -self.model.score_samples(X)              # higher = more anomalous
        labels = (self.model.predict(X) == -1).astype(int) # 1 = anomaly, 0 = normal
        return scores, labels
