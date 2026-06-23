import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.config import get_settings


class AnomalyScorer:
    def __init__(self) -> None:
        settings = get_settings()
        self.model = IsolationForest(
            contamination=settings.isolation_forest_contamination,
            random_state=settings.random_state,
            n_estimators=200,
        )
        self.model_path = settings.project_root / settings.model_path / "isolation_forest.joblib"

    def fit_predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.model.fit(features)
        raw_scores = self.model.decision_function(features)
        # Normalize to 0-1 where higher = more anomalous
        normalized = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
        predictions = (self.model.predict(features) == -1).astype(int)
        return normalized, predictions

    def save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def load(self) -> None:
        self.model = joblib.load(self.model_path)
