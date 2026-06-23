import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from src.config import get_settings


class SupervisedClassifier:
    """Train Logistic Regression + Decision Tree on analyst-labeled anomalies."""

    def __init__(self) -> None:
        settings = get_settings()
        self.logistic = LogisticRegression(max_iter=1000, random_state=settings.random_state)
        self.tree = DecisionTreeClassifier(max_depth=5, random_state=settings.random_state)
        self.model_dir = settings.project_root / settings.model_path
        self.active_model = self.logistic

    def train(self, features: np.ndarray, labels: np.ndarray) -> dict:
        if len(np.unique(labels)) < 2 or len(labels) < 10:
            return {"trained": False, "reason": "insufficient labeled data"}

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.25, random_state=get_settings().random_state, stratify=labels
        )
        self.logistic.fit(x_train, y_train)
        self.tree.fit(x_train, y_train)

        log_pred = self.logistic.predict(x_test)
        tree_pred = self.tree.predict(x_test)
        log_precision = precision_score(y_test, log_pred, zero_division=0)
        tree_precision = precision_score(y_test, tree_pred, zero_division=0)

        self.active_model = self.logistic if log_precision >= tree_precision else self.tree
        active_pred = self.active_model.predict(x_test)
        precision = precision_score(y_test, active_pred, zero_division=0)

        self._save()
        return {
            "trained": True,
            "logistic_precision": round(log_precision, 4),
            "tree_precision": round(tree_precision, 4),
            "selected_model": "logistic_regression" if self.active_model is self.logistic else "decision_tree",
            "validation_precision": round(precision, 4),
            "report": classification_report(y_test, active_pred, zero_division=0),
        }

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if hasattr(self.active_model, "predict_proba"):
            return self.active_model.predict_proba(features)[:, 1]
        return self.active_model.predict(features).astype(float)

    def _save(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.logistic, self.model_dir / "logistic_regression.joblib")
        joblib.dump(self.tree, self.model_dir / "decision_tree.joblib")
        joblib.dump(type(self.active_model).__name__, self.model_dir / "active_model.txt")

    def load(self) -> None:
        from sklearn.tree import DecisionTreeClassifier as DTC

        self.logistic = joblib.load(self.model_dir / "logistic_regression.joblib")
        self.tree = joblib.load(self.model_dir / "decision_tree.joblib")
        active = (self.model_dir / "active_model.txt").read_text().strip()
        self.active_model = self.logistic if "Logistic" in active else self.tree
