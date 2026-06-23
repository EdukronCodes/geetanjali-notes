import logging
from datetime import datetime

import numpy as np
import pandas as pd

from src.config import get_settings
from src.database import AnomalyRun, SessionLocal, init_db, load_transactions_from_db, seed_from_csv
from src.features.feature_engineer import FeatureEngineer
from src.models.isolation_forest import AnomalyScorer
from src.models.supervised_classifier import SupervisedClassifier
from src.reporting.excel_reporter import ExcelReporter

logger = logging.getLogger(__name__)


class AnomalyPipeline:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.feature_engineer = FeatureEngineer()
        self.isolation_forest = AnomalyScorer()
        self.classifier = SupervisedClassifier()

    def run(self, run_month: str | None = None) -> dict:
        init_db()
        seed_from_csv()

        df = load_transactions_from_db()
        enriched, features = self.feature_engineer.transform(df)

        iso_scores, iso_flags = self.isolation_forest.fit_predict(features.values)
        enriched["isolation_score"] = iso_scores
        enriched["isolation_flag"] = iso_flags

        labeled = enriched[enriched["analyst_label"].notna()].copy()
        train_metrics = {"trained": False}
        if len(labeled) >= 10:
            labeled_features = features.loc[labeled.index].values
            train_metrics = self.classifier.train(labeled_features, labeled["analyst_label"].astype(int).values)
        else:
            logger.warning("Not enough labeled data for supervised classifier")

        if train_metrics.get("trained"):
            proba = self.classifier.predict_proba(features.values)
            enriched["supervised_score"] = proba
            enriched["supervised_flag"] = (proba >= self.settings.anomaly_score_threshold).astype(int)
        else:
            enriched["supervised_score"] = iso_scores
            enriched["supervised_flag"] = iso_flags

        enriched["risk_score"] = (
            0.4 * enriched["isolation_score"]
            + 0.6 * enriched["supervised_score"]
        ).round(4)
        enriched["final_flag"] = (
            (enriched["risk_score"] >= self.settings.anomaly_score_threshold)
            | (enriched["duplicate_posting_flag"] == 1)
        ).astype(int)

        month = run_month or self.settings.run_month
        if month == "auto":
            month = datetime.now().strftime("%Y-%m")

        precision = train_metrics.get("validation_precision")
        reporter = ExcelReporter(self.settings.project_root / self.settings.output_path)
        output_file = reporter.write(enriched, month, precision)
        self.isolation_forest.save()

        session = SessionLocal()
        try:
            run = AnomalyRun(
                run_month=month,
                total_transactions=len(enriched),
                flagged_count=int(enriched["final_flag"].sum()),
                precision_score=precision,
                output_file=str(output_file),
            )
            session.add(run)
            session.commit()
        finally:
            session.close()

        return {
            "run_month": month,
            "total_transactions": len(enriched),
            "flagged_count": int(enriched["final_flag"].sum()),
            "validation_precision": precision,
            "training": train_metrics,
            "output_file": str(output_file),
            "top_risks": enriched.nlargest(5, "risk_score")[
                ["transaction_id", "vendor_id", "amount", "risk_score", "final_flag"]
            ].to_dict(orient="records"),
        }


def main() -> None:
    logging.basicConfig(level=get_settings().log_level)
    pipeline = AnomalyPipeline()
    result = pipeline.run()
    import json

    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
