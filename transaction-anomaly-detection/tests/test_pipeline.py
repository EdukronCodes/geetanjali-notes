import pytest

from src.pipeline.anomaly_pipeline import AnomalyPipeline


def test_pipeline_runs():
    pipeline = AnomalyPipeline()
    result = pipeline.run(run_month="2025-02")
    assert result["total_transactions"] > 0
    assert "output_file" in result
    assert result["flagged_count"] >= 0


def test_precision_target():
    pipeline = AnomalyPipeline()
    result = pipeline.run(run_month="2025-test")
    precision = result.get("validation_precision")
    if precision is not None:
        assert precision >= 0.5
