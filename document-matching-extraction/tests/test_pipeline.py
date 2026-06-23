import pytest

from src.database import init_db, seed_from_csv
from src.pipeline.orchestrator import DocumentPipeline


@pytest.fixture(scope="module", autouse=True)
def setup():
    init_db()
    seed_from_csv()


def test_pipeline_invoice():
    from src.config import get_settings

    pipeline = DocumentPipeline()
    doc = get_settings().project_root / "data/incoming_documents/invoice_acme_4421.txt"
    result = pipeline.process_file(doc)
    assert result.extracted.vendor is not None
    assert result.extracted.amount == 12500.0
    assert result.audit_id is not None
