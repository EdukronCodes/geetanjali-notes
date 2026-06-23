import pytest
from fastapi.testclient import TestClient

from src.database import init_db, seed_sample_data
from src.main import app


@pytest.fixture(scope="module", autouse=True)
def setup_db():
    init_db()
    seed_sample_data()


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_dispute():
    response = client.post(
        "/chat",
        json={
            "message": "I want to dispute transaction TXN-900001 on account ACC-10001 for $450 billing error in Dec 2025"
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "transaction_dispute"
    assert "reply" in data
    assert data["entities"]["account_number"] == "ACC-10001"


def test_chat_escalation():
    response = client.post(
        "/chat",
        json={"message": "This is fraud! I need a supervisor immediately for $80000 payment issue"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is True
    assert data["case_summary"] is not None
