from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = None


class EntityResult(BaseModel):
    account_number: str | None = None
    transaction_id: str | None = None
    date_range: str | None = None
    amount: float | None = None


class IntentResult(BaseModel):
    intent: Literal[
        "transaction_dispute",
        "billing_discrepancy",
        "reconciliation_status",
        "payment_delay",
        "general_inquiry",
        "escalation_request",
    ]
    confidence: float


class AccountSnapshot(BaseModel):
    account_id: str
    customer_name: str
    status: str
    recent_transactions: list[dict]


class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    entities: EntityResult
    escalated: bool
    case_summary: str | None = None
    complaint_id: int | None = None
    sources: list[str] = []


class HealthResponse(BaseModel):
    status: str
    azure_openai: bool
    database: str
    faiss_index: bool
