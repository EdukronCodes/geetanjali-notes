import hashlib
import json
import re
from typing import Any

from src.config import get_settings
from src.models.schemas import EntityResult, IntentResult


INTENT_KEYWORDS = {
    "transaction_dispute": ["dispute", "chargeback", "unauthorized", "wrong charge", "incorrect transaction"],
    "billing_discrepancy": ["billing", "invoice error", "duplicate", "overcharged", "tax error"],
    "reconciliation_status": ["reconciliation", "reconcile", "statement status", "close status"],
    "payment_delay": ["payment delay", "late payment", "settlement delay", "not received"],
    "escalation_request": ["supervisor", "escalate", "manager", "speak to human", "complaint officer"],
}


def classify_intent(message: str) -> IntentResult:
    text = message.lower()
    if "dispute" in text or "chargeback" in text:
        return IntentResult(intent="transaction_dispute", confidence=0.93)

    best_intent = "general_inquiry"
    best_score = 0.0

    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in text)
        score = hits / max(len(keywords), 1)
        if hits > 0 and score >= best_score:
            best_score = score
            best_intent = intent

    confidence = 0.92 if best_score > 0 else 0.65
    if "fraud" in text or "regulatory" in text:
        best_intent = "escalation_request"
        confidence = 0.95

    return IntentResult(intent=best_intent, confidence=confidence)


def extract_entities(message: str) -> EntityResult:
    account_match = re.search(r"\bACC[- ]?\d{4,6}\b", message, re.I)
    txn_match = re.search(r"\bTXN[- ]?\d{4,8}\b", message, re.I)
    date_match = re.search(
        r"\b(20\d{2}[-/]\d{2}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+20\d{2})\b",
        message,
        re.I,
    )
    amount_match = re.search(r"\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", message)

    return EntityResult(
        account_number=account_match.group(0).upper().replace(" ", "-") if account_match else None,
        transaction_id=txn_match.group(0).upper().replace(" ", "-") if txn_match else None,
        date_range=date_match.group(0) if date_match else None,
        amount=float(amount_match.group(1).replace(",", "")) if amount_match else None,
    )


def should_escalate(message: str, intent: IntentResult, entities: EntityResult) -> bool:
    text = message.lower()
    if intent.intent == "escalation_request":
        return True
    if any(kw in text for kw in ["fraud", "regulatory", "legal action", "sec complaint"]):
        return True
    if entities.amount and entities.amount >= 50000:
        return True
    if intent.intent == "general_inquiry" and intent.confidence < 0.7:
        return True
    return False


class MockLLMClient:
    """Deterministic mock for local dev without Azure OpenAI credentials."""

    def complete(self, system: str, user: str) -> str:
        digest = hashlib.md5(user.encode()).hexdigest()[:8]
        return (
            "Based on our finance policy, your request has been reviewed. "
            f"Reference: {digest}. "
            "For transaction disputes, allow up to 5 business days for investigation. "
            "This response is informational and does not constitute legal advice."
        )


class AzureOpenAIClient:
    def __init__(self) -> None:
        from openai import AzureOpenAI

        settings = get_settings()
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self.deployment = settings.azure_openai_deployment

    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""


def get_llm_client() -> Any:
    settings = get_settings()
    if settings.use_azure_openai:
        return AzureOpenAIClient()
    return MockLLMClient()


def build_case_summary(
    message: str,
    intent: IntentResult,
    entities: EntityResult,
    account_snapshot: dict | None,
    sources: list[str],
) -> str:
    payload = {
        "customer_message": message,
        "intent": intent.intent,
        "confidence": intent.confidence,
        "entities": entities.model_dump(),
        "account_snapshot": account_snapshot,
        "rag_sources": sources,
    }
    return json.dumps(payload, indent=2)
