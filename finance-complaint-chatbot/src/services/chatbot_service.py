from src.database import Complaint, get_session
from src.models.schemas import ChatRequest, ChatResponse, EntityResult, IntentResult
from src.nlp.intent_classifier import (
    build_case_summary,
    classify_intent,
    extract_entities,
    should_escalate,
)
from src.rag.chain import RAGChain
from src.services.account_lookup import format_account_context, lookup_account
from src.services.escalation_service import send_escalation_email


class ChatbotService:
    def __init__(self) -> None:
        self.rag = RAGChain()

    def handle(self, request: ChatRequest) -> ChatResponse:
        intent = classify_intent(request.message)
        entities = extract_entities(request.message)
        escalated = should_escalate(request.message, intent, entities)

        with get_session() as session:
            snapshot = lookup_account(session, entities.account_number)
            account_ctx = format_account_context(snapshot)
            reply, sources = self.rag.answer(request.message, account_ctx)

            case_summary = None
            if escalated:
                case_summary = build_case_summary(
                    request.message,
                    intent,
                    entities,
                    snapshot.model_dump() if snapshot else None,
                    sources,
                )
                send_escalation_email(
                    subject=f"[Finance Chatbot Escalation] {intent.intent} — {entities.account_number or 'No account'}",
                    body=case_summary,
                )

            complaint = Complaint(
                account_id=entities.account_number,
                transaction_id=entities.transaction_id,
                intent=intent.intent,
                message=request.message,
                bot_response=reply,
                escalated=1 if escalated else 0,
                case_summary=case_summary,
            )
            session.add(complaint)
            session.flush()

            return ChatResponse(
                reply=reply,
                intent=intent.intent,
                confidence=intent.confidence,
                entities=entities,
                escalated=escalated,
                case_summary=case_summary,
                complaint_id=complaint.complaint_id,
                sources=sources,
            )
