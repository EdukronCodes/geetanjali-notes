import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.config import get_settings
from src.database import init_db, seed_sample_data
from src.models.schemas import ChatRequest, ChatResponse, HealthResponse
from src.rag.vectorstore import FAISSVectorStore
from src.services.chatbot_service import ChatbotService

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Finance Customer Complaint Chatbot",
    description="RAG-powered finance ops chatbot with intent classification and escalation",
    version="1.0.0",
)
chatbot = ChatbotService()


@app.on_event("startup")
def startup() -> None:
    init_db()
    seed_sample_data()
    FAISSVectorStore().load()
    logger.info("Finance chatbot API started")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    index_exists = (Path(settings.faiss_index_path) / "index.faiss").exists()
    return HealthResponse(
        status="ok",
        azure_openai=settings.use_azure_openai,
        database=settings.db_engine,
        faiss_index=index_exists,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return chatbot.handle(request)
    except Exception as exc:
        logger.exception("Chat handling failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run("src.main:app", host=settings.app_host, port=settings.app_port, reload=True)
