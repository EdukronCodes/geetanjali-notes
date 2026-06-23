from src.nlp.intent_classifier import get_llm_client
from src.rag.vectorstore import FAISSVectorStore

SYSTEM_PROMPT = """You are a finance operations assistant for customer complaints.
Use ONLY the provided context and account data. Be concise and professional.
Include compliance disclaimer: "This response is informational and does not constitute legal advice."
Do not promise refunds or share data about other customers.
If information is missing, ask for account number and transaction ID."""


class RAGChain:
    def __init__(self) -> None:
        self.vectorstore = FAISSVectorStore()
        self.llm = get_llm_client()

    def answer(self, message: str, account_context: str = "") -> tuple[str, list[str]]:
        self.vectorstore.load()
        hits = self.vectorstore.search(message, k=4)
        context_blocks = [f"[{h['source']}]\n{h['text']}" for h in hits]
        context = "\n\n".join(context_blocks)
        sources = list({h["source"] for h in hits})

        user_prompt = f"""Customer message:
{message}

Retrieved policy context:
{context}

Live account data:
{account_context or "Not available — request account number."}

Provide a helpful finance ops response."""

        reply = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        return reply, sources
