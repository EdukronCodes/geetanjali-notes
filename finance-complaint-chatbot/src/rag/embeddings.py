from pathlib import Path

from src.config import get_settings


class EmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class MockEmbeddings(EmbeddingProvider):
    """Hash-based embeddings for offline FAISS indexing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import hashlib

        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [((b / 255.0) * 2 - 1) for b in digest] * 12  # 384-dim
            vectors.append(vec[:384])
        return vectors


class AzureOpenAIEmbeddings(EmbeddingProvider):
    def __init__(self) -> None:
        from openai import AzureOpenAI

        settings = get_settings()
        self.client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        self.deployment = settings.azure_openai_embedding_deployment

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.deployment, input=texts)
        return [item.embedding for item in response.data]


def get_embeddings() -> EmbeddingProvider:
    settings = get_settings()
    if settings.use_azure_openai:
        return AzureOpenAIEmbeddings()
    return MockEmbeddings()
