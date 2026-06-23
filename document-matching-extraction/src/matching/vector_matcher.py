import hashlib
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src.config import get_settings
from src.models.schemas import ExtractedFields, MatchResult


class EmbeddingProvider:
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class MockEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vec = [((b / 255.0) * 2 - 1) for b in digest] * 12
            vectors.append(vec[:384])
        return np.array(vectors, dtype="float32")


class AzureEmbeddings(EmbeddingProvider):
    def embed(self, texts: list[str]) -> np.ndarray:
        from openai import AzureOpenAI

        settings = get_settings()
        client = AzureOpenAI(
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
        )
        response = client.embeddings.create(model=settings.azure_openai_embedding_deployment, input=texts)
        return np.array([item.embedding for item in response.data], dtype="float32")


def _get_embedder() -> EmbeddingProvider:
    return AzureEmbeddings() if get_settings().use_azure_openai else MockEmbeddings()


class VectorMatcher:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_path = Path(self.settings.faiss_index_path)
        self.records: list[dict] = []
        self.index: faiss.IndexFlatL2 | None = None

    def build_index(self) -> None:
        csv_path = Path(self.settings.reference_csv_path)
        if not csv_path.is_absolute():
            csv_path = self.settings.project_root / csv_path
        df = pd.read_csv(csv_path)
        self.records = df.to_dict(orient="records")
        texts = [
            f"{r['vendor_name']} {r['reference_id']} {r['document_type']} amount {r['amount']} cost center {r['cost_center']}"
            for r in self.records
        ]
        vectors = _get_embedder().embed(texts)
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path / "index.faiss"))
        (self.index_path / "records.json").write_text(json.dumps(self.records), encoding="utf-8")
        self.index = index

    def load(self) -> None:
        index_file = Path(self.settings.faiss_index_path) / "index.faiss"
        if not index_file.exists():
            self.build_index()
        else:
            self.index = faiss.read_index(str(index_file))
            self.records = json.loads((Path(self.settings.faiss_index_path) / "records.json").read_text())

    def match(self, extracted: ExtractedFields, document_text: str) -> MatchResult | None:
        self.load()

        ref_key = extracted.po_number or extracted.contract_id or extracted.invoice_number
        if ref_key:
            for record in self.records:
                if record["reference_id"].upper() == ref_key.upper():
                    return MatchResult(
                        reference_id=record["reference_id"],
                        vendor_name=record["vendor_name"],
                        similarity_score=0.98,
                        matched=True,
                    )

        query = (
            f"{extracted.vendor or ''} {ref_key or ''} "
            f"{extracted.document_type or ''} amount {extracted.amount or ''} cost center {extracted.cost_center or ''} "
            f"{document_text[:500]}"
        )
        vector = _get_embedder().embed([query])
        distances, indices = self.index.search(vector, 1)
        idx = int(indices[0][0])
        if idx < 0:
            return None
        record = self.records[idx]
        similarity = max(0.0, 1.0 / (1.0 + float(distances[0][0])))
        return MatchResult(
            reference_id=record["reference_id"],
            vendor_name=record["vendor_name"],
            similarity_score=round(similarity, 4),
            matched=similarity >= self.settings.confidence_threshold,
        )
