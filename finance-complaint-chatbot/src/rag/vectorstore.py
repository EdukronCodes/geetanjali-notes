import json
from pathlib import Path

import faiss
import numpy as np

from src.config import get_settings
from src.rag.embeddings import get_embeddings


class FAISSVectorStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.index_path = Path(self.settings.faiss_index_path)
        self.meta_path = self.index_path / "metadata.json"
        self.index: faiss.IndexFlatL2 | None = None
        self.documents: list[dict] = []

    def _chunk_text(self, text: str, source: str, chunk_size: int = 500) -> list[dict]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size // 5):
            chunk_words = words[i : i + chunk_size // 5]
            if chunk_words:
                chunks.append({"text": " ".join(chunk_words), "source": source})
        return chunks or [{"text": text, "source": source}]

    def build_from_knowledge_base(self) -> None:
        kb_path = Path(self.settings.knowledge_base_path)
        if not kb_path.is_absolute():
            kb_path = self.settings.project_root / kb_path

        all_chunks: list[dict] = []
        for md_file in kb_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            all_chunks.extend(self._chunk_text(content, md_file.name))

        if not all_chunks:
            raise FileNotFoundError(f"No knowledge base documents found in {kb_path}")

        embeddings = get_embeddings()
        vectors = np.array(embeddings.embed_documents([c["text"] for c in all_chunks]), dtype="float32")
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        self.index_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path / "index.faiss"))
        self.meta_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
        self.index = index
        self.documents = all_chunks

    def load(self) -> None:
        index_file = self.index_path / "index.faiss"
        if not index_file.exists():
            self.build_from_knowledge_base()
            return
        self.index = faiss.read_index(str(index_file))
        self.documents = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def search(self, query: str, k: int = 4) -> list[dict]:
        if self.index is None:
            self.load()
        embeddings = get_embeddings()
        vector = np.array([embeddings.embed_query(query)], dtype="float32")
        distances, indices = self.index.search(vector, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                doc = dict(self.documents[idx])
                doc["score"] = float(dist)
                results.append(doc)
        return results
