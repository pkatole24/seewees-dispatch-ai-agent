from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title).strip()


def _infer_stream(section_title: str, content: str) -> str:
    lowered = f"{section_title}\n{content}".lower()
    reference_markers = (
        "appendix",
        "item master",
        "alias",
        "legacy",
        "mapping",
        "corridor catalog",
        "reporting requirements",
        "weather triggers",
        "travel time buffer",
        "data quality rules",
    )
    if any(marker in lowered for marker in reference_markers) or "|" in content:
        return "reference"
    return "policy"


def docs_to_snippets(docs: Iterable[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


@dataclass
class KnowledgeRag:
    persist_dir: str = "chroma_db"
    collection_name: str = "business_context"

    def _fingerprint(self, source_paths: Sequence[str]) -> str:
        payload_parts: list[str] = []
        for source_path in sorted(set(source_paths)):
            path = Path(source_path)
            stat = path.stat()
            payload_parts.append(
                f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
            )
        payload = "||".join(payload_parts)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _split_markdown_sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_title = "Document Overview"
        current_lines: list[str] = []

        for line in text.splitlines():
            if re.match(r"^#{1,6}\s+", line):
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = _normalize_title(re.sub(r"^#{1,6}\s+", "", line))
                current_lines = []
                continue
            current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))
        return [(title, body) for title, body in sections if body]

    def _load_markdown_documents(self, source_path: str) -> list[Document]:
        path = Path(source_path)
        text = path.read_text(encoding="utf-8")
        base_docs: list[Document] = []

        for section_title, content in self._split_markdown_sections(text):
            stream = _infer_stream(section_title, content)
            base_docs.append(
                Document(
                    page_content=f"{section_title}\n\n{content}",
                    metadata={
                        "source_path": str(path),
                        "source_name": path.name,
                        "source_kind": "markdown",
                        "section_title": section_title,
                        "stream": stream,
                    },
                )
            )

        chunked_docs: list[Document] = []
        for doc in base_docs:
            chunk_size = 1400 if doc.metadata.get("stream") == "reference" else 1000
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=150,
            )
            chunked_docs.extend(splitter.split_documents([doc]))
        return chunked_docs

    def _load_pdf_documents(self, source_path: str) -> list[Document]:
        path = Path(source_path)
        docs = PyPDFLoader(str(path)).load()
        for doc in docs:
            doc.metadata["source_path"] = str(path)
            doc.metadata["source_name"] = path.name
            doc.metadata["source_kind"] = "pdf"
            doc.metadata["section_title"] = f"Page {doc.metadata.get('page', '?')}"
            doc.metadata["stream"] = "policy"

        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        return splitter.split_documents(docs)

    def _load_source(self, source_path: str) -> list[Document]:
        suffix = Path(source_path).suffix.lower()
        if suffix == ".pdf":
            return self._load_pdf_documents(source_path)
        if suffix in {".md", ".txt"}:
            return self._load_markdown_documents(source_path)
        raise ValueError(f"Unsupported knowledge source: {source_path}")

    def build(self, source_paths: Sequence[str]) -> Chroma:
        valid_paths = [str(Path(path)) for path in source_paths if Path(path).exists()]
        if not valid_paths:
            raise FileNotFoundError("No valid knowledge sources found for indexing.")

        os.makedirs(self.persist_dir, exist_ok=True)

        fingerprint = self._fingerprint(valid_paths)
        collection_name = f"{self.collection_name}_{fingerprint}"
        vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=OpenAIEmbeddings(),
            persist_directory=self.persist_dir,
        )

        existing_count = 0
        try:
            existing_count = vectordb._collection.count()
        except Exception:
            existing_count = 0

        if existing_count > 0:
            return vectordb

        docs: list[Document] = []
        for source_path in valid_paths:
            docs.extend(self._load_source(source_path))

        vectordb.add_documents(docs)
        return vectordb

    def retrieve(
        self,
        vectordb: Chroma,
        query: str,
        *,
        k: int = 6,
        stream: str | None = None,
    ) -> list[Document]:
        if stream:
            docs = vectordb.similarity_search(query, k=k, filter={"stream": stream})
            if docs:
                return docs
        return vectordb.similarity_search(query, k=k)

    def retriever(self, vectordb: Chroma, k: int = 6):
        return vectordb.as_retriever(search_kwargs={"k": k})


@dataclass
class PdfRag(KnowledgeRag):
    def build(self, pdf_path: str | Sequence[str]) -> Chroma:  # type: ignore[override]
        if isinstance(pdf_path, str):
            return super().build([pdf_path])
        return super().build(pdf_path)
