from __future__ import annotations

from dataclasses import dataclass
import os
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


@dataclass
class PdfRag:
    persist_dir: str = "chroma_db"
    collection_name: str = "business_context"

    def _fingerprint(self, pdf_path: str) -> str:
        """Stable fingerprint to detect if this exact PDF has been indexed."""
        st = os.stat(pdf_path)
        payload = f"{os.path.abspath(pdf_path)}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def build(self, pdf_path: str) -> Chroma:
        # Ensure persist dir exists before we touch it
        os.makedirs(self.persist_dir, exist_ok=True)

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=embeddings,
            persist_directory=self.persist_dir,
        )

        fp = self._fingerprint(pdf_path)
        marker = os.path.join(self.persist_dir, f".indexed_{self.collection_name}_{fp}")

        # If marker exists AND collection is non-empty, reuse
        if os.path.exists(marker):
            try:
                if vectordb._collection.count() > 0:
                    return vectordb
            except Exception:
                # If count check fails, fall through and rebuild
                pass

        docs = PyPDFLoader(pdf_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
        chunks = splitter.split_documents(docs)

        vectordb.add_documents(chunks)

        with open(marker, "w", encoding="utf-8") as f:
            f.write(fp)

        return vectordb

    def retriever(self, vectordb: Chroma, k: int = 6):
        return vectordb.as_retriever(search_kwargs={"k": k})