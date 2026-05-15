"""
Phase 3 - Vector Store (RAG Engine)
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Constants ──────────────────────────────────────────────────────────────────


VECTORSTORE_PATH = "vectorstore/faiss_index"


EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── Embeddings ─────────────────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model (downloads on first run, cached after)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── Build ──────────────────────────────────────────────────────────────────────

def build_vectorstore(pdf_paths: list[str]) -> FAISS:
   
    print(f"Loading {len(pdf_paths)} PDF(s)...")
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
    print(f"Loaded {len(docs)} pages total.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Embed and store
    print("Embedding chunks... (first run downloads the model ~90MB)")
    embeddings = _get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)

    # Save to disk so we don't have to rebuild every time
    db.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")

    return db


# ── Load ───────────────────────────────────────────────────────────────────────

def load_vectorstore() -> FAISS | None:
    
    if not Path(VECTORSTORE_PATH).exists():
        return None
    print("Loading existing vector store from disk...")
    return FAISS.load_local(
        VECTORSTORE_PATH,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# ── Retrieve ───────────────────────────────────────────────────────────────────

def retrieve(query: str, db: FAISS, k: int = 4) -> str:
   
    results = db.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in results)