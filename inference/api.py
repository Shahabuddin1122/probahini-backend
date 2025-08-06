from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from langchain.document_loaders import TextLoader

from config.constants import PERSIST_DIRECTORY, COLLECTION_NAME
from inference.pipeline import run_rag_pipeline
from splitter.text_splitter import get_markdown_splitter
from vector_store.embedder import get_embedder
from vector_store.store import build_vector_store

router = APIRouter()


class VectorDBRequest(BaseModel):
    language: str  # "bangla" or "english"


class ChatRequest(BaseModel):
    user_id: str
    query: str


@router.post("/build-vectordb")
def build_vector_db(request: VectorDBRequest):
    lang = request.language.lower()
    if lang not in ["bangla", "english"]:
        raise HTTPException(status_code=400, detail="Language must be 'bangla' or 'english'.")

    data_dir = Path(f"data/raw/{lang}")
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Directory {data_dir} not found.")

    files = list(data_dir.glob("*.txt"))
    if not files:
        raise HTTPException(status_code=404, detail=f"No .txt files found in {data_dir}.")

    splitter = get_markdown_splitter()
    embedder = get_embedder()

    documents = []
    for file in files:
        loader = TextLoader(str(file), encoding='utf-8')
        loaded_docs = loader.load()
        for doc in loaded_docs:
            chunks = splitter.split_text(doc.page_content)
            documents.extend(chunks)

    if not documents:
        raise HTTPException(status_code=400, detail="No documents found after splitting.")

    persist_dir = f"{PERSIST_DIRECTORY}/{COLLECTION_NAME}_{lang}"
    build_vector_store(documents, embedder, persist_dir, collection_name=f"{lang}_chunks")

    return {
        "message": f"Stored {len(documents)} chunks into ChromaDB at '{persist_dir}'",
        "chunks_stored": len(documents),
        "path": persist_dir
    }


@router.post("/chat")
def chat_endpoint(request: ChatRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = run_rag_pipeline(query, request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    return response
