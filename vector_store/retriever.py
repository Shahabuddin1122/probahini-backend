from langchain.vectorstores import Chroma
from config.constants import PERSIST_DIRECTORY, COLLECTION_NAME
from vector_store.embedder import get_embedder


def get_vectordb(language: str):
    persist_dir = f"{PERSIST_DIRECTORY}/{COLLECTION_NAME}_{language}"
    collection_name = f"menstrual_health_chunks_{language[:2]}"
    embedder = get_embedder()

    return Chroma(
        embedding_function=embedder,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
