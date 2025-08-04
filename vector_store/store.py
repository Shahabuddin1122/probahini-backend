from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings


def build_vector_store(documents, embedding: Embeddings, persist_dir: str, collection_name: str):
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_dir
    )
    db.persist()
    return db
