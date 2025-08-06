import re
from typing import Dict, Tuple, List

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from inference.predictor import get_llm
from preprocess.language_detector import detect_language

# Global cache
_embedder = None
_vectordb_cache: Dict[str, any] = {}
_qa_chain_cache: Dict[str, RetrievalQA] = {}
_chat_history: Dict[str, List[Tuple[str, str]]] = {}

# Base prompt template
template = """You are a knowledgeable assistant helping with menstrual health education.

Use the following context and chat history to answer the question. Be factual, clear, concise, and respond in the same language as the question.

Context:
{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate(input_variables=["context", "question"], template=template)


def clean_response(raw_result: str) -> str:
    """Clean unnecessary tags from response."""
    return re.sub(r"<think>.*?</think>", "", raw_result, flags=re.DOTALL).strip()


def format_chat_history(history: List[Tuple[str, str]]) -> str:
    """Format the last 3 chat history items into a string."""
    return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history[-3:]])


def preload_resources(languages: List[str]) -> None:
    global _embedder, _vectordb_cache, _qa_chain_cache

    from vector_store.embedder import get_embedder
    from vector_store.retriever import get_vectordb

    _embedder = get_embedder()
    llm = get_llm()

    for lang in languages:
        vectordb = get_vectordb(lang)
        _vectordb_cache[lang] = vectordb

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        _qa_chain_cache[lang] = qa_chain

        try:
            qa_chain.invoke("Hello")
        except Exception as e:
            print(f"Warm-up failed for language '{lang}': {e}")

    print("Resources Loading completed and warmed up")


def run_rag_pipeline(query: str, user_id: str) -> dict:
    """
    Main function to handle RAG-based question answering with contextual chat history.
    """
    global _embedder, _vectordb_cache, _qa_chain_cache, _chat_history

    language = detect_language(query)

    # If this is the first time we are handling this language, load it
    if language not in _vectordb_cache:
        from vector_store.embedder import get_embedder
        from vector_store.retriever import get_vectordb

        _embedder = _embedder or get_embedder()
        vectordb = get_vectordb(language)
        _vectordb_cache[language] = vectordb

        llm = get_llm()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        _qa_chain_cache[language] = qa_chain

    # Retrieve chat history and format
    user_history = _chat_history.get(user_id, [])
    history_context = format_chat_history(user_history)

    # Inject chat history into the prompt dynamically
    contextual_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template.replace(
            "Context:",
            f"Chat History:\n{history_context}\n\nContext:" if history_context else "Context:"
        )
    )

    # Swap prompt in the QA chain
    qa_chain = _qa_chain_cache[language]
    qa_chain.combine_documents_chain.llm_chain.prompt = contextual_prompt

    # Run the query
    response = qa_chain.invoke(query)
    result = clean_response(response.get("result", ""))

    # Save response in history
    _chat_history.setdefault(user_id, []).append((query, result))

    return {
        "query": query,
        "language": language,
        "response": result
    }
