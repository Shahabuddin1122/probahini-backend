from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from inference.predictor import get_llm
from preprocess.language_detector import detect_language

import re

# Global cache
_embedder = None
_vectordb_cache = {}
_qa_chain_cache = {}
_chat_history = {}

# Prompt Template (reused)
template = """You are a knowledgeable assistant helping with menstrual health education.

Use the following context and chat history to answer the question. Be factual, clear, concise, and respond in the same language as the question.

Context:
{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate(input_variables=["context", "question"], template=template)


def clean_response(raw_result: str) -> str:
    return re.sub(r"<think>.*?</think>", "", raw_result, flags=re.DOTALL).strip()


def format_chat_history(history: list[tuple[str, str]]) -> str:
    return "\n\n".join([f"Q: {q}\nA: {a}" for q, a in history[-3:]])  # last 3 rounds


def run_rag_pipeline(query: str, user_id: str) -> dict:
    global _embedder, _vectordb_cache, _qa_chain_cache, _chat_history

    language = detect_language(query)

    # Load components
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

    # Get chat history
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

    # Swap prompt in chain
    qa_chain = _qa_chain_cache[language]
    qa_chain.combine_documents_chain.llm_chain.prompt = contextual_prompt

    response = qa_chain.invoke(query)
    result = clean_response(response.get("result", ""))

    # Update chat history
    _chat_history.setdefault(user_id, []).append((query, result))

    return {
        "query": query,
        "language": language,
        "response": result
    }
