import re
import os
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from vector_store.embedder import get_embedder
from config.constants import PERSIST_DIRECTORY, COLLECTION_NAME

load_dotenv()


class MenstrualHealthRAG:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=os.getenv('GROQ_API_KEY'),
            model_name=os.getenv('GROQ_MODEL', 'llama3-8b-8192')
        )
        self.embedder = get_embedder()  # Initialize embedder once
        self.chat_history: Dict[str, List[Tuple[str, str]]] = {}

        # Cache for vector stores
        self.vectordb_cache: Dict[str, Chroma] = {}

    def _get_vectordb(self, language: str) -> Chroma:
        """Get or create Chroma vector store for a language with caching"""
        # Standardize language input to match how the vector DB was built
        lang = language.lower()
        if lang not in ["bangla", "english"]:
            lang = "english"  # default to english if unknown language

        if lang not in self.vectordb_cache:
            persist_dir = f"{PERSIST_DIRECTORY}/{COLLECTION_NAME}_{lang}"
            collection_name = f"{lang}_chunks"  # matches the naming in build_vector_db

            # Verify the vector store exists
            if not os.path.exists(persist_dir):
                raise ValueError(f"Vector database for language '{lang}' not found at {persist_dir}. "
                               "Please build it first using the /build-vectordb endpoint.")

            self.vectordb_cache[lang] = Chroma(
                embedding_function=self.embedder,
                collection_name=collection_name,
                persist_directory=persist_dir
            )
        return self.vectordb_cache[lang]

    def _clean_response(self, raw_result: str, query: str = None) -> str:
        """Remove unwanted tags, conversational fillers, and exact queries with associated punctuation from the
        model's output."""
        # Remove <think> tags
        cleaned = re.sub(r"<think>.*?</think>", "", raw_result, flags=re.DOTALL)
        # Remove Bangla conversational fillers
        cleaned = re.sub(r"আমি বলতে পারি\s*", "", cleaned, flags=re.UNICODE)
        # Remove exact query with associated punctuation (if query is provided)
        if query:
            # Escape special regex characters in the query to match it exactly
            escaped_query = re.escape(query)
            # Match query followed by optional punctuation (e.g., :, ?, !, .) and optional whitespace
            cleaned = re.sub(rf"\b{escaped_query}\s*[?:!.]?\s*", "", cleaned, flags=re.UNICODE)
        return cleaned.strip()

    def _format_history(self, history: List[Tuple[str, str]]) -> str:
        """Format last 3 chat history entries."""
        return "\n".join([f"User: {q}\nAssistant: {a}" for q, a in history[-3:]])

    def get_response(self, query: str, user_id: str) -> dict:
        """Main entry point: retrieve docs, build prompt, run LLM, update history."""
        # Detect language (using your existing function)
        from preprocess.language_detector import detect_language
        language = detect_language(query)

        try:
            # Get vector store
            vectordb = self._get_vectordb(language)

            # Retrieve relevant documents
            retriever = vectordb.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            context = "\n".join([doc.page_content for doc in docs])

            # Get user chat history
            user_history = self.chat_history.get(user_id, [])
            history_context = self._format_history(user_history)

            # Build and run prompt
            template = """You are an expert assistant specializing in menstrual health education, dedicated to providing accurate, empathetic, and accessible information.

            Context:
            {context}

            Chat History:
            {history}

            User Question: {question}

            Instructions:
            - Respond in the same language as the question strictly, using clear, concise, and simple language suitable for mobile devices.
            - Keep responses short (under 100 words) and focused, avoiding unnecessary details.
            - Provide factual, evidence-based answers, avoiding complex medical terminology.
            - Address sensitive topics with empathy, discretion, and cultural sensitivity.
            - Never repeat or rephrase the user's question in the response.
            - If the query is unclear, provide the most relevant information based on the context.
            - Offer concise, practical suggestions when appropriate.
            - Do not provide medical diagnoses; recommend consulting a healthcare professional for specific concerns.
            - Avoid conversational fillers or introductory phrases like 'আমি বলতে পারি' in Bangla responses; provide only the direct, relevant answer.
            - Do not Include the question in the response.
            """

            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm

            response = chain.invoke({
                'context': context,
                'history': history_context,
                'question': query
            })
            cleaned_response = self._clean_response(response.content, query=query)

            # Update history (limit to last 5 exchanges)
            self.chat_history.setdefault(user_id, []).append((query, cleaned_response))
            if len(self.chat_history[user_id]) > 5:
                self.chat_history[user_id] = self.chat_history[user_id][-5:]

            return {
                "query": query,
                "language": language,
                "response": cleaned_response
            }

        except ValueError as e:
            # Handle case where vector DB doesn't exist
            return {
                "query": query,
                "language": language,
                "response": f"Sorry, the knowledge base for {language} is not available yet."
            }
