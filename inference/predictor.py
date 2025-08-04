from langchain_groq import ChatGroq

from config.constants import GROQ_MODEL


def get_llm():
    return ChatGroq(model_name=GROQ_MODEL, temperature=0)
