from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
from config.constants import GROQ_MODEL

load_dotenv()


def get_llm():
    return ChatGroq(
        model_name=GROQ_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

