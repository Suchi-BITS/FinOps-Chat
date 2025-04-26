from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.chat_models.base import BaseChatModel
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
def get_groq_llm() -> BaseChatModel:
    return ChatGroq(
        temperature=0.2,
        model_name="llama3-8b-8192",  # Ensure you have access to this model on Groq
        streaming=True,
    )
