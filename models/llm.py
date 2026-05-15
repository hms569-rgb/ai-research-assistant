"""
Phase 1 - LLM Setup

"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv()


def get_llm(temperature: float = 0.3) -> ChatGroq:
    
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found!\n"
            "Make sure your .env file exists and contains:\n"
            "GROQ_API_KEY=your_key_here"
        )

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        groq_api_key=api_key,
    )