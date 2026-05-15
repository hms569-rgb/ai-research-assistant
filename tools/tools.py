"""
Phase 4 - Tools

"""

import re
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from models.llm import get_llm


# ── Tool 1: Text Summarizer ────────────────────────────────────────────────────

def summarize_text(text: str, mode: str = "beginner") -> str:
    """
    Summarize any text using the LLM.

    """
    if not text.strip():
        return "No text provided to summarize."

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research summarization assistant.
Mode: {mode}
- beginner: Use plain language, short sentences, no jargon.
- expert:   Use technical terms, be dense and precise.

Give a 3-4 sentence summary. No bullet points. Just clean prose."""),
        ("human", "{text}"),
    ])

    llm = get_llm(temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text, "mode": mode})


# ── Tool 2: Keyword Extractor ──────────────────────────────────────────────────

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract the most important words from a piece of text.

    """
    if not text.strip():
        return []

    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "day",
        "get", "has", "him", "his", "how", "its", "may", "new",
        "now", "old", "see", "two", "who", "did", "she", "use",
        "way", "will", "with", "this", "that", "they", "been",
        "from", "have", "more", "also", "into", "than", "then",
        "them", "were", "what", "when", "your", "said", "each",
        "which", "their", "time", "would", "there", "could",
        "other", "about", "these", "some", "just", "very",
    }

    words = re.findall(r'\b[a-z]{4,}\b', text.lower())

    freq: dict[str, int] = {}
    for word in words:
        if word not in stopwords:
            freq[word] = freq.get(word, 0) + 1

    return sorted(freq, key=lambda w: freq[w], reverse=True)[:top_n]


# ── Tool 3: Word Count & Reading Stats ────────────────────────────────────────

def word_count(text: str) -> dict:
    
    if not text.strip():
        return {
            "words": 0,
            "sentences": 0,
            "characters": 0,
            "paragraphs": 0,
            "estimated_read_time": "0 min",
            "estimated_study_time": "0 min",
        }

    words      = text.split()
    sentences  = [s for s in re.split(r'[.!?]+', text) if s.strip()]
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    read_time  = round(len(words) / 200, 1)
    study_time = round(read_time * 2, 1)

    return {
        "words": len(words),
        "sentences": len(sentences),
        "characters": len(text),
        "paragraphs": len(paragraphs),
        "estimated_read_time": f"{read_time} min",
        "estimated_study_time": f"{study_time} min",
    }