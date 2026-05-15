"""
Utility helpers shared across the app.
Small reusable functions that don't belong to any specific phase.
"""

import tempfile
from pathlib import Path


# ── PDF Helper ─────────────────────────────────────────────────────────────────

def save_uploaded_pdf(uploaded_file) -> str:
    """
    Streamlit's file uploader gives us a file-like object in memory.
    But PyPDFLoader needs an actual file path on disk.
    
    This function:
        1. Takes the uploaded file from Streamlit
        2. Saves it to a temporary location on disk
        3. Returns the path so PyPDFLoader can read it

    Example:
        path = save_uploaded_pdf(uploaded_file)
        → "C:/Users/temp/tmpxyz123.pdf"
    """
    suffix = Path(uploaded_file.name).suffix  # gets ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


# ── Display Helpers ────────────────────────────────────────────────────────────

def format_mode_label(mode: str) -> str:
    """
    Convert mode string to a nice display label with emoji.

    Example:
        format_mode_label("beginner") → "🟢 Beginner"
        format_mode_label("expert")   → "🔵 Expert"
    """
    labels = {
        "beginner": "🟢 Beginner",
        "expert":   "🔵 Expert",
    }
    return labels.get(mode, mode)


def truncate(text: str, max_chars: int = 300) -> str:
    """
    Truncate long text for display purposes.
    Adds "..." at the end if text was cut.

    Example:
        truncate("A very long text...", max_chars=10)
        → "A very lon..."
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def format_stats_for_display(stats: dict) -> str:
    """
    Takes the output of word_count() and formats it
    as a clean readable string for the UI.

    Example output:
        📝 Words: 245
        📖 Read time: 1.2 min
        🎓 Study time: 2.4 min
        💬 Sentences: 18
        📄 Paragraphs: 4
    """
    return (
        f"📝 Words: {stats['words']}\n"
        f"💬 Sentences: {stats['sentences']}\n"
        f"📄 Paragraphs: {stats['paragraphs']}\n"
        f"🔤 Characters: {stats['characters']}\n"
        f"📖 Read time: {stats['estimated_read_time']}\n"
        f"🎓 Study time: {stats['estimated_study_time']}"
    )