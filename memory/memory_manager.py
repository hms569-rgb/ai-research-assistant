"""
Phase 2 - Conversation Memory
This makes the assistant remember what was said earlier in a conversation.

"""

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory



_store: dict[str, ChatMessageHistory] = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


def clear_session(session_id: str) -> None:
   
    _store.pop(session_id, None)


def wrap_with_memory(runnable):
    
    return RunnableWithMessageHistory(
        runnable,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )