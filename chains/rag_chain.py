"""
Phase 3 - Chains

"""

from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from models.llm import get_llm
from prompts.prompt_templates import chat_prompt, rag_prompt, summarize_prompt
from memory.memory_manager import wrap_with_memory


# ── Chain 1: Plain Chat (Phase 1 + 2) ─────────────────────────────────────────

def build_chat_chain():
    
    llm = get_llm()

    # The pipe operator | chains steps together
    # chat_prompt formats the message → llm generates response → StrOutputParser converts to string
    base_chain = chat_prompt | llm | StrOutputParser()

    # Wrap with memory so it remembers previous messages
    return wrap_with_memory(base_chain)


# ── Chain 2: RAG Chain (Phase 3) ──────────────────────────────────────────────

def build_rag_chain(vectorstore):
    
    llm = get_llm()

    def retrieve_and_inject(inputs: dict) -> dict:
       
        query = inputs["input"]
        results = vectorstore.similarity_search(query, k=4)
        inputs["context"] = "\n\n---\n\n".join(
            doc.page_content for doc in results
        )
        return inputs

    base_chain = (
        RunnableLambda(retrieve_and_inject)  # retrieve context from PDFs
        | rag_prompt                          # format prompt with context
        | llm                                 # generate answer
        | StrOutputParser()                   # convert to plain string
    )

    return wrap_with_memory(base_chain)


# ── Chain 3: Summarize Chain (standalone) ─────────────────────────────────────

def build_summarize_chain():
    
    llm = get_llm(temperature=0.1)
    return summarize_prompt | llm | StrOutputParser()