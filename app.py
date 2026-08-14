import os
from typing import TypedDict

from fastapi import FastAPI

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnableLambda

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. GEMINI API KEY
# ============================================================

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set")


# ============================================================
# 2. GEMINI MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# ============================================================
# 3. DOCUMENTS
# ============================================================

documents = [
    Document(
        page_content="""
        Retrieval Augmented Generation, or RAG, is a technique
        where relevant information is retrieved from documents
        and provided to a language model before generating an answer.
        """
    ),

    Document(
        page_content="""
        LangGraph is a framework for building stateful applications
        and workflows using language models. A graph contains nodes
        and edges that control the execution flow.
        """
    ),

    Document(
        page_content="""
        LangChain is a framework for developing applications powered
        by language models. It provides components for prompts,
        models, retrievers, document loaders and vector stores.
        """
    )
]


# ============================================================
# 4. SPLIT DOCUMENTS
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)


# ============================================================
# 5. VECTOR STORE
# ============================================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 6. LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# ============================================================
# 7. RETRIEVE NODE
# ============================================================

def retrieve_node(state: RAGState):

    docs = retriever.invoke(
        state["question"]
    )

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


# ============================================================
# 8. GENERATE NODE
# ============================================================

def generate_node(state: RAGState):

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

Context:
{state["context"]}

Question:
{state["question"]}

If the answer is not available in the context,
say that the information is not available.
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# ============================================================
# 9. CREATE LANGGRAPH
# ============================================================

builder = StateGraph(RAGState)


builder.add_node(
    "retrieve",
    retrieve_node
)


builder.add_node(
    "generate",
    generate_node
)


builder.add_edge(
    START,
    "retrieve"
)


builder.add_edge(
    "retrieve",
    "generate"
)


builder.add_edge(
    "generate",
    END
)


rag_graph = builder.compile()


# ============================================================
# 10. CREATE PLAYGROUND INPUT
# ============================================================

def run_rag(input_data):

    question = input_data["question"]

    result = rag_graph.invoke({
        "question": question,
        "context": "",
        "answer": ""
    })

    return {
        "question": question,
        "answer": result["answer"]
    }


rag_runnable = RunnableLambda(run_rag)


# ============================================================
# 11. FASTAPI
# ============================================================

app = FastAPI(
    title="LangGraph RAG API",
    version="1.0"
)


# ============================================================
# 12. LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    rag_runnable,
    path="/agent"
)


# ============================================================
# 13. HOME
# ============================================================

@app.get("/")
def home():

    return {
        "message": "LangGraph RAG LangServe API is running"
    }
