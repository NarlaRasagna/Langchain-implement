import os

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

from typing import TypedDict


# -----------------------------
# Gemini API
# -----------------------------

api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# -----------------------------
# Sample documents
# -----------------------------

documents = [
    Document(
        page_content="""
        Retrieval Augmented Generation, or RAG, is a technique where
        relevant information is retrieved from documents and provided
        to a language model before generating an answer.
        """
    ),
    Document(
        page_content="""
        LangGraph is a framework for building stateful applications
        and workflows using language models. A graph consists of nodes
        and edges that control the execution flow.
        """
    )
]


# -----------------------------
# Split documents
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)


# -----------------------------
# Vector store
# -----------------------------

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# -----------------------------
# LangGraph State
# -----------------------------

class RAGState(TypedDict):
    question: str
    context: str
    answer: str


# -----------------------------
# Retrieve node
# -----------------------------

def retrieve_node(state: RAGState):

    docs = retriever.invoke(state["question"])

    context = "\n\n".join(
        doc.page_content
        for doc in docs
    )

    return {
        "context": context
    }


# -----------------------------
# Generate node
# -----------------------------

def generate_node(state: RAGState):

    prompt = f"""
    Answer the question using only the context below.

    Context:
    {state["context"]}

    Question:
    {state["question"]}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content
    }


# -----------------------------
# Build LangGraph
# -----------------------------

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


# -----------------------------
# FastAPI
# -----------------------------

app = FastAPI(
    title="LangGraph RAG API"
)


class Question(BaseModel):
    question: str


@app.get("/")
def home():

    return {
        "message": "LangGraph RAG API is running"
    }


@app.post("/rag")
def rag_endpoint(data: Question):

    result = rag_graph.invoke({
        "question": data.question,
        "context": "",
        "answer": ""
    })

    return {
        "question": data.question,
        "answer": result["answer"]
    }
