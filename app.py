import os
from typing import TypedDict

from fastapi import FastAPI
from pydantic import BaseModel

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from langchain_core.vectorstores import InMemoryVectorStore

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END

from langserve import add_routes


# ============================================================
# 1. GEMINI API KEY
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY environment variable is not set."
    )


# ============================================================
# 2. GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)


# ============================================================
# 3. GEMINI EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# 4. DOCUMENTS
# ============================================================

documents = [

    Document(
        page_content="""
        Retrieval Augmented Generation, commonly called RAG,
        is a technique that retrieves relevant information
        from documents and provides that information to a
        language model before generating an answer.
        """
    ),

    Document(
        page_content="""
        LangGraph is a framework for building stateful workflows
        and applications with language models. A LangGraph
        application contains nodes and edges that control
        how information flows through the application.
        """
    ),

    Document(
        page_content="""
        LangChain is a framework for building applications
        powered by large language models. It provides tools
        for prompts, document loaders, embeddings, retrievers,
        vector stores and language models.
        """
    ),

    Document(
        page_content="""
        A vector database stores numerical representations
        called embeddings. Similarity search can be used to
        find documents that are semantically related to a
        user's question.
        """
    ),

]


# ============================================================
# 5. TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)


chunks = text_splitter.split_documents(
    documents
)


# ============================================================
# 6. VECTOR STORE
# ============================================================

vector_store = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
)


# ============================================================
# 7. RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ============================================================
# 8. LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict):

    question: str
    context: str
    answer: str


# ============================================================
# 9. RETRIEVE NODE
# ============================================================

def retrieve_node(
    state: RAGState
):

    question = state["question"]

    documents = retriever.invoke(
        question
    )

    context = "\n\n---\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "context": context
    }


# ============================================================
# 10. GENERATE NODE
# ============================================================

def generate_node(
    state: RAGState
):

    question = state["question"]

    context = state["context"]

    prompt = f"""
You are a helpful Retrieval Augmented Generation assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I could not find enough information in the provided documents."

Do not invent information.

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}

---------------- ANSWER ----------------
"""

    response = llm.invoke(
        prompt
    )

    if isinstance(
        response.content,
        str
    ):
        answer = response.content

    else:
        answer = str(
            response.content
        )

    return {
        "answer": answer
    }


# ============================================================
# 11. BUILD LANGGRAPH
# ============================================================

graph_builder = StateGraph(
    RAGState
)


graph_builder.add_node(
    "retrieve",
    retrieve_node
)


graph_builder.add_node(
    "generate",
    generate_node
)


graph_builder.add_edge(
    START,
    "retrieve"
)


graph_builder.add_edge(
    "retrieve",
    "generate"
)


graph_builder.add_edge(
    "generate",
    END
)


rag_graph = graph_builder.compile()


# ============================================================
# 12. TEST LANGGRAPH
# ============================================================

def test_graph():

    result = rag_graph.invoke({

        "question":
            "What is Retrieval Augmented Generation?",

        "context":
            "",

        "answer":
            "",
    })

    print(
        result["answer"]
    )


# ============================================================
# 13. LANGSERVE INPUT
# ============================================================

class RAGInput(BaseModel):

    question: str


# ============================================================
# 14. FUNCTION FOR LANGSERVE
# ============================================================

def run_rag(
    input_data
):

    # Handle dictionary input
    if isinstance(
        input_data,
        dict
    ):

        question = input_data.get(
            "question"
        )

    # Handle Pydantic input
    else:

        question = input_data.question


    if not question:

        return {
            "error":
                "Question is required."
        }


    result = rag_graph.invoke({

        "question":
            question,

        "context":
            "",

        "answer":
            "",
    })


    return {

        "question":
            question,

        "answer":
            result["answer"],

    }


# ============================================================
# 15. CREATE LANGCHAIN RUNNABLE
# ============================================================

rag_runnable = RunnableLambda(
    run_rag
).with_types(
    input_type=RAGInput
)


# ============================================================
# 16. FASTAPI APPLICATION
# ============================================================

app = FastAPI(

    title="LangGraph RAG API",

    description=(
        "LangChain + LangGraph + Gemini "
        "RAG application"
    ),

    version="1.0.0",
)


# ============================================================
# 17. LANGSERVE ROUTE
# ============================================================

add_routes(

    app,

    rag_runnable,

    path="/agent",

)


# ============================================================
# 18. HOME ROUTE
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "LangGraph RAG API is running",

        "playground":
            "/agent/playground/",

        "docs":
            "/docs",

    }


# ============================================================
# 19. SIMPLE HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "healthy"

    }


# ============================================================
# 20. RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app:app",

        host="0.0.0.0",

        port=int(
            os.getenv(
                "PORT",
                "8000"
            )
        ),

        reload=False,
    )
