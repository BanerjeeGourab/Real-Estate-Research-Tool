from uuid import uuid4
from pathlib import Path
import os
import certifi
import requests
from bs4 import BeautifulSoup
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.schema import Document

# Load .env for local development
load_dotenv()

# Load API key: Streamlit secrets > .env
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets."
    )

# Ensure SSL works for requests
os.environ["SSL_CERT_FILE"] = certifi.where()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

# Global objects
llm = None
vector_store = None


def initialize_components():
    """Initialize the LLM and Vector Store once"""
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.9,
            max_tokens=500,
            api_key=GROQ_API_KEY
        )

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def fetch_text_from_url(url):
    """Fetch text content from a webpage using requests + BeautifulSoup"""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])
            return text
        else:
            return ""
    except Exception:
        return ""


def process_urls(urls):
    """
    Scrapes data from given URLs and stores them in a vector database
    """
    yield "Initializing Components...✅"
    initialize_components()

    yield "Resetting vector store...✅"
    global vector_store
    ef = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

    # Delete old collection if exists
    try:
        vector_store._client.delete_collection(name=COLLECTION_NAME)
        yield "Old collection deleted...✅"
    except Exception:
        yield "No existing collection to delete...✅"

    # Recreate vector store (fresh collection)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=ef,
        persist_directory=str(VECTORSTORE_DIR)
    )

    yield "Fetching data from URLs...✅"
    data = []
    for url in urls:
        text = fetch_text_from_url(url)
        if text:
            data.append(Document(page_content=text, metadata={"source": url}))
        else:
            yield f"Warning: Could not fetch content from {url}"

    if not data:
        yield "No content could be fetched from any URL. Aborting..."
        return

    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Finished adding docs to vector database...✅"


def generate_answer(query):
    """Queries the vector database using LLM"""
    if not vector_store:
        raise RuntimeError("Vector database is not initialized. Please process URLs first.")

    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    result = qa_chain({"query": query})
    answer = result.get("result", "")
    sources = ""
    if "source_documents" in result:
        sources = "\n".join([doc.metadata.get("source", "Unknown") for doc in result["source_documents"]])

    return answer, sources



