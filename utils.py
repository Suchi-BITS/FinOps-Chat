import os
import requests
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Vector store directory
PERSIST_DIRECTORY = "chroma_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

def load_and_split_file(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    return text_splitter.split_documents(documents)

def load_and_split_web_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return []

def ingest_documents(docs):
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory=PERSIST_DIRECTORY
    )
    vectordb.persist()
    return vectordb
