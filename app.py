import streamlit as st
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import uuid

# --------------- CONFIGURATION ---------------
st.set_page_config(page_title="FinOps Knowledge Chat", layout="wide")

# Predefined URLs (hardcoded)
URLS = [
    "https://www.infracost.io/finops-policies/amazon-ec2-consider-using-latest-generation-instances-for-g-family-instances/",
    # Add more URLs here
]

# ChromaDB Settings
chroma_client = chromadb.Client(Settings(anonymized_telemetry=False))
collection_name = "finops_docs"

# Create or get collection
if collection_name not in [col.name for col in chroma_client.list_collections()]:
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

# Embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


# --------------- HELPER FUNCTIONS ---------------
def fetch_text_from_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/110.0.5481.177 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all('p')
            text = "\n".join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
            return text
        else:
            print(f"Error {response.status_code} fetching {url}")
            return ""
    except Exception as e:
        print(f"Exception fetching {url}: {str(e)}")
        return ""


def load_and_split_documents(urls, chunk_size=500):
    texts = []
    for url in urls:
        content = fetch_text_from_url(url)
        if content:
            for i in range(0, len(content), chunk_size):
                texts.append(content[i:i+chunk_size])
    return texts

def embed_texts(text_list):
    if not text_list:
        return []
    return embed_model.encode(text_list).tolist()

# --------------- INITIAL LOAD ---------------
if not collection.count():
    st.info("Loading and embedding documents for the first time...")
    all_chunks = load_and_split_documents(URLS)

    if not all_chunks:
        st.error("No valid text found in the URLs. Please check URLs or try different ones.")
        st.stop()

    embeddings = embed_texts(all_chunks)
    
    # Safety Check
    if not embeddings:
        st.error("Embedding failed. No content to embed.")
        st.stop()

    ids = [str(uuid.uuid4()) for _ in all_chunks]

    collection.add(documents=all_chunks, ids=ids, embeddings=embeddings)
    st.success("Documents loaded successfully!")

# --------------- STREAMLIT UI ---------------
st.title("FinOps Chat Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input("Ask your FinOps question:", key="user_input")

if query:
    query_embedding = embed_model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    if results['documents']:
        context = " ".join([doc for docs in results['documents'] for doc in docs])
        response = f"Based on documents: {context[:500]}..."
    else:
        response = "Sorry, no relevant information found."

    st.session_state.chat_history.append(("You", query))
    st.session_state.chat_history.append(("Bot", response))

# WhatsApp style chat
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"<div style='background-color:#DCF8C6;padding:10px;border-radius:10px;margin:10px 0;text-align:right'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='background-color:#ECECEC;padding:10px;border-radius:10px;margin:10px 0;text-align:left'><b>{sender}:</b> {message}</div>", unsafe_allow_html=True)
