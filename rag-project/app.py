# ============================================================
# RAG SYSTEM - PHASE 2
# Upgrades: Persistent DB + Multi-doc + Streamlit UI
# ============================================================

import os
import streamlit as st
import chromadb
from google import genai
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import time
from dotenv import load_dotenv
load_dotenv()  

# ── CONFIG ───────────────────────────────────────────────────
DOCS_FOLDER   = "docs"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHROMA_PATH   = "./chroma_db"        # ← saved to disk now
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 75
TOP_K         = 5

# ── PAGE SETUP ───────────────────────────────────────────────
st.set_page_config(page_title="RAG Chat", page_icon="🧠", layout="wide")
st.title("🧠 RAG Chat — Ask Your Documents")
st.caption("Phase 2: Multi-doc + Persistent DB + Chat UI")

# ── LOAD MODELS (cached so they only load once) ───────────────
@st.cache_resource
def load_embedder():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# This spinner only shows while actually loading — disappears after
with st.spinner("Loading embedding model... (first time only)"):
    embedder = load_embedder()

@st.cache_resource
def load_gemini():
    return genai.Client(api_key=os.environ["GEMINI_API_KEY"])

embedder    = load_embedder()
gemini      = load_gemini()

# ── CHROMADB SETUP (persistent) ───────────────────────────────
@st.cache_resource
def load_chroma():
    # PersistentClient saves to disk at CHROMA_PATH
    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("rag_docs")
    return client, collection

chroma_client, collection = load_chroma()

# ── INDEX ALL PDFs IN docs/ FOLDER ───────────────────────────
def get_indexed_sources():
    """Returns set of filenames already indexed in ChromaDB."""
    existing = collection.get()
    sources = set()
    for meta in existing["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sources

def index_documents():
    """Reads all PDFs in docs/ and indexes any that aren't already stored."""
    docs_path      = Path(DOCS_FOLDER)
    pdf_files      = list(docs_path.glob("*.pdf"))
    indexed        = get_indexed_sources()
    new_files      = [f for f in pdf_files if f.name not in indexed]

    if not new_files:
        return 0, []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    total_chunks = 0
    added_files  = []

    for pdf_file in new_files:
        # Load PDF
        loader    = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        # Split into chunks
        chunks = splitter.split_documents(documents)

        # Embed
        texts      = [c.page_content for c in chunks]
        embeddings = embedder.embed_documents(texts)

        # Build unique IDs using filename + chunk number
        # (so IDs don't clash if you add more PDFs later)
        start_id = collection.count()
        ids      = [f"{pdf_file.name}_{start_id + i}" for i in range(len(chunks))]

        # Store with metadata (which file each chunk came from)
        metadatas = [{"source": pdf_file.name, "page": c.metadata.get("page", 0)}
                     for c in chunks]

        collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        total_chunks += len(chunks)
        added_files.append(pdf_file.name)

    return total_chunks, added_files

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Manager")

    # Show already indexed docs
    indexed_sources = get_indexed_sources()
    if indexed_sources:
        st.success(f"✅ {len(indexed_sources)} doc(s) indexed")
        for src in indexed_sources:
            st.write(f"• {src}")
    else:
        st.warning("No documents indexed yet")

    # Index new docs button
    if st.button("🔄 Index New Documents", use_container_width=True):
        with st.spinner("Indexing..."):
            count, files = index_documents()
        if files:
            st.success(f"Added {count} chunks from: {', '.join(files)}")
            st.rerun()
        else:
            st.info("No new PDFs found in docs/ folder")

    st.divider()
    st.caption(f"Total chunks in DB: {collection.count()}")

    # Clear DB button
    if st.button("🗑️ Clear Database", use_container_width=True):
        chroma_client.delete_collection("rag_docs")
        st.cache_resource.clear()
        st.rerun()



def ask(user_query: str, chat_history: list) -> tuple[str, list[str], str]:
    """Returns (answer, sources, standalone_question_used)."""

    # Build history string
    history_text = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    # Step 1: Embed the raw query first (no API call needed)
    # If there's history, append it to help the embedding
    search_query = user_query
    if history_text:
        search_query = f"{history_text}User: {user_query}"

    query_embedding = embedder.embed_query(search_query)

    # Step 2: Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"]
    )

    top_chunks    = results["documents"][0]
    top_metadatas = results["metadatas"][0]
    sources       = list({m["source"] for m in top_metadatas if m})
    context       = "\n\n---\n\n".join(top_chunks)

    # Step 3: ONE Gemini call that does both — understands context AND answers
    prompt = f"""You are a helpful assistant having a conversation.
Answer using ONLY the context below. If the answer is not in the context, say "I don't know based on the provided documents."

Context from documents:
{context}

Conversation so far:
{history_text}
User: {user_query}

Answer:"""

    # Retry up to 3 times if rate limited
    for attempt in range(3):
        try:
            response = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text, sources, search_query

        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 15 * (attempt + 1)  # wait 15s, then 30s
                st.warning(f"⏳ Rate limit hit — waiting {wait} seconds then retrying...")
                time.sleep(wait)
            else:
                raise e

# ── CHAT UI ───────────────────────────────────────────────────

# Initialize chat history in session state
# session_state persists across reruns within same browser session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

# Chat input box at bottom
# Chat input box at bottom
if prompt := st.chat_input("Ask something about your documents..."):

    # Check documents are indexed
    if collection.count() == 0:
        st.warning("⚠️ No documents indexed yet! Click 'Index New Documents' in the sidebar.")
        st.stop()

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get and show AI answer — NOTE: this block is INSIDE the if prompt: block
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources, standalone = ask(prompt, st.session_state.messages)
        st.markdown(answer)
        st.caption(f"📄 Sources: {', '.join(sources)}")
        with st.expander("🔍 Search query used"):
            st.caption(standalone)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })