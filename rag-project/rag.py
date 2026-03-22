# ============================================================
# SIMPLE RAG SYSTEM - PHASE 1
# Uses: HuggingFace Embeddings (free) + ChromaDB + Gemini
# ============================================================

import os
import chromadb
from google import genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── CONFIG ──────────────────────────────────────────────────
PDF_PATH = "docs/sample.pdf"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 75
TOP_K = 5

# ── STEP 1: Load PDF ────────────────────────────────────────
print("📄 Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()
print(f"   Loaded {len(documents)} pages")

# ── STEP 2: Split into Chunks ────────────────────────────────
print("✂️  Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)
chunks = splitter.split_documents(documents)
print(f"   Created {len(chunks)} chunks")

# ── STEP 3: Load Embedding Model ─────────────────────────────
print("🔢 Loading embedding model (first time downloads ~90MB)...")
embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
print("   Embedding model ready!")

# ── STEP 4: Store in ChromaDB ────────────────────────────────
print("🗄️  Storing vectors in ChromaDB...")
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("rag_docs")

texts = [c.page_content for c in chunks]
embeddings = embedder.embed_documents(texts)
ids = [str(i) for i in range(len(chunks))]

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids
)
print(f"   Stored {len(chunks)} vectors in ChromaDB")

# ── STEP 5: Set up Gemini (new SDK) ──────────────────────────
print("🤖 Connecting to Gemini...")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
print("   Gemini ready!")

# ── STEP 6: The RAG Query Function ───────────────────────────
def ask(user_query: str) -> str:
    print(f"\n🔍 Searching for: '{user_query}'")

    query_embedding = embedder.embed_query(user_query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    top_chunks = results["documents"][0][:TOP_K]
    context = "\n\n---\n\n".join(top_chunks)

    print(f"   Found {len(top_chunks)} relevant chunks")

    prompt = f"""You are a helpful assistant. Answer ONLY using the context below.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {user_query}

Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ── STEP 7: Interactive Chat Loop ────────────────────────────
print("\n" + "="*50)
print("✅ RAG SYSTEM READY! Type your questions below.")
print("   Type 'quit' to exit")
print("="*50 + "\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    if not user_input:
        continue

    answer = ask(user_input)
    print(f"\nAI: {answer}\n")
    print("-"*50)