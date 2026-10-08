import os
from io import BytesIO

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from google import genai


# -----------------------------
# Configuration
# -----------------------------
load_dotenv()

st.set_page_config(
    page_title="StudyAI - NEXUS Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

# -----------------------------
# Load embedding model
# -----------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# -----------------------------
# Session state
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Document extraction
# -----------------------------
def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(BytesIO(uploaded_file.getvalue()))

        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)

        return "\n".join(pages)

    elif uploaded_file.name.lower().endswith(".txt"):
        return uploaded_file.getvalue().decode("utf-8", errors="ignore")

    return ""


# -----------------------------
# Text chunking
# -----------------------------
def chunk_text(text, chunk_size=700, overlap=120):
    text = text.replace("\r", "\n")

    words = text.split()

    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": chunk,
                }
            )

        if end >= len(words):
            break

        start = end - overlap
        chunk_id += 1

    return chunks


# -----------------------------
# Build vector index
# -----------------------------
def build_index(documents):
    all_chunks = []

    for document_name, text in documents:
        chunks = chunk_text(text)

        for chunk in chunks:
            chunk["document"] = document_name
            all_chunks.append(chunk)

    if not all_chunks:
        return []

    texts = [chunk["text"] for chunk in all_chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    for chunk, embedding in zip(all_chunks, embeddings):
        chunk["embedding"] = embedding

    return all_chunks


# -----------------------------
# Semantic search
# -----------------------------
def search_chunks(query, chunks, top_k=5):
    if not chunks:
        return []

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    scores = []

    for chunk in chunks:
        score = float(np.dot(query_embedding, chunk["embedding"]))

        scores.append(
            {
                "document": chunk["document"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "score": score,
            }
        )

    scores.sort(key=lambda x: x["score"], reverse=True)

    return scores[:top_k]


# -----------------------------
# Generate grounded answer
# -----------------------------
def generate_answer(query, retrieved_chunks):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return (
            "Gemini API key is not configured. "
            "Add GEMINI_API_KEY to your Streamlit secrets."
        )

    client = genai.Client(api_key=api_key)

    context_parts = []

    for item in retrieved_chunks:
        context_parts.append(
            f"[Document: {item['document']} | "
            f"Chunk: {item['chunk_id']}]\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    instructions = """
You are StudyAI, a knowledge assistant for first-year students.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts.
3. If the provided context does not contain enough information,
   clearly say that the answer cannot be determined from the uploaded documents.
4. Give a clear, student-friendly explanation.
5. Keep the answer reasonably concise.
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=f"""
{instructions}

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}
""",
    )

    return response.text


# -----------------------------
# UI
# -----------------------------
st.title("📚 StudyAI")
st.subheader("Smart Document Knowledge Assistant")

st.write(
    "Upload your lecture notes and ask questions. "
    "StudyAI retrieves the most relevant document sections "
    "and generates an answer grounded in those sections."
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    process_button = st.button(
        "Process Documents",
        use_container_width=True,
    )

    if process_button:
        if not uploaded_files:
            st.warning("Please upload at least one PDF or TXT file.")
        else:
            documents = []

            with st.spinner("Reading and indexing documents..."):
                for file in uploaded_files:
                    text = extract_text(file)

                    if text.strip():
                        documents.append(
                            (
                                file.name,
                                text,
                            )
                        )

                st.session_state.chunks = build_index(documents)

            st.success(
                f"Processed {len(documents)} document(s) "
                f"with {len(st.session_state.chunks)} text chunks."
            )

    st.divider()

    st.write(
        f"**Indexed chunks:** "
        f"{len(st.session_state.chunks)}"
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Chat history
# -----------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Chat input
# -----------------------------
query = st.chat_input(
    "Ask a question about your uploaded documents..."
)


if query:

    if not st.session_state.chunks:
        st.warning(
            "Please upload and process a document first."
        )
        st.stop()

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve relevant chunks
    with st.spinner("Searching your documents..."):
        retrieved = search_chunks(
            query,
            st.session_state.chunks,
            top_k=5,
        )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Generating grounded answer..."):
            answer = generate_answer(
                query,
                retrieved,
            )

        st.markdown(answer)

        st.markdown("### 📌 Referenced document snippets")

        for item in retrieved:
            with st.expander(
                f"{item['document']} — "
                f"Chunk {item['chunk_id']} "
                f"(similarity: {item['score']:.3f})"
            ):
                st.write(item["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
