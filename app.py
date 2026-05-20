import streamlit as st
import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Research Dashboard", layout="wide")

# -----------------------------
# CUSTOM CSS (DASHBOARD STYLE)
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.big-title {
    font-size: 40px;
    font-weight: bold;
    color: #22c55e;
}
.card {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
.answer-box {
    background: #111827;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid #22c55e;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# FUNCTIONS (same backend)
# -----------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    text = re.sub(r'\s+', ' ', text)
    parts = re.split("Abstract", text, flags=re.IGNORECASE)
    text = parts[-1] if len(parts) > 1 else text
    text = re.split("References", text, flags=re.IGNORECASE)[0]

    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model


def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


def search(query, model, index, chunks):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), 5)

    results = []
    for i in indices[0]:
        if len(chunks[i]) > 100:
            results.append(chunks[i])

    return results[:3]


def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are an AI research assistant.

- Answer clearly
- Use simple language
- Use only the given context

Context:
{context}

Question:
{query}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def summarize_paper(text):
    short_text = text[:3000]

    prompt = f"""
Summarize this research paper:
- Problem
- Method
- Results
- Conclusion

Text:
{short_text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# -----------------------------
# HEADER
# -----------------------------
st.markdown('<div class="big-title">🧠 AI Research Dashboard</div>', unsafe_allow_html=True)
st.caption("Analyze and understand research papers instantly")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")

# -----------------------------
# SESSION STATE
# -----------------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

# -----------------------------
# PROCESS FILE
# -----------------------------
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    text = extract_text("temp.pdf")
    chunks = chunk_text(text)
    embeddings, model = create_embeddings(chunks)
    index = create_faiss_index(embeddings)

    st.session_state.text = text
    st.session_state.chunks = chunks
    st.session_state.model = model
    st.session_state.index = index
    st.session_state.processed = True

# -----------------------------
# DASHBOARD METRICS
# -----------------------------
if st.session_state.processed:

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">📄 Chunks<br><b>{}</b></div>'.format(len(st.session_state.chunks)), unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">🧠 Model<br><b>MiniLM</b></div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">⚡ Status<br><b>Ready</b></div>', unsafe_allow_html=True)

# -----------------------------
# MAIN SECTION
# -----------------------------
col1, col2 = st.columns([2,1])

# LEFT: Q&A
with col1:
    st.subheader("💬 Ask Questions")

    if st.session_state.processed:
        query = st.text_input("Type your question here")

        if query:
            results = search(
                query,
                st.session_state.model,
                st.session_state.index,
                st.session_state.chunks
            )

            answer = generate_answer(query, results)

            st.markdown('<div class="answer-box">{}</div>'.format(answer), unsafe_allow_html=True)
    else:
        st.warning("Upload a PDF first")

# RIGHT: SUMMARY PANEL
with col2:
    st.subheader("📌 Paper Summary")

    if st.session_state.processed:
        if st.button("Generate Summary"):
            summary = summarize_paper(st.session_state.text)
            st.write(summary)
    else:
        st.info("Waiting for PDF...")
