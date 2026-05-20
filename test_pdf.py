import fitz
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import ollama

# -----------------------------
# 1. EXTRACT & CLEAN TEXT
# -----------------------------
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    # Clean text
    text = re.sub(r'\s+', ' ', text)

    # Remove author/header junk (start from Abstract)
    parts = re.split("Abstract", text, flags=re.IGNORECASE)
    text = parts[-1] if len(parts) > 1 else text

    # Remove references section
    text = re.split("References", text, flags=re.IGNORECASE)[0]

    return text


# -----------------------------
# 2. CHUNK TEXT
# -----------------------------
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


# -----------------------------
# 3. CREATE EMBEDDINGS
# -----------------------------
def create_embeddings(chunks):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings, model


# -----------------------------
# 4. CREATE FAISS INDEX
# -----------------------------
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


# -----------------------------
# 5. IMPROVED SEARCH
# -----------------------------
def search(query, model, index, chunks, k=5):
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k)

    results = []
    for i in indices[0]:
        chunk = chunks[i]

        # Filter out very small / useless chunks
        if len(chunk) > 100:
            results.append(chunk)

    return results[:3]


# -----------------------------
# 6. GENERATE ANSWER (OLLAMA)
# -----------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a research paper assistant.

Instructions:
- Answer ONLY using the given context
- Use simple language
- Keep answer short (5-7 lines)
- If answer is not found, say "Not found in paper"
- Do NOT make assumptions

Context:
{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="llama3",   # or "mistral"
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]
def summarize_paper(text):
    # limit text (LLM constraint)
    short_text = text[:3000]

    prompt = f"""
You are a research assistant.

Summarize the paper in simple language using:

1. Problem
2. Method
3. Results
4. Conclusion

Text:
{short_text}
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

# -----------------------------
# 7. MAIN PROGRAM
# -----------------------------
if __name__ == "__main__":

    pdf_path = "paper.pdf"

    print("\n📄 Extracting and cleaning text...")
    text = extract_text(pdf_path)

    print("✂️ Chunking text...")
    chunks = chunk_text(text)
    print("Total chunks:", len(chunks))

    print("\n🧠 Creating embeddings...")
    embeddings, model = create_embeddings(chunks)

    print("📦 Creating FAISS index...")
    index = create_faiss_index(embeddings)

    print("\n🤖 AI Assistant Ready!\n")

    while True:
        query = input("👉 Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        # Retrieve relevant chunks
        retrieved_chunks = search(query, model, index, chunks)

        print("\n📌 Retrieved Context (debug):\n")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"--- Chunk {i+1} ---\n{chunk[:300]}\n")

        # Generate answer
        answer = generate_answer(query, retrieved_chunks)

        print("\n🤖 Answer:\n")
        print(answer)
        print("\n" + "="*50)