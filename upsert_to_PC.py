import os
import uuid
import fitz  # PyMuPDF
import joblib
import openai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "new-ambedkar-gpt-large-v1"
namespace = "default"

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1536


def create_index():
    pc = Pinecone(api_key='pcsk_4hjDw2_U46hXKECsg93cXYmSbfcLDVkAsE48UxdgiYtLUiAgV2VDpNgm3nquLjrxu8e8P7')

    # Delete the old one if exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create new index with correct dimensions
    pc.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="dotproduct",  # or "euclidean"
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return pc.Index(index_name)


index = create_index()


# ✂️ Chunking with overlap
def split_text(text: str, max_tokens: int = 1000, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words)
        chunks.append(chunk)
        i += max_tokens - overlap
    return chunks


def split_text_with_metadata(text: str, page_num: int, chapter: str, max_tokens: int = 1000, overlap: int = 50) -> List[
    str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_tokens]
        chunk_text = " ".join(chunk_words).strip()

        # Prepend metadata for search context
        # print("ca",chapter)
        # chunk_with_header = f"{chapter}\n\n{chunk_text}"
        print("ch",chunk_text)
        print("---"*8)
        chunks.append(chunk_text)

        if i + max_tokens >= len(words):
            break
        i += max_tokens - overlap

    return chunks


# 📚 Try to extract chapter from text
def extract_chapter(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        if "chapter" in line.lower() or re.match(r'^\s*(CHAPTER\s+\w+|[IVXLCDM]+)\s*$', line, re.IGNORECASE):
            return line.strip()
        if line.isupper() and len(line.strip()) > 5:
            return line.strip()
    return "Unknown"


# 🧠 Embed + Upsert
def upsert_pdf_to_pinecone(pdf_path: str):
    doc = fitz.open(pdf_path)
    total_chunks = 0
    corpus = []  # For TF-IDF
    metadata_list = []

    # First pass: collect chunks for TF-IDF
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text().strip()
        if not page_text:
            continue

        chapter_title = extract_chapter(page_text)
        chunks = split_text_with_metadata(page_text, page_num, chapter_title)

        for chunk in chunks:
            corpus.append(chunk)
            metadata_list.append({
                "page": page_num,
                "chapter": chapter_title,
                "text": chunk,
                "source": "Annihilation of Caste",
                "chunk_index": total_chunks
            })
            total_chunks += 1

    # Compute sparse vectors using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    tfidf_vocabulary = vectorizer.vocabulary_

    # Now embed + upsert
    for i, metadata in enumerate(metadata_list):
        try:
            dense_embedding = openai.Embedding.create(
                input=[metadata['text']],
                model=EMBED_MODEL,
                dimensions=EMBED_DIM
            )['data'][0]['embedding']

            # Sparse vector: dict of index:value
            sparse_vector = {
                "indices": tfidf_matrix[i].indices.tolist(),
                "values": tfidf_matrix[i].data.tolist()
            }

            vector_id = str(uuid.uuid4())
            index.upsert(
                vectors=[{
                    "id": vector_id,
                    "values": dense_embedding,
                    "sparse_values": sparse_vector,
                    "metadata": metadata
                }],
                namespace=namespace
            )

        except Exception as e:
            print(f"❌ Error embedding chunk {i}: {e}")

    print(f"✅ Hybrid upsert complete. Total chunks uploaded: {total_chunks}")


# ▶️ Run this
if __name__ == "__main__":
    upsert_pdf_to_pinecone("Annihilation of caste.pdf")
