import os
import uuid
import fitz  # PyMuPDF
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from typing import List, Dict
import re
# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = "ambedkar-gpt-gemini"

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)


def create_index():
    index_name = PINECONE_INDEX
    pc = Pinecone(api_key='pcsk_4hjDw2_U46hXKECsg93cXYmSbfcLDVkAsE48UxdgiYtLUiAgV2VDpNgm3nquLjrxu8e8P7')

    # Delete the old one if exists
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    # Create new index with correct dimensions
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",  # or "euclidean"
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    return pc.Index(index_name)



# Split text into word chunks with overlap
def split_text(text, max_words=250, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
        i += max_words - overlap
    return chunks


# Extract chapter using a regex pattern
def extract_chapter(text):
    match = re.search(r"(CHAPTER\s+\w+|X{0,3}(IX|IV|V?I{0,3})|[A-Z\s]{5,})", text[:300])
    return match.group(0).strip() if match else "Unknown"


# Embed text using Gemini embedding model
def get_gemini_embedding(text, task_type="retrieval_document"):
    # model = genai.EmbeddingModel(model_name="models/embedding-001")
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type=task_type,
        title="Ambedkar PDF Chunk"
    )
    print(response)
    return response['embedding']


# Upsert a single chunk to Pinecone
def upsert_chunk(chunk, page, chapter):
    # Step 1: Get or create Pinecone index
    index = create_index()  # This should return pc.Index object

    # Step 2: Get Gemini embedding
    embedding = get_gemini_embedding(chunk)

    # ✅ Step 3: Flatten the embedding safely

    # print("Raw embedding:", raw_embedding)
    print("Type:", type(embedding))
    # Optional debug
    # print("✅ Embedding length:", len(embedding))
    assert isinstance(embedding[0], float), "❌ Embedding is not a flat list!"
    vector_id = str(uuid.uuid4())

    index.upsert(
        vectors=[{
            "id": "vector_id",
            "values": "embedding",
            "metadata": {
                "text": "chunk",
                "page": "page_num",
                "chapter": "chapter_title",
                "source": "Annihilation of Caste",
                "chunk_index": "total_chunks"
            }
        }],
        namespace="namespace"
    )
    stats = index.describe_index_stats()
    print(stats)


# Process entire PDF
def upsert_pdf_to_pinecone(pdf_path):
    doc = fitz.open(pdf_path)
    total_chunks = 0

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        chapter = extract_chapter(text)
        chunks = split_text(text)

        for chunk in chunks:
            # try:
            upsert_chunk(chunk, page_num, chapter)
            total_chunks += 1
            print(f"Processing upsert chunk on page {page_num} {total_chunks}")
        # except Exception as e:
        #     print(f"❌ Failed to upsert chunk on page {page_num}: {e}")

    print(f"✅ Upsert complete. Total chunks: {total_chunks}")


# Example usage
if __name__ == "__main__":
    upsert_pdf_to_pinecone("Annihilation of caste.pdf")
