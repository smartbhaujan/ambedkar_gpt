import os
import uuid
import fitz  # PyMuPDF
import joblib
import openai
import json
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
CITATION_AUTHOR_CONSTANT = "Babasaheb Ambedkar"
CITATION_COLLECTION_TITLE_CONSTANT = "in _Dr. Babasaheb Ambedkar: Writings and Speeches_, ed. Vasant Menon"
CITATION_PUBLICATION_DETAILS_CONSTANT = "(New Delhi: Dr. Ambedkar Foundation, Ministry of Social Justice & Empowerment, Govt. of India, 1979)"


def create_index():
    if index_name in pc.list_indexes().names():
        pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=EMBED_DIM,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    return pc.Index(index_name)


index = create_index()


def extract_chapter(text: str) -> str:
    lines = text.splitlines()
    for line in lines:
        if "chapter" in line.lower() or re.match(r'^\s*(CHAPTER\s+\w+|[IVXLCDM]+)\s*$', line, re.IGNORECASE):
            return line.strip()
        if line.isupper() and len(line.strip()) > 5:
            return line.strip()
    return "Unknown"


def split_text_with_metadata(text: str, max_tokens: int = 1000, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + max_tokens]
        chunk_text = " ".join(chunk_words).strip()
        chunks.append(chunk_text)
        if i + max_tokens >= len(words):
            break
        i += max_tokens - overlap
    return chunks


def upsert_all_pdfs(dataset_dir: str, metadata_json_path: str):
    with open(metadata_json_path, "r") as f:
        metadata_entries = json.load(f)

    all_chunks = []
    metadata_list = []

    for entry in metadata_entries:
        pdf_path = os.path.join(dataset_dir, entry["filename"])
        if not os.path.exists(pdf_path):
            print(f"⚠️ File not found: {pdf_path}")
            continue

        doc = fitz.open(pdf_path)
        total_chunks = 0
        page_to_add = int(entry.get("page_start_from", 0))
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            page_num_from_full_volume=page_to_add+page_num
            if not page_text:
                continue

            chapter = extract_chapter(page_text)
            chunks = split_text_with_metadata(page_text)
            for chunk in chunks:
                all_chunks.append(chunk)
                source_line = (
                    f"{CITATION_AUTHOR_CONSTANT}, "
                    f"{entry['chapter_name_manifest']} "
                    f"{CITATION_COLLECTION_TITLE_CONSTANT}, "
                    f"Vol.{entry['volume_number_manifest']}{CITATION_PUBLICATION_DETAILS_CONSTANT}, "
                    f"{page_num_from_full_volume}"
                )

                metadata_list.append({
                    "page": page_num_from_full_volume,
                    "chapter": chapter,
                    "text": chunk,
                    "source": source_line,
                    "volume": entry["volume_number_manifest"],
                    "filename": entry["filename"],
                    "chunk_index": total_chunks,
                    "citation_author_constant": CITATION_AUTHOR_CONSTANT,
                    "citation_collection_title_constant": CITATION_COLLECTION_TITLE_CONSTANT,
                    "citation_publication_details_constant": CITATION_PUBLICATION_DETAILS_CONSTANT,
                    "citation_source_filename": entry["filename"],
                    "citation_source_page": page_num_from_full_volume,
                    "citation_source_volume": entry["volume_number_manifest"],
                    "citation_source_chapter": entry["chapter_name_manifest"],
                    "citation_source_line": source_line
                })

                total_chunks += 1

        print(f"✅ Processed {entry['filename']} — {total_chunks} chunks.")

    # TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_chunks)
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    # Upsert
    for i, metadata in enumerate(metadata_list):
        try:
            dense_embedding = openai.Embedding.create(
                input=[metadata['text']],
                model=EMBED_MODEL,
                dimensions=EMBED_DIM
            )['data'][0]['embedding']

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
            print(f"❌ Error embedding chunk {i} from {metadata.get('filename')}: {e}")

    print(f"✅ Hybrid upsert complete for all PDFs. Total chunks: {len(metadata_list)}")


# ▶️ Run
if __name__ == "__main__":
    upsert_all_pdfs(dataset_dir="dataset", metadata_json_path="dataset/pdfs_metadata.json")
