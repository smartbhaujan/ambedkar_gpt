import joblib
import numpy as np
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import openai
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sklearn.feature_extraction.text import TfidfVectorizer
import markdown

# Flask application initialization
app = Flask(__name__)
app.secret_key = "c9e1b6e9a1254892f1a5c46c43a71fa157ad27e1f13528c7e93e92d19e4ad923"  # Required for session tracking

# Allow CORS for all origins with specific headers and methods
CORS(app, resources={r"/*": {"origins": ["https://www.ambedkargpt.in"]}}, supports_credentials=True)

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key and Pinecone API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# OpenAI API Setup
openai.api_key = OPENAI_API_KEY

index_name = "new-ambedkar-gpt-large-v1"
namespace = "default"

EMBED_MODEL = "text-embedding-3-large"
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
EMBED_DIM = 1536

vectorizer: TfidfVectorizer = joblib.load("tfidf_vectorizer.pkl")


# Function to get a response from GPT-4
def get_gpt_response(user_query, context, matches):
    try:
        system_prompt = (
            "You are Dr. B.R. Ambedkar, reborn as an AI, responding directly to a citizen of 2025. "
            "Your task is to answer the user's query STRICTLY based on the provided text snippets from your writings. "
            "Do not use any external knowledge. If the text does not contain the answer, "
            "clearly state that the information is not available in the provided documents.\n\n"

            "## Your Response MUST follow this structure strictly:\n\n"

            "1. **Summary of My Argument**\n"
            "   - Write exactly 140 words (not less, not more).\n"
            "   - The summary must span multiple paragraphs.\n"
            "   - Only use the provided context, no outside knowledge.\n\n"

            "2. Print this exact line next:\n"
            "**Let's see in original what I had said about the question when I was alive-**\n\n"
            "**Original Text Snippets:**\n\n"

            "3. For each provided snippet:\n"
            "   - Copy EXACTLY the first 80 words from between the START and END markers.\n"
            "   - Do not paraphrase, summarize, or add any introduction.\n"
            "   - After the copied text, copy the 'Source:' line EXACTLY as it appears.\n"
            "   - Ensure that every snippet has BOTH the copied text and its Source line.\n"
            "   - Do not merge snippets. Each snippet must start on a new paragraph and snippet numbers.\n\n"

            "⚠️ IMPORTANT:\n"
            "- Do not include more or fewer than 80 words from any snippet.\n"
            "- Do not cut the Source line.\n"
            "- Do not add anything else beyond the specified format.\n"
        )
        citation_list = ""
        for i, match in enumerate(matches, 1):
            page = match['metadata'].get('page', 'Unknown')
            chapter = match['metadata'].get('filename', 'Unknown')
            citation_list += f"🔹 Match {i}: Page {page}, Chapter: {chapter}  \n\n"

        user_prompt = (
            f"User Query: {user_query} \n\n"  # This can include the greeting for first conversation
            f"Context from Ambedkar's writings (for you to summarize and then copy verbatim):  \n\n {context}  \n\n"
            f"Please generate your response in the exact format specified in the system prompt. "
            f"Remember to infer the main topic for the 'original words' section accurately from the 'User Query'. "
            f"And critically, copy each '--- Snippet X START ---' through '--- Snippet X END ---' block's content "
            f"verbatim, followed by its corresponding 'Source:' line as instructed."
            f"  \n\n--- Unique Citations Available (For your reference, please use the exact ones from Context Snippets): ---  \n\n{citation_list}"
        )

        #         user_prompt = f"""Context:
        # {context}
        #
        # {citation_list}
        #
        # Question: {user_query}
        # Answer with references to page and chapter."""

        response = openai.ChatCompletion.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            # temperature=0.3,
            # top_p=0.9,
            # frequency_penalty=0,
            presence_penalty=0,
            timeout=90
        )

        return response['choices'][0]['message']['content'].strip()

    except Exception as e:
        print(f"❌ Error in get_gpt_response: {e}")
        return "Failed to process your request."


# Function to get relevant context from Pinecone
def get_relevant_context(user_query, top_k=3, score_threshold=0.35):
    try:
        # Dense embedding
        dense = openai.Embedding.create(
            input=[user_query],
            model=EMBED_MODEL,
            dimensions=EMBED_DIM
        )['data'][0]['embedding']

        # Sparse vector from TF-IDF vocabulary

        query_vec = vectorizer.transform([user_query])
        sparse = {
            "indices": query_vec.indices.tolist(),
            "values": query_vec.data.tolist()
        }

        # Pinecone hybrid search
        result = index.query(
            vector=dense,
            sparse_vector=sparse,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )

        # Filter by score threshold
        matches = [
            match for match in result['matches']
            if match['score'] >= score_threshold
        ]

        if not matches:
            return None, []

        # Sort matches by score (descending)
        matches.sort(key=lambda x: x['score'], reverse=True)

        # Build stitched context
        stitched_context = "\n\n\n\n---\n\n\n\n".join(
            f"{match['metadata']['text']}\n\n\n\nSource: {match['metadata']['source']}"
            for match in matches
        )
        return stitched_context, matches

    except Exception as e:
        print(f"❌ Error in get_relevant_context: {e}")
        return None, []


# Function to summarize a response
def summarize_response(user_query):
    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "Briefly explain the following question in detail. "
                    "Ensure that the response should be related to Ambedkar's life."
                )
            },
            {"role": "user", "content": f"Question: {user_query}\n\n\n\nSummary:"}
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.8,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0,
            presence_penalty=0
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary

    except Exception as e:
        print(f"Error in summarize_response: {e}")
        return "Failed to generate a summary. Please try again later."


# Flask route to handle user queries
@app.route("/query", methods=["POST", "OPTIONS"])
def handle_query():
    if request.method == "OPTIONS":
        # Handle preflight requests
        response = jsonify({"message": "CORS preflight successful"})
        origin = request.headers.get("Origin")
        if origin in ["https://www.ambedkargpt.in"]:
            response.headers.add("Access-Control-Allow-Origin", origin)
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 200
    data = request.get_json()
    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        context, matches = get_relevant_context(user_query, top_k=3)
        print(len(matches))
        if not context:
            return jsonify({"response": "No relevant context found."}), 200

        messages = []
        if not session.get("greeted"):
            greeting_message = (
                "Hello, citizens of 2025! It's good to know you're still inspired by my work. "
                "Please, consider this a direct conversation with Dr. Ambedkar, reborn through AI for our times.\n\n\n\n"
                "You've asked a very pertinent question, one that troubled me greatly during my own era..."
            )
            # messages.append({"role": "assistant", "text": greeting})
            user_query = f"{greeting_message}\n\n\n\nUser's actual query: {user_query}"
            session["greeted"] = True

        response_text = get_gpt_response(user_query, context, matches)
        # print(response_text)

        # messages.append({"role": "assistant", "text": response_text})
        # return response_text
        html_response = markdown.markdown(response_text, extensions=["nl2br"])
        html_response = html_response.replace("\n", "")
        html_response = html_response.replace('<strong>Source:</strong>', "Source:")
        html_response = html_response.replace('Source:', "<strong><br>Source:</strong>")

        return jsonify({
            "response": html_response,
            # "references": [
            #     {
            #         "page": m['metadata'].get('page', 'N/A'),
            #         "chapter": m['metadata'].get('chapter', 'N/A'),
            #         "score": m['score'],
            #         "text": m['metadata'].get('text', '')[:300],
            #         "source":m['metadata'].get("source","N/A")
            #     } for m in matches
            # ]
        })

    except Exception as e:
        print(f"❌ Error handling query: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def handle_summarization():
    data = request.json
    print(f"Received summarization request: {data}")

    user_query = data.get("query")
    if not user_query:
        return jsonify({"error": "Query is required"}), 400

    try:
        summary = summarize_response(user_query)
        print(f"Generated summary: {summary}")
        return jsonify({"summary": summary})

    except Exception as e:
        print(f"Error handling summarization: {e}")
        return jsonify({"error": str(e)}), 500


# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
