import os
from google import genai
from google.genai import types

def _get_client():
    """Create and return a configured google-genai client forced to stable v1 API."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set. Please add it to your .env or sidebar.")
    return genai.Client(
        api_key=api_key,
        http_options={"api_version": "v1alpha"},  # Latest preview models depend on v1alpha
    )


def generate_answer(context, question, chat_history="", model="gemini-3.1-flash-lite-preview"):
    """
    Generates an answer using the selected Gemini model based STRICTLY on the given context.
    If the context does not contain the answer, returns 'I don't know'.

    model: gemini-3.1-flash-lite-preview (default) | gemini-2.5-flash-lite-preview | gemini-2.0-flash | gemini-2.0-flash-lite | gemini-1.5-flash
    """
    client = _get_client()

    prompt = f"""You are DocuMindAI, a strictly document-grounded AI assistant.
Your task is to answer the user's question relying ONLY on the provided context.

RULES:
1. If the answer cannot be found in the provided context, respond EXACTLY with "I don't know".
2. Do NOT use outside knowledge. Do NOT hallucinate.
3. Be concise and provide a clear, direct answer.
4. Do not include phrases like "based on the context provided" — just state the answer.

Context Documents:
{context}

Chat History (if any):
{chat_history}

User Question: {question}

Answer:"""

    try:
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()
    except Exception as e:
        raise  # Re-raise so app.py can show a user-friendly message


def generate_search_queries(question, chat_history=""):
    """
    Optional Multi-query retrieval: asks Gemini to rephrase the query
    into 3 optimized variants for better semantic retrieval coverage.
    """
    try:
        client = _get_client()
        prompt = f"""You are an AI assistant helping extract information from a document database.
Rephrase the user question into 3 different, optimized search queries for better document retrieval.
Return ONLY the 3 queries separated by newlines, with no bullets or extra text.

Chat History:
{chat_history}

User Question:
{question}
"""
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt,
        )
        queries = [q.strip() for q in response.text.strip().split("\n") if q.strip()]
        return queries[:3] if queries else [question]
    except Exception:
        return [question]
