import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from google import genai
from typing import List


class GeminiEmbeddings(Embeddings):
    """
    Custom LangChain-compatible wrapper around the new google-genai SDK
    for the gemini-embedding-001 model.
    Uses v1beta API endpoint — gemini-embedding-001 is not available on v1 stable.
    """

    def __init__(self, api_key: str):
        # v1beta is required — gemini-embedding-001 is NOT available on v1 stable
        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1beta"},
        )
        self.model = "models/gemini-embedding-001"  # Confirmed via ListModels

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents for indexing (task: RETRIEVAL_DOCUMENT)."""
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model=self.model,
                contents=text,
                config={"task_type": "RETRIEVAL_DOCUMENT"},
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (task: RETRIEVAL_QUERY)."""
        result = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config={"task_type": "RETRIEVAL_QUERY"},
        )
        return result.embeddings[0].values


def get_huggingface_embeddings():
    """
    Returns the HuggingFace open-source embedding model: BAAI/bge-small-en.
    - Free to use, runs locally on CPU.
    - normalize_embeddings=True ensures compatibility with cosine similarity.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_gemini_embeddings():
    """
    Returns the Gemini embedding model: gemini-embedding-001.
    - Requires GOOGLE_API_KEY in environment.
    - Uses task_type to distinguish document embedding vs query embedding.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Set it in your .env file or sidebar to use Gemini embeddings."
        )
    return GeminiEmbeddings(api_key=api_key)


def get_embedding_model(method="huggingface"):
    """
    Factory function to return the correct embedding model.

    method: "huggingface" -> BAAI/bge-small-en (local, free, open-source)
            "gemini"      -> Google gemini-embedding-001 (cloud, requires API key)
    """
    if method == "gemini":
        return get_gemini_embeddings()
    return get_huggingface_embeddings()
