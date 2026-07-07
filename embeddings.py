import os
import time
import re
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from google import genai
from typing import List, Union

# Free tier limit: 100 requests/min → 1 req per 0.6s = ~100/min.
# We use 0.7s to stay safely under the cap.
_INTER_REQUEST_DELAY = 0.7   # seconds between each embed_content call
_MAX_RETRIES = 3             # retries on 429 before re-raising


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

    def _embed_with_retry(self, contents: Union[str, List[str]], task_type: str) -> Union[List[float], List[List[float]]]:
        """
        Embed call with exponential-backoff retry on 429.
        Supports both single text (str) and batch (List[str]).
        """
        delay = _INTER_REQUEST_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=contents,
                    config={"task_type": task_type},
                )
                
                # If single string was passed, return single embedding
                if isinstance(contents, str):
                    return result.embeddings[0].values
                
                # If list was passed, return list of embeddings
                return [e.values for e in result.embeddings]
                
            except Exception as e:
                err = str(e)
                is_quota = "429" in err or "RESOURCE_EXHAUSTED" in err
                if is_quota and attempt < _MAX_RETRIES:
                    match = re.search(r"retryDelay.*?(\d+)s", err)
                    wait = int(match.group(1)) + 2 if match else delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents in batches to significantly speed up processing.
        Gemini supports up to 100 items per batch; we use 50 for safety.
        """
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_with_retry(batch, "RETRIEVAL_DOCUMENT")
            all_embeddings.extend(batch_embeddings)
            
            # Brief throttle between batches to respect free-tier 100 RPM
            if i + batch_size < len(texts):
                time.sleep(_INTER_REQUEST_DELAY)
                
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string (task: RETRIEVAL_QUERY)."""
        return self._embed_with_retry(text, "RETRIEVAL_QUERY")



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
