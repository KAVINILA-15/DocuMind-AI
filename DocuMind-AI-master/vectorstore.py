import os
import shutil
from langchain_community.vectorstores import FAISS
from embeddings import get_embedding_model


def delete_vector_store(persist_directory="faiss_index"):
    """
    Permanently deletes the entire FAISS vector store directory,
    including all embedded chunks, index files, and metadata.
    Returns True if deleted, False if nothing existed.
    """
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        return True
    return False


def create_vector_store(
    chunks,
    embedding_method="huggingface",
    persist_directory="faiss_index",
    mode="replace",  # "replace" or "append"
):
    """
    Creates or updates a FAISS vector database from text chunks.

    mode:
      "replace" -> Deletes any existing index and builds a fresh one from the new chunks.
      "append"  -> Loads the existing index and merges the new chunks into it.
                   The existing embedding method must match the new one.

    embedding_method:
      "huggingface" -> BAAI/bge-small-en (local, free, open-source)
      "gemini"      -> Google text-embedding-004 (cloud, requires API key)
    """
    if not chunks:
        raise ValueError("Cannot create a vector store from empty chunks.")

    embeddings = get_embedding_model(method=embedding_method)

    if mode == "append" and os.path.exists(persist_directory):
        # Load existing index and merge new chunks into it
        existing = FAISS.load_local(
            persist_directory,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        new_store = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new_store)
        vector_store = existing
    else:
        # Replace mode: delete existing and build from scratch
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index to disk
    vector_store.save_local(persist_directory)

    # Persist the embedding method so it can be auto-detected on reload
    with open(os.path.join(persist_directory, "embedding_method.txt"), "w") as f:
        f.write(embedding_method)

    return vector_store


def load_vector_store(persist_directory="faiss_index", embedding_method=None):
    """
    Loads an existing FAISS vector database from the local directory.
    If embedding_method is not provided, it will be read from the saved metadata file.
    Returns (vector_store, embedding_method) or None if the directory does not exist.
    """
    if not os.path.exists(persist_directory):
        return None

    # Auto-detect the embedding method from metadata file if not explicitly provided
    if embedding_method is None:
        meta_file = os.path.join(persist_directory, "embedding_method.txt")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                embedding_method = f.read().strip()
        else:
            embedding_method = "huggingface"  # safe fallback

    embeddings = get_embedding_model(method=embedding_method)

    # allow_dangerous_deserialization=True is required to load FAISS indices locally
    vector_store = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store, embedding_method
