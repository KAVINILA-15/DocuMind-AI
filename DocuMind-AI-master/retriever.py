from vectorstore import load_vector_store


def get_retriever(strategy="similarity", k=3, persist_directory="faiss_index", embedding_method=None):
    """
    Returns a retriever initialized from the FAISS database.

    strategy:         "similarity" -> Cosine Similarity (default)
                      "mmr"        -> Max Marginal Relevance
    k:                Top K documents to retrieve (default 3)
    embedding_method: "huggingface" or "gemini" — loaded automatically if not specified
    """
    result = load_vector_store(persist_directory, embedding_method=embedding_method)
    if not result:
        raise Exception("FAISS index not found. Please upload and process documents first.")

    vector_store, used_embedding = result

    if strategy == "mmr":
        search_type = "mmr"
        search_kwargs = {"k": k, "fetch_k": 20, "lambda_mult": 0.5}
    else:
        search_type = "similarity"
        search_kwargs = {"k": k}

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    return retriever


def format_docs_for_citation(docs):
    """
    Utility to format retrieved documents into prompt context and UI citation list.
    """
    formatted_citations = []
    formatted_context = ""

    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")
        snippet = doc.page_content.replace("\n", " ").strip()

        # Build context string for the prompt
        formatted_context += f"[Document {i+1}] Source: {source}, Page: {page}\nContent: {snippet}\n\n"

        # Build citation list for the UI
        formatted_citations.append({
            "source": source,
            "page": page,
            "snippet": doc.page_content,
        })

    return formatted_context, formatted_citations
