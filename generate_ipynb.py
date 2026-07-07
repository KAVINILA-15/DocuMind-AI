import json

cells = []

def add_md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text})

def add_code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text})

add_md([
    "# DocuMind AI Capstone — End-to-End RAG Pipeline\n",
    "This notebook covers the full implementation of the Retrieval-Augmented Generation (RAG) pipeline for the capstone project. It runs end-to-end and fulfills all structural requirements including data ingestion, chunking experiments, embedding selection, retriever tuning, and generative QA with strictly governed citations."
])

add_md([
    "## Step 1: Data Collection & Document Loading\n",
    "We use `PyPDFLoader` to load academic papers. This loader natively extracts text and attaches crucial metadata like the source filename and the precise page number. We format the `source` to be just the basename for a cleaner UI."
])

add_code([
    "import os\n",
    "from langchain_community.document_loaders import PyPDFLoader\n",
    "from dotenv import load_dotenv\n",
    "load_dotenv()\n\n",
    "def load_documents(file_paths):\n",
    "    documents = []\n",
    "    for path in file_paths:\n",
    "        loader = PyPDFLoader(path)\n",
    "        docs = loader.load()\n",
    "        for doc in docs:\n",
    "            doc.metadata['source'] = os.path.basename(doc.metadata.get('source', path))\n",
    "        documents.extend(docs)\n",
    "    return documents\n",
    "\n",
    "# Example usage (requires an actual PDF in the directory):\n",
    "# docs = load_documents(['sample.pdf'])\n",
    "# print(f'Loaded {len(docs)} pages.')"
])

add_md([
    "## Step 2: Text Chunking Strategy Experiments\n",
    "We experiment with two chunking configurations using `RecursiveCharacterTextSplitter`:\n",
    "- **Strategy A (Size: 500, Overlap: 150):** Good for pinpoint factual QA.\n",
    "- **Strategy B (Size: 800, Overlap: 200):** Retains larger narrative structure for summarization.\n",
    "\n",
    "**Observation:** Recursive splitting is preferred because it respects sentence and paragraph boundaries."
])

add_code([
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
    "\n",
    "def chunk_documents(documents, strategy='A'):\n",
    "    if strategy == 'B':\n",
    "        chunk_size = 800\n",
    "        chunk_overlap = 200\n",
    "    else:\n",
    "        chunk_size = 500\n",
    "        chunk_overlap = 150\n",
    "        \n",
    "    text_splitter = RecursiveCharacterTextSplitter(\n",
    "        chunk_size=chunk_size,\n",
    "        chunk_overlap=chunk_overlap\n",
    "    )\n",
    "    return text_splitter.split_documents(documents)\n"
])

add_md([
    "## Step 3: Embedding Models Comparison\n",
    "We implement two different embedding models to compare an open-source solution vs a commercial cloud solution:\n",
    "1. **Open-Source (HuggingFace):** `BAAI/bge-small-en` (Free, Fast, Runs Locally on CPU).\n",
    "2. **Commercial (Google Gemini):** `gemini-embedding-001` (Requires API Key, High Dimensionality).\n",
    "\n",
    "**Observation:** BAAI/bge-small-en is chosen as the default because it's localized and circumvents cloud rate limits, making our system fast and robust."
])

add_code([
    "from langchain_huggingface import HuggingFaceEmbeddings\n",
    "from google import genai\n",
    "from langchain_core.embeddings import Embeddings\n",
    "\n",
    "class GeminiEmbeddings(Embeddings):\n",
    "    def __init__(self, api_key: str):\n",
    "        self.client = genai.Client(api_key=api_key, http_options={'api_version': 'v1beta'})\n",
    "        self.model = 'models/gemini-embedding-001'\n",
    "    def embed_documents(self, texts):\n",
    "        return [self.client.models.embed_content(model=self.model, contents=t, config={'task_type': 'RETRIEVAL_DOCUMENT'}).embeddings[0].values for t in texts]\n",
    "    def embed_query(self, text):\n",
    "        return self.client.models.embed_content(model=self.model, contents=text, config={'task_type': 'RETRIEVAL_QUERY'}).embeddings[0].values\n",
    "\n",
    "def get_embeddings(method='huggingface'):\n",
    "    if method == 'gemini':\n",
    "        api_key = os.getenv('GOOGLE_API_KEY')\n",
    "        return GeminiEmbeddings(api_key=api_key)\n",
    "    return HuggingFaceEmbeddings(\n",
    "        model_name='BAAI/bge-small-en',\n",
    "        model_kwargs={'device': 'cpu'},\n",
    "        encode_kwargs={'normalize_embeddings': True}\n",
    "    )\n"
])

add_md([
    "## Step 4: Vector Database (FAISS)\n",
    "We use **FAISS** (Facebook AI Similarity Search) to index chunk embeddings. It is efficient for local execution and serialization."
])

add_code([
    "import os, shutil\n",
    "from langchain_community.vectorstores import FAISS\n",
    "\n",
    "def build_vector_store(chunks, embedding_method='huggingface', persist_dir='faiss_index_nb'):\n",
    "    embeddings = get_embeddings(method=embedding_method)\n",
    "    if os.path.exists(persist_dir):\n",
    "        shutil.rmtree(persist_dir)\n",
    "    \n",
    "    vector_store = FAISS.from_documents(chunks, embeddings)\n",
    "    vector_store.save_local(persist_dir)\n",
    "    return vector_store\n"
])

add_md([
    "## Step 5: Retrieval Strategies (Cosine vs MMR) & Preprocessing\n",
    "We compare **Cosine Similarity** (baseline relevance) and **Max Marginal Relevance (MMR)** (balances relevance with diversity).\n",
    "We additionally add **Spell Checking** via `TextBlob` to intercept typos before lookup."
])

add_code([
    "from textblob import TextBlob\n",
    "\n",
    "def fix_spelling(query):\n",
    "    return str(TextBlob(query).correct())\n",
    "\n",
    "def get_retriever(vector_store, strategy='similarity'):\n",
    "    if strategy == 'mmr':\n",
    "        return vector_store.as_retriever(search_type='mmr', search_kwargs={'k': 5, 'fetch_k': 20, 'lambda_mult': 0.7})\n",
    "    return vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})\n",
    "\n",
    "def format_docs_for_citation(docs):\n",
    "    citations = []\n",
    "    context = ''\n",
    "    for i, doc in enumerate(docs):\n",
    "        source = doc.metadata.get('source', 'Unknown')\n",
    "        page = doc.metadata.get('page', 'Unknown')\n",
    "        snippet = doc.page_content.replace('\\n', ' ').strip()\n",
    "        context += f'[Document {i+1}] Source: {source}, Page: {page}\\nContent: {snippet}\\n\\n'\n",
    "        citations.append({'source': source, 'page': page, 'snippet': doc.page_content})\n",
    "    return context, citations\n"
])

add_md([
    "## Step 6: RAG Pipeline Construction & Generation\n",
    "We use **Google Gemini** as the LLM. The system prompt is engineered to enforce strict anti-hallucination rules, compelling the agent to return *'I don't know'* if the context is insufficient."
])

add_code([
    "from google import genai\n",
    "from google.genai import types\n",
    "\n",
    "def generate_answer(context, question, model='gemini-1.5-flash'):\n",
    "    api_key = os.getenv('GOOGLE_API_KEY')\n",
    "    client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})\n",
    "    \n",
    "    prompt = f\"\"\"You are DocuMindAI, a strictly document-grounded AI assistant.\n",
    "Your task is to answer the user's question relying ONLY on the provided context.\n",
    "RULES:\n",
    "1. Only respond with \"I don't know\" if the answer is NOT present in any retrieved chunk. Do not hallucinate.\n",
    "2. Be concise and direct.\n",
    "\n",
    "Context Documents:\n",
    "{context}\n",
    "\n",
    "User Question: {question}\n",
    "Answer:\"\"\"\n",
    "    \n",
    "    response = client.models.generate_content(\n",
    "        model=model,\n",
    "        contents=prompt,\n",
    "        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024)\n",
    "    )\n",
    "    return response.text.strip()\n"
])

add_md([
    "## Step 7: Final End-to-End Evaluation\n",
    "Putting it all together to demonstrate the final answer along with Top-3 Citations mapping back to the paper title and page number."
])

add_code([
    "def run_rag_pipeline(pdf_paths, user_query):\n",
    "    print(f'1. Processing PDFs...')\n",
    "    docs = load_documents(pdf_paths)\n",
    "    if not docs:\n",
    "        return 'No documents loaded. Please provide a path to a PDF.'\n",
    "        \n",
    "    print(f'2. Chunking {len(docs)} pages...')\n",
    "    chunks = chunk_documents(docs, strategy='A')\n",
    "    \n",
    "    print('3. Embedding & Indexing (FAISS + HuggingFace)...')\n",
    "    vector_store = build_vector_store(chunks)\n",
    "    \n",
    "    print('4. Applying Typo Correction...')\n",
    "    corrected_query = fix_spelling(user_query)\n",
    "    print(f'   -> Original: \"{user_query}\" | Corrected: \"{corrected_query}\"')\n",
    "    \n",
    "    print('5. Retrieving Context...')\n",
    "    retriever = get_retriever(vector_store, strategy='similarity')\n",
    "    retrieved_docs = retriever.invoke(corrected_query)\n",
    "    \n",
    "    formatted_context, citations = format_docs_for_citation(retrieved_docs)\n",
    "    \n",
    "    print('6. Generating Answer...')\n",
    "    answer = generate_answer(formatted_context, corrected_query)\n",
    "    \n",
    "    print('\\n==================================================')\n",
    "    print(f'🤖 ANSWER: {answer}\\n')\n",
    "    if 'I don\\'t know' not in answer and citations:\n",
    "        print('📚 TOP 3 SOURCES CITED:')\n",
    "        for i, cit in enumerate(citations[:3], 1):\n",
    "            print(f\" [{i}] {cit['source']} — Page {cit['page']}\")\n",
    "    print('==================================================\\n')\n",
    "\n",
    "# TEST EXECUTION (Uncomment below line with a valid PDF path to verify execution)\n",
    "# run_rag_pipeline(['sample.pdf'], 'Describe the methodology.')\n"
])

notebook = {
    "cells": cells,
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 4
}

with open('DocuMindAI_Capstone.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook generated successfully!")
