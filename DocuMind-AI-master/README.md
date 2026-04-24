# 🤖 DocuMindAI — Strict Document QA Agent

A powerful **Retrieval-Augmented Generation (RAG)** application that answers questions **strictly from your uploaded PDF documents** — no hallucinations, with full source citations.

Built with **Streamlit**, **LangChain**, **FAISS**, and **Google Gemini**.

---

## ✨ Features

- 📄 Upload multiple research PDFs and build a searchable knowledge base
- 🧠 Ask questions — answers are grounded strictly in your documents
- 📚 Full source citations with document name and page number
- 🔢 Two embedding options:
  - 🟠 **HuggingFace** — `BAAI/bge-small-en` (runs locally, free, no API needed)
  - 🔵 **Google Gemini** — `gemini-embedding-001` (cloud, requires API key)
- ⚡ Two retrieval strategies:
- **Cosine Similarity**
- **MMR**
- Vector storage using FAISS
- LLM-based answer generation
- Top-3 supporting source citations
- No hallucination (strict QA mode)
- ➕ Append or replace document index on each upload
- 🗑️ Two-step deletion for safe index management

---

## 🗂️ Project Structure

DocuMindAI/
├── app.py              # Main Streamlit UI entry point
├── loader.py           # PDF loading with PyPDF
├── chunking.py         # Text splitting strategies (A & B)
├── embeddings.py       # HuggingFace & Gemini embedding wrappers
├── vectorstore.py      # FAISS index create / load / delete
├── retriever.py        # Retriever logic (similarity & MMR)
├── rag_pipeline.py     # Gemini LLM answer generation
├── requirements.txt    # Python dependencies
└── faiss_index/        # Auto-created after processing PDFs

---

## 🏗️ Architecture

- Upload PDF documents
- Extract and split text into chunks
- Generate embeddings
- Store vectors in FAISS
- Retrieve relevant chunks
- Generate answer using LLM
- Display answer with source references

---

## ⚙️ Tech Stack

- Python
- Streamlit
- FAISS
- HuggingFace Sentence Transformers
- Google Gemini API

---

## ⚙️ Prerequisites

- **Python 3.10 – 3.13** (Python 3.14 has limited package support, use 3.11 or 3.12 if possible)
- A **Google Gemini API Key** (free tier available) — get yours at [aistudio.google.com](https://aistudio.google.com/)
- Git (optional, for cloning)

---

## 🧪 Embedding Comparison

| Model                | Type                | Result     | Observation                       | Final Decision |
| -------------------- | ------------------- | ---------- | --------------------------------- | -------------- |
| BAAI/bge-small-en    | Open-source (local) | Stable     | Free, fast, no API limits         | Selected       |
| Gemini embedding-001 | Cloud (commercial)  | Limited    | Good quality but API quota issues | Not selected   |

---

## 🔍 Retrieval Strategy Comparison

| Strategy          | Result     | Observation                       | Final Decision |
| ----------------- | ---------- | --------------------------------- | -------------- |
| Cosine Similarity | Accurate   | Best for direct questions         | Selected       |
| MMR               | Diverse    | Good for variety but less precise | Secondary      |

---

## 📊 Testing & Evaluation

| Question                 | Result         | Observation             |
| ------------------------ | -------------- | ----------------------- |
| What is attention?       | Correct        | Retrieved from document |
| Summarize the document   | Correct        | Good coverage           |
| How does attention help? | Correct        | Matches source          |
| What is RAG?             | Not answered   | Not present in PDF      |
| Summarize experiments    | Correct        | Supported by sources    |

---

## 🧠 Key Feature (Strict QA Mode)

- This system does not hallucinate.
- If the answer is not found in the document, it returns:
- “I don't know”

---

## 🚀 Setup Instructions

### 1. Clone or Download the Project

```bash
git clone <your-repo-url>
cd DocuMindAI
```

Or download and extract the ZIP, then open a terminal inside the `DocuMindAI` folder.

---

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Activate it:**

- **Windows:**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

You should see `(venv)` appear in your terminal prompt.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏳ This may take a few minutes on first run — it will download the HuggingFace model weights (~130MB) the first time you use the HuggingFace embedding option.

Also install the Google GenAI SDK (required for Gemini generation + embeddings):

```bash
pip install google-genai langchain-core
```

---

### 4. Set Up Your API Key

Create a `.env` file in the project root:

```bash
# Windows
copy NUL .env

# Mac / Linux
touch .env
```

Open `.env` and add:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> 💡 You can also paste your API key directly into the sidebar inside the app — no `.env` file required.

---

### 5. Run the App

```bash
python -m streamlit run app.py
```

The app will open automatically in your browser at:
**http://localhost:8501**

---

## 📸 Demo Screenshot

<img width="1920" height="1080" alt="Screenshot (485)" src="https://github.com/user-attachments/assets/ca6347fd-3a04-49f0-bd1a-f87dc76903da" />


## 📖 How to Use

### Step 1 — Configure the Sidebar
1. **Enter your Google Gemini API Key** (if not set in `.env`)
2. **Choose a Generation Model** — `gemini-3.1-flash-lite-preview` is the default (fastest)
3. **Choose an Embedding Model:**
   - 🟠 HuggingFace — works offline, no API cost
   - 🔵 Google Gemini — cloud-based, requires API key

### Step 2 — Upload & Process PDFs
1. Click **Upload PDF Papers** and select one or more PDF files
2. Choose your **Chunking Strategy** (A = smaller chunks, B = larger context)
3. Choose your **Index Mode** (Replace = fresh start, Append = add to existing)
4. Click **⚡ Process Documents** — wait for the success message

### Step 3 — Ask Questions
- Type your question in the chat box at the bottom
- The assistant will answer using **only** the content from your uploaded documents
- Expand **📚 Top 3 Supporting Sources** to see which pages were cited

---

## 🔑 Model Reference

| Purpose | Model | Notes |
|---|---|---|
| Answer Generation | `gemini-3.1-flash-lite-preview` | Default — fastest, experimental |
| Answer Generation | `gemini-2.0-flash` | Stable, recommended backup |
| Answer Generation | `gemini-1.5-flash` | Highest free-tier quota |
| Embeddings (cloud) | `gemini-embedding-001` | Requires API key, v1beta endpoint |
| Embeddings (local) | `BAAI/bge-small-en` | Free, runs on CPU, no API needed |

---

## 🛠️ Troubleshooting

### `Fatal error in launcher` when running `streamlit run app.py`
Your Streamlit launcher may be broken. Use this instead:
```bash
python -m streamlit run app.py
```

### `404 NOT_FOUND` for Gemini embedding model
Make sure your API key is valid and has access to the Gemini API. The embedding client uses the `v1beta` endpoint which requires a valid Gemini API key.

### `RESOURCE_EXHAUSTED` / 429 Rate Limit Error
Your API free-tier quota is exhausted for the selected model. Switch to **`gemini-1.5-flash`** or **`gemini-2.0-flash-lite`** in the sidebar — they have separate quotas.

### HuggingFace model download is slow
The first run downloads ~130MB of model weights. Subsequent runs are fast as they are cached locally.

### `No index found` error when asking a question
You need to upload PDFs and click **⚡ Process Documents** before asking questions.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain`, `langchain-community` | RAG orchestration |
| `langchain-huggingface` | HuggingFace embeddings |
| `langchain-core` | Core LangChain abstractions |
| `faiss-cpu` | Local vector similarity search |
| `pypdf` | PDF text extraction |
| `python-dotenv` | Load `.env` API keys |
| `google-genai` | Google Gemini SDK (generation + embeddings) |
| `sentence-transformers` | HuggingFace model backend |

---

## 🔒 Privacy & Security

- Your PDFs are **processed locally** — text is extracted on your machine
- The **FAISS index** is saved locally in the `faiss_index/` folder
- **API calls** are made to Google's servers only for Gemini generation and Gemini embeddings
- Using **HuggingFace embeddings** keeps everything fully local (no external API calls for embedding)
- Never commit your `.env` file — it's already listed in `.gitignore`

---

## 📝 License

This project is for educational and research purposes.
