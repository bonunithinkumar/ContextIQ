# 📚 DEEP NOTES — RAG Practice Project
### Complete Explanation: Every Concept, Every Line, Every Tool

---

## 🌟 WHAT IS THIS PROJECT? (Big Picture — Start Here)

Imagine you have a huge stack of books, PDFs, and text files on your desk. You want to ask a question like **"What is Naive Bayes?"** and get a smart, meaningful answer from the content of those books — not from the internet, not from ChatGPT's general knowledge, but from **your own documents**.

That's exactly what this project does.

This is a **RAG System** — **Retrieval-Augmented Generation**. It is one of the most powerful and popular AI architectures in the world right now.

In plain English:
- **Retrieval** = Finding the most relevant paragraphs from your documents for a given question
- **Augmented** = Adding those paragraphs as context/hints to the AI
- **Generation** = The AI uses those hints to generate a proper, accurate answer

Think of it like this: instead of the AI guessing from memory, you first give it the right book pages to look at, and THEN it writes the answer. This makes the AI dramatically more accurate and grounded in real facts.

---

## 🧠 THE CORE IDEA: WHY RAG EXISTS

### The Problem with Normal AI (LLMs)

When you ask ChatGPT something, it answers from what it learned during training (which happened months or years ago). It:
- Cannot access your private documents
- Cannot know about recent events after its training cutoff
- Sometimes "hallucinates" (makes up facts confidently)
- Has no idea what's inside your company's PDFs

### The RAG Solution

RAG solves this by:
1. First searching your documents for relevant content
2. Giving that content to the AI as context
3. Asking the AI to answer **based on that specific content**

It's like giving an open-book exam instead of a closed-book one. The AI can look at the relevant pages before answering.

---

## 📁 PROJECT STRUCTURE — FILE BY FILE

```
RAG_Practice/
├── app.py                  ← Entry point — the main script you run
├── .env                    ← Secret API key storage
├── requirements.txt        ← List of Python libraries needed
├── data/                   ← Your knowledge documents (the "books")
│   ├── pdf/                ← PDF files
│   │   ├── Machine_Learning_Basics.pdf
│   │   ├── Data_Structures.pdf
│   │   ├── Python_Basics.pdf
│   │   └── proposal.pdf
│   └── text_files/         ← Plain text files
│       ├── machine_learning.txt
│       └── python_intro.txt
├── faiss_store/            ← Saved vector database (pre-processed knowledge)
│   ├── faiss.index         ← The actual vector index (1.9 MB)
│   └── metadata.pkl        ← Text chunks linked to vectors
└── src/                    ← Python source code (the brain)
    ├── __init__.py         ← Marks this folder as a Python package
    ├── data_loader.py      ← Reads and loads documents
    ├── embedding.py        ← Converts text into numbers (vectors)
    ├── vectorstore.py      ← Stores and searches vectors using FAISS
    └── search.py           ← Searches + talks to the AI (Groq LLM)
```

---

## 🗺️ THE COMPLETE DATA PIPELINE (Step-by-Step Journey of Your Data)

Think of this as a factory assembly line with 5 stations:

```
📄 Raw Documents (PDFs, TXT files)
        ↓
🔍 Station 1: DATA LOADING (data_loader.py)
   "Read every document I can find"
        ↓
✂️  Station 2: CHUNKING (embedding.py - chunk_documents)
   "Cut long documents into small digestible pieces"
        ↓
🔢 Station 3: EMBEDDING (embedding.py - embed_chunks)
   "Convert each piece of text into a list of numbers (a vector)"
        ↓
💾 Station 4: STORAGE (vectorstore.py - save)
   "Store all those number-lists in a fast searchable database (FAISS)"
        ↓
❓ USER QUERY → Incoming Question
        ↓
🔎 Station 5: RETRIEVAL + GENERATION (search.py)
   "Find the most similar chunks to the question,
    send them to Groq AI, get a human-readable answer"
        ↓
✅ FINAL ANSWER
```

---

## 📂 FILE-BY-FILE DEEP EXPLANATION

---

### 1. `app.py` — The Boss / Entry Point

```python
import os
from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents
from src.search import RAGSearch

store = FaissVectorStore("faiss_store")
faiss_path = "faiss_store/faiss.index"

if not os.path.exists(faiss_path):
    docs = load_all_documents("data")
    store.build_from_documents(docs)
else:
    store.load()

if __name__ == "__main__":
    store = FaissVectorStore("faiss_store")
    store.load()

    rag_search = RAGSearch()
    query = "What is Naive bayes?"
    summary = rag_search.search_and_summarize(query, top_k=3)

    print("\n\n\nSummary:", summary)
```

**What is this doing?**

Think of `app.py` as the **MANAGER** of the entire operation.

**First block (lines 6-13) — Smart Initialization:**
- It checks: "Has the vector database (FAISS) already been built?"
- If **NO** (first time running): Load all documents → build the database → save it
- If **YES** (database already exists): Just load the saved database (much faster!)

This is called "caching" — you don't want to re-process 100 PDFs every single time you run the program. So once it's done, it saves the result and reuses it.

**Second block (lines 17-26) — The Main Run:**
- Creates a `RAGSearch` object
- Defines the question: `"What is Naive bayes?"`
- Calls `search_and_summarize()` with `top_k=3` (find the 3 most relevant chunks)
- Prints the final answer

---

### 2. `src/data_loader.py` — The Librarian

This file is responsible for one thing: **reading and loading all documents from the `data/` folder**.

It supports 6 types of files:
- **PDF** — via `PyPDFLoader`
- **TXT** — via `TextLoader`
- **CSV** — via `CSVLoader`
- **Excel (.xlsx)** — via `UnstructuredExcelLoader`
- **Word (.docx)** — via `Docx2txtLoader`
- **JSON** — via `JSONLoader`

**Key function:**
```python
def load_all_documents(data_dir: str) -> List[Any]:
```

**What happens inside?**
1. It takes the path to your `data/` folder
2. Uses `Path.glob('**/*.pdf')` to recursively find ALL PDFs anywhere inside that folder (even in subdirectories — that's what `**` means)
3. For each file found, it creates a "Loader" object appropriate for that file type
4. The loader reads the file and returns it as **LangChain Document objects**

**What is a LangChain Document?**
A standardized container with two parts:
- `page_content` → the actual text from the document
- `metadata` → extra info like filename, page number, etc.

This standardization means all 6 file types, despite being completely different formats, come out as the same consistent structure.

**DEBUG messages:** You'll notice many `print(f"[DEBUG] ...")` lines. These are just helpful status messages so you can see what's happening when you run the script. They don't affect functionality.

---

### 3. `src/embedding.py` — The Translator (Text → Numbers)

This is where the magic really begins.

**The Core Concept: Why Convert Text to Numbers?**

Computers fundamentally do not understand language. They only understand numbers. To compare two pieces of text and say "these are similar," we need to convert text into numbers where **similar meanings result in similar numbers**.

This is called an **Embedding**. An embedding is a list of numbers (called a "vector") that captures the *meaning* of a piece of text.

Example:
- "king" might become: [0.2, -0.5, 0.8, 0.1, ...]
- "queen" might become: [0.19, -0.48, 0.77, 0.12, ...]
- "banana" might become: [-0.8, 0.3, -0.2, 0.9, ...]

"King" and "Queen" have similar numbers → their meanings are close.
"King" and "banana" have very different numbers → meanings are far apart.

This allows us to mathematically measure how similar two pieces of text are!

**The `EmbeddingPipeline` class:**

```python
class EmbeddingPipeline:
    def __init__(self, model_name="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
        self.model = SentenceTransformer(model_name)
```

**Model Used: `all-MiniLM-L6-v2`**

This is a free, open-source AI model from HuggingFace specifically designed to convert text into embeddings. It:
- Is fast and lightweight (runs on your CPU without GPU)
- Outputs 384-dimensional vectors (each text becomes a list of 384 numbers)
- Was trained on millions of sentence pairs to understand semantic similarity
- "MiniLM" = Mini Language Model (a smaller, faster version of BERT/transformer models)
- "L6" = 6 transformer layers
- "v2" = version 2

**Step 1: Chunking — Cutting Big Documents into Small Pieces**

```python
def chunk_documents(self, documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
```

Think of a book with 500 pages. You can't embed the entire 500 pages as one unit — it would lose all the specific detail. Instead, we cut it into chunks.

**Why chunk_size=1000?**
Each chunk is at most 1000 characters long. Why? Because:
- Embedding models have a maximum input limit
- Smaller chunks are more focused → better retrieval precision
- When someone asks a question, you want to retrieve the exact relevant paragraph, not 50 irrelevant pages

**Why chunk_overlap=200?**
Imagine a chapter where the key information spans two natural paragraphs. If you cut exactly between them, you lose context. By overlapping 200 characters between chunks, you ensure important information isn't accidentally split.

**The separators list:** `["\n\n", "\n", " ", ""]`
The splitter tries to cut at natural boundaries first:
1. Double newline (paragraph break) — best split point
2. Single newline (line break)
3. Space (word boundary)
4. Anywhere (last resort)

This ensures you never cut in the middle of a word if possible.

**Step 2: Embedding — Converting Each Chunk to a Vector**

```python
def embed_chunks(self, chunks):
    texts = [chunk.page_content for chunk in chunks]
    embeddings = self.model.encode(texts, show_progress_bar=True)
    return embeddings
```

This takes all the text chunks and passes them through the SentenceTransformer model. The output is a 2D NumPy array where:
- Each row = one chunk's embedding
- Each row has 384 numbers

If you had 1000 chunks, you'd get a (1000, 384) array.

---

### 4. `src/vectorstore.py` — The Searchable Database

This file manages the **FAISS vector store** — the database where all the embeddings live, and where we search for relevant chunks.

**What is FAISS?**

FAISS stands for **Facebook AI Similarity Search**. It's an open-source library built by Facebook (Meta) Research. It solves one specific problem incredibly efficiently:

> "Given a million vectors, find the K vectors that are most similar to a query vector as fast as possible."

Without FAISS, you'd have to compare your query vector against every single stored vector one by one. With 100,000 chunks, that's 100,000 comparisons for every single question. FAISS uses smart mathematical indexing to make this much, much faster.

**The `FaissVectorStore` class:**

```python
class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2"):
        self.index = None          # The FAISS index (empty at start)
        self.metadata = []         # List of text chunks linked to each vector
```

**Key Methods:**

**`build_from_documents(documents)`:**
The main build pipeline:
1. Uses `EmbeddingPipeline` to chunk + embed all documents
2. Converts embeddings to float32 (FAISS requirement)
3. Calls `add_embeddings()` → `save()`

**`add_embeddings(embeddings, metadatas)`:**
```python
self.index = faiss.IndexFlatL2(dim)  # Create L2 distance index
self.index.add(embeddings)           # Add all vectors
```

`IndexFlatL2` = A FAISS index that uses **L2 (Euclidean) distance** to measure similarity. Lower distance = more similar.

The metadata (actual text of each chunk) is stored in a Python list in parallel with the FAISS index.

**`save()`:**
Saves two files:
- `faiss_store/faiss.index` — The FAISS index (vectors only, binary format, ~1.9 MB)
- `faiss_store/metadata.pkl` — The text metadata (pickle format, ~1 MB)

**What is Pickle?** Python's built-in serialization format. Like converting a Python object into bytes that can be saved to disk and loaded back exactly as-is.

**`query(query_text, top_k=5)`:**
```python
query_emb = self.model.encode([query_text]).astype('float32')
return self.search(query_emb, top_k=top_k)
```
1. Embeds the query text into a vector using the same model
2. Calls `search()` with that vector

**`search(query_embedding, top_k=5)`:**
```python
D, I = self.index.search(query_embedding, top_k)
```
FAISS returns two arrays:
- `D` = Distances (how similar — lower is better)
- `I` = Indices (which stored vectors are the closest matches)

Then for each result, it retrieves the corresponding metadata (actual text chunk).

---

### 5. `src/search.py` — The Smart Answerer (RAG in Action)

This is the final, most exciting piece. It takes a question, finds relevant content, and uses a real AI language model to generate an answer.

```python
class RAGSearch:
    def __init__(self, persist_dir="faiss_store", embedding_model="all-MiniLM-L6-v2", llm_model="gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        self.llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=groq_api_key)
```

**What is Groq?**

Groq is a cloud service that gives you API access to powerful open-source language models (like Meta's LLaMA, Google's Gemma) at **blazing fast speeds**. Groq has custom hardware chips (LPUs — Language Processing Units) that are specifically engineered for running AI inference much faster than regular GPUs.

**Model used: `llama-3.1-8b-instant`**
This is Meta's LLaMA 3.1 model, 8 billion parameters version. "instant" means it's optimized for speed. It's a powerful, high-quality language model that:
- Can understand complex questions
- Can read and synthesize long context
- Can produce human-like, coherent summaries

**The `search_and_summarize()` method — This is RAG in 5 lines:**

```python
def search_and_summarize(self, query: str, top_k: int = 5) -> str:
    # Step 1: Find relevant chunks
    results = self.vectorstore.query(query, top_k=top_k)

    # Step 2: Extract actual text from results
    texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
    context = "\n\n".join(texts)

    # Step 3: Build a prompt
    prompt = f"""Summarize the following context for the query: '{query}'
Context:
{context}
Summary:"""

    # Step 4: Send to LLM and return the answer
    response = self.llm.invoke([prompt])
    return response.content
```

**Breaking it down:**

1. **Query the vector store** with `top_k=3` → get back the 3 most relevant text chunks from your documents
2. **Extract the text** from each result's metadata
3. **Join them** into one big "context" block
4. **Build a prompt**: Tell the AI "Here is some context from documents, please answer this question based on it"
5. **Send to Groq** via `self.llm.invoke()` → get back the AI-generated answer
6. **Return** the answer text

This is the entire RAG flow in one function!

---

## 🔑 KEY TECHNOLOGIES EXPLAINED DEEPLY

---

### 🦜 LangChain

**What is it?** LangChain is a Python framework for building AI applications. It provides:
- Standard interfaces for documents, loaders, embeddings
- A massive library of connectors to 100+ data sources
- Chain/pipeline building tools
- Memory management for chatbots

**Why use it?** Instead of writing custom code to handle PDFs, TXTs, CSVs, etc., you use LangChain's battle-tested loaders. Instead of implementing text splitters from scratch, you use its `RecursiveCharacterTextSplitter`. It dramatically accelerates building AI applications.

**Modules used in this project:**
- `langchain_community.document_loaders` — File loaders (PDF, TXT, CSV, etc.)
- `langchain_text_splitters` — Text chunking
- `langchain_groq` — Connection to Groq API

---

### 🤗 Sentence Transformers (HuggingFace)

**What is it?** A Python library for generating sentence/text embeddings. Built on top of HuggingFace Transformers.

**How it works internally:**
1. Tokenizes text (splits into sub-words)
2. Passes through a BERT-like transformer model
3. Applies mean pooling to get a fixed-size vector
4. Optionally normalizes the vector

**The model `all-MiniLM-L6-v2`:**
- Architecture: BERT-based (Bidirectional Encoder Representations from Transformers)
- Training: Contrastive learning on millions of (sentence, similar_sentence) pairs
- Output dimensions: 384
- Speed: ~14,000 sentences/second on CPU
- Great for semantic search, clustering, and RAG systems

---

### ⚡ FAISS (Facebook AI Similarity Search)

**What is it?** A library for efficient similarity search and clustering of dense vectors. Created by FAIR (Facebook AI Research).

**Types of indexes:**
- `IndexFlatL2` (used here) — Exact search using Euclidean distance. No approximation. 100% accurate but O(n) time.
- `IndexIVFFlat` — Inverted index, approximate but much faster for large scale
- `IndexHNSW` — Hierarchical Navigable Small World graphs, very fast approximate search

**Why L2 distance?**
L2 (Euclidean) distance measures the straight-line distance between two points in high-dimensional space. For normalized embeddings, this correlates directly with cosine similarity. Smaller L2 distance = vectors are closer = texts are semantically similar.

**The `faiss.index` file (1.9 MB):**
This contains a binary-encoded flat matrix of all the vectors. When loaded, it becomes an in-memory structure that FAISS can search through.

---

### 🧩 NumPy

**What is it?** The fundamental library for numerical computing in Python.

**Why used here?**
- FAISS requires `float32` arrays (not Python lists)
- NumPy arrays are memory-efficient and extremely fast
- `np.ndarray` is the standard format for storing embedding matrices

```python
embeddings = np.array(embeddings).astype('float32')  # Convert to FAISS-compatible format
```

---

### 🌿 python-dotenv

**What is it?** A tiny library that reads from `.env` files and sets them as environment variables.

**Why use it?** You never want to hardcode API keys in your source code, because:
- If you push to GitHub, your key is exposed to the world
- Other developers can't use their own keys
- You can't easily switch between development/production keys

The `.env` file:
```
GROQ_API_KEY = "gsk_..."
```

The code:
```python
from dotenv import load_dotenv
load_dotenv()  # Reads .env and sets variables
groq_api_key = os.getenv("GROQ_API_KEY")  # Safe access
```

The `.gitignore` file includes `.env` so it's never committed to git.

---

### 📄 PyPDF / PyMuPDF

**What are they?**
- **PyPDF** (`pypdf`) — Pure Python PDF reader. Can extract text page by page.
- **PyMuPDF** (`fitz`) — Faster, more powerful PDF processing using the MuPDF library

**How LangChain uses them:**
`PyPDFLoader` uses `pypdf` under the hood to read each page of a PDF and return it as a separate LangChain Document. So a 50-page PDF becomes 50 Document objects.

---

### 🏗️ Python Packages & Imports

**`from pathlib import Path`**
Modern Python way to handle file paths. Works on Windows, Mac, and Linux without issues. `Path.glob('**/*.pdf')` means "find all PDF files in this directory and all subdirectories recursively."

**`from typing import List, Any`**
Python type hints. `List[Any]` means "a list containing anything." These are optional but make code documentation and IDE support much better.

**`import pickle`**
Python's built-in serialization module. Like a "freeze-dry" for Python objects — saves them to disk, reloads them exactly as they were.

**`import os`**
Standard Python library for operating system operations: checking if files exist (`os.path.exists`), joining paths (`os.path.join`), reading environment variables (`os.getenv`).

---

## 🔄 COMPLETE EXECUTION FLOW (What Happens When You Run `python app.py`)

### First Run (No FAISS index yet):

```
python app.py
  ↓
Check: faiss_store/faiss.index exists? → NO
  ↓
load_all_documents("data")
  → Scans data/pdf/ → finds 4 PDFs
  → Loads each with PyPDFLoader → ~100+ Document objects
  → Scans data/text_files/ → finds 2 TXTs
  → Loads each with TextLoader → 2 Document objects
  → Total: ~102+ documents
  ↓
store.build_from_documents(docs)
  → EmbeddingPipeline.chunk_documents()
     → RecursiveCharacterTextSplitter splits 102 docs
     → Creates thousands of 1000-char chunks
  → EmbeddingPipeline.embed_chunks()
     → all-MiniLM-L6-v2 converts each chunk to 384 numbers
     → Output: (N, 384) numpy array
  → add_embeddings()
     → Creates faiss.IndexFlatL2(384)
     → Adds all N vectors to the index
  → save()
     → Writes faiss_store/faiss.index (binary)
     → Writes faiss_store/metadata.pkl (pickle)
  ↓
RAGSearch() initialized
  → Loads faiss.index
  ↓
rag_search.search_and_summarize("What is Naive bayes?", top_k=3)
  → query embedding: "What is Naive bayes?" → [0.12, -0.03, ...] (384 numbers)
  → FAISS search: find 3 closest vectors in index
  → Retrieve metadata[idx] for each result (actual text chunks)
  → Build prompt with those 3 text chunks as context
  → Send prompt to Groq (llama-3.1-8b-instant)
  → Get back human-readable answer
  ↓
Print: "Summary: Naive Bayes is a probabilistic classifier..."
```

### Second Run (FAISS index already exists):

```
python app.py
  ↓
Check: faiss_store/faiss.index exists? → YES
  ↓
store.load()  ← Skip all the heavy processing!
  → Read faiss_store/faiss.index
  → Load faiss_store/metadata.pkl
  ↓
(same RAG search as above...)
```

This is much faster — seconds instead of minutes!

---

## 🧪 THE DATA: What Knowledge Does This RAG System Know?

The system's knowledge comes from the `data/` folder:

| File | Type | Content |
|------|------|---------|
| `Machine_Learning_Basics.pdf` | PDF (10.8 MB) | ML concepts: regression, classification, Naive Bayes, SVM, etc. |
| `Data_Structures.pdf` | PDF (1.1 MB) | CS data structures: arrays, trees, graphs, etc. |
| `Python_Basics.pdf` | PDF (1.7 MB) | Python programming fundamentals |
| `proposal.pdf` | PDF (130 KB) | Some kind of project proposal |
| `machine_learning.txt` | TXT | "Machine learning is a subset of AI..." |
| `python_intro.txt` | TXT | "Python is a high level interpreted language..." |

The test query in `app.py` is `"What is Naive bayes?"` — a topic covered in the Machine Learning PDF. This is a great test case because Naive Bayes is a specific topic that's clearly explained in one of your documents.

---

## 🏛️ ARCHITECTURE PATTERNS USED

### 1. Separation of Concerns
Each module has ONE responsibility:
- `data_loader.py` → Only loads data
- `embedding.py` → Only handles embeddings
- `vectorstore.py` → Only manages the vector database
- `search.py` → Only orchestrates the RAG query

This makes the code clean, testable, and maintainable.

### 2. Lazy Loading / Caching
The system checks if the FAISS index already exists before building it. This is a fundamental optimization pattern: compute once, reuse many times.

### 3. Python Package Structure
The `src/` directory has an `__init__.py` file. This makes `src` a Python "package" — a group of related modules that can be imported as `from src.vectorstore import FaissVectorStore`. Without `__init__.py`, Python wouldn't recognize `src` as a package.

### 4. Environment-Based Configuration
API keys and sensitive information are stored in `.env` (not in code). This follows the "12-Factor App" methodology for clean, deployable software.

---

## 🔧 WHAT EACH LIBRARY IN `requirements.txt` DOES

| Library | What It Does |
|---------|-------------|
| `langchain` | Core LangChain framework — chains, prompts, document structures |
| `langchain-community` | Community integrations — PDF loaders, CSV loaders, etc. |
| `langchain-groq` | Connects LangChain to Groq's API for fast LLM inference |
| `langchain-text-splitters` | Smart text chunking utilities |
| `sentence-transformers` | Pre-trained models for converting text to vectors |
| `faiss-cpu` | FAISS library for CPU (no GPU required) — vector similarity search |
| `python-dotenv` | Loads `.env` files into environment variables |
| `pypdf` | Reads PDF files, extracts text page by page |
| `pymupdf` | Advanced PDF processing (faster alternative to pypdf) |

---

## 💡 CONCEPTS GLOSSARY (Plain English)

| Term | Plain English Explanation |
|------|--------------------------|
| **RAG** | Teach AI to answer from YOUR documents, not just its training data |
| **Embedding** | Converting text into a list of numbers that captures its meaning |
| **Vector** | Just a list of numbers (e.g., [0.2, -0.5, 0.8, ...]) |
| **Vector Store** | A database optimized for storing and searching vectors |
| **Semantic Search** | Finding similar meaning, not just matching exact words |
| **Chunking** | Cutting long documents into smaller, focused pieces |
| **FAISS** | Facebook's super-fast vector search engine |
| **LLM** | Large Language Model — the AI that generates human-like text |
| **Groq** | Cloud service that runs LLMs at extremely fast speeds |
| **LangChain** | Framework that makes it easy to build AI pipelines |
| **Inference** | Running an AI model to get an output (vs training it) |
| **Token** | A small unit of text (roughly 4 characters or 3/4 of a word) |
| **Prompt** | The input/instruction you give to an AI model |
| **Context Window** | How much text an LLM can read at one time |
| **Hallucination** | When AI confidently says something false |
| **IndexFlatL2** | FAISS's exact search using Euclidean distance |
| **pickle** | Python's way of saving complex objects to disk |
| **Environment Variable** | A variable stored in the OS, not in code (used for secrets) |
| **top_k** | "Give me the K most relevant results" — in this case, top 3 chunks |
| **float32** | A data type for numbers with decimal points; FAISS requires it |
| **API** | A way for one program to talk to another over the internet |
| **API Key** | A password that authorizes you to use an external service |

---

## 🔎 FEATURE UPDATE: SOURCE CITATIONS & PROOF

To ensure factuality and verification, a **Source Citation** engine has been added to the codebase. When the AI model produces an answer, the system retrieves and displays the exact source citations of the text fragments that were fed as context.

### How Citations are Extracted and Formatted:
* **PDF Ingestion**: Preserves the `page` metadata (0-indexed) returned by the LangChain loaders, presenting it as 1-indexed to the user.
  * *Citation layout*: `Doc: <filename>, Page: <page_no>`
* **Text (.txt) Ingestion**: Reads the original `.txt` file at runtime and searches for the extracted text chunk, computing its exact start line number dynamically.
  * *Citation layout*: `Doc: <filename>, Line: <line_no>`
* **Tabular (.csv) Ingestion**: Leverages the loader's native `row` index mapping.
  * *Citation layout*: `Doc: <filename>, Row: <row_no>`
* **Hierarchical (.json) Ingestion**: Tracks the sequence index of the JSON payload.
  * *Citation layout*: `Doc: <filename>, Sequence: <index_no>`

### Workflow Changes:
1. **`src/vectorstore.py`**: The metadata array now preserves the complete loader metadata block (`**chunk.metadata`) when saving FAISS state rather than stripping it.
2. **`src/search.py`**: The `search_and_summarize()` function returns a structured `dict` (containing both the LLM `"summary"` text and a list of `"citations"` strings) rather than a simple string.
3. **`app.py`**: Iterates and prints the retrieved source list at the base of every answer.

---

## 🚀 POTENTIAL IMPROVEMENTS & EXTENSIONS

Since this is a "practice" project, here are natural next steps:

1. **Add a Web Interface** — Use Flask/FastAPI + React to build a chat UI
2. **Add Streaming** — Stream tokens as they're generated instead of waiting for full response
3. **Add Memory** — Let users have multi-turn conversations with conversation history
4. **Better Chunking** — Use semantic chunking (split by meaning, not just character count)
5. **Re-ranking** — After FAISS retrieval, re-rank results using a cross-encoder model
6. **Multiple Query Expansion** — Generate multiple versions of the query to search from different angles
7. **Metadata Filtering** — Filter chunks by document source before semantic search
8. **Evaluation** — Add RAGAs framework to measure retrieval precision and answer quality
9. **Hybrid Search** — Combine FAISS semantic search with keyword search (BM25) for better results
10. **Document Update Handling** — Detect when documents change and re-embed only changed ones

---

## 📊 VISUAL SUMMARY OF THE RAG PIPELINE

```
┌─────────────────────────────────────────────────────────┐
│                    INDEXING PHASE                        │
│              (Done once, saved to disk)                  │
│                                                          │
│  📄 PDFs + TXTs                                          │
│       ↓ data_loader.py                                   │
│  📋 LangChain Documents                                  │
│       ↓ embedding.py (chunk_documents)                   │
│  ✂️  Text Chunks (1000 chars each)                       │
│       ↓ embedding.py (embed_chunks)                      │
│  🔢 Vectors (384 numbers each)                           │
│       ↓ vectorstore.py (add_embeddings + save)           │
│  💾 FAISS Index + Metadata PKL                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   QUERYING PHASE                         │
│            (Done every time you ask a question)          │
│                                                          │
│  ❓ User Question: "What is Naive Bayes?"               │
│       ↓ sentence-transformers model                      │
│  🔢 Query Vector: [0.12, -0.03, 0.87, ...]              │
│       ↓ FAISS IndexFlatL2.search(top_k=3)               │
│  📌 Top 3 Matching Chunks (from your PDFs)              │
│       ↓ search.py (build prompt)                         │
│  📝 Prompt = "Given this context: [chunks], answer: [Q]"│
│       ↓ Groq API (llama-3.1-8b-instant)                 │
│  ✅ Final Answer: "Naive Bayes is a probabilistic..."   │
└─────────────────────────────────────────────────────────┘
```

---

*These notes were automatically generated from a complete analysis of the RAG_Practice codebase.*
