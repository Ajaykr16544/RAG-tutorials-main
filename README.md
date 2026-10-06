# RAG Tutorial Project

A simple Retrieval-Augmented Generation (RAG) application using:

* Python
* LangChain
* FAISS
* Sentence Transformers
* Groq LLM
* Hugging Face
* `.env` for API key management

The application loads documents, converts them into embeddings, stores them in FAISS, retrieves relevant information, and uses Groq LLM to generate an answer.

---

# 1. Project Structure

```text
RAG-Tutorials-main/
│
├── data/
│   ├── company.txt
│   ├── products.txt
│   └── attention.txt
│
├── faiss_store/
│   ├── faiss.index
│   └── metadata.pkl
│
├── src/
│   ├── search.py
│   └── vectorstore.py
│
├── app.py
├── data_loader.py
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 2. Open Project Folder

Open CMD and run:

```cmd
cd /d D:\RAG-Tutorials-main\RAG-Tutorials-main
```

---

# 3. Create Virtual Environment

Create Python virtual environment:

```cmd
python -m venv venv
```

Activate it:

```cmd
venv\Scripts\activate
```

After activation, terminal should show:

```text
(venv) D:\RAG-Tutorials-main\RAG-Tutorials-main>
```

---

# 4. Upgrade pip

Run:

```cmd
python -m pip install --upgrade pip
```

---

# 5. Install Required Packages

Install FAISS:

```cmd
python -m pip install faiss-cpu
```

Install Sentence Transformers:

```cmd
python -m pip install sentence-transformers
```

Install scikit-learn:

```cmd
python -m pip install scikit-learn
```

Install LangChain:

```cmd
python -m pip install langchain
```

Install LangChain Groq:

```cmd
python -m pip install langchain-groq
```

Install LangChain text splitters:

```cmd
python -m pip install langchain-text-splitters
```

Install python-dotenv:

```cmd
python -m pip install python-dotenv
```

---

# 6. Verify Python Packages

Run:

```cmd
python -c "import faiss; print('FAISS OK')"
```

Run:

```cmd
python -c "import sentence_transformers; print('SENTENCE TRANSFORMERS OK')"
```

Run:

```cmd
python -c "import sklearn; print('SKLEARN OK')"
```

Run:

```cmd
python -c "import langchain; print('LANGCHAIN OK')"
```

Run:

```cmd
python -c "from langchain_groq import ChatGroq; print('LANGCHAIN GROQ OK')"
```

Run:

```cmd
python -c "from dotenv import load_dotenv; print('DOTENV OK')"
```

---

# 7. Create Data Folder

Create the data folder:

```cmd
mkdir data
```

The `data` folder contains the documents that the RAG system will search.

---

# 8. Add Company Document

Create:

```text
data/company.txt
```

Example:

```text
NovaTech Innovations is a technology company focused on
software development, cloud computing, artificial intelligence,
and modern digital solutions.

The company provides technology solutions for businesses and
helps organizations improve their digital operations.
```

---

# 9. Add Products Document

Create:

```text
data/products.txt
```

Example:

```text
NovaTech Innovations provides several technology products.

NovaCloud is a cloud platform for deploying and managing
applications.

NovaDeploy is a DevOps deployment platform that helps teams
automate application deployment.

NovaAI provides artificial intelligence and machine learning
solutions.

NovaSecure provides cybersecurity and security monitoring
solutions.
```

---

# 10. Add Attention Mechanism Document

Create:

```text
data/attention.txt
```

Add:

```text
Attention Mechanism

Attention mechanism is a technique used in deep learning and
Natural Language Processing (NLP) that allows a model to focus
on the most important parts of the input.

It helps the model understand which words or tokens are more
important when processing a sentence.

Attention is a key component of Transformer models. It is used
in modern AI systems such as language models.

Self-attention allows each token in a sequence to look at other
tokens and determine how strongly they are related.

The main purpose of attention is to capture relationships between
different parts of the input and improve the quality of the
model's understanding.
```

---

# 11. Create `.env`

Create:

```text
.env
```

Inside `.env` add:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Replace `YOUR_GROQ_API_KEY` with your own Groq API key.

Never upload the real API key to GitHub.

---

# 12. Check `.env`

Run:

```cmd
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GROQ KEY FOUND:', bool(os.getenv('GROQ_API_KEY')))"
```

Expected:

```text
GROQ KEY FOUND: True
```

---

# 13. `.gitignore`

Create:

```text
.gitignore
```

Add:

```text
venv/
.env
__pycache__/
*.pyc
faiss_store/
```

This prevents private API keys, virtual environment files, and generated FAISS files from being uploaded to GitHub.

---

# 14. Fix for scikit-learn

If scikit-learn shows an error even though pip says it is installed, run:

```cmd
python -m pip install --ignore-installed --no-deps scikit-learn==1.9.1
```

Then verify:

```cmd
python -c "import sklearn; print(sklearn.__version__)"
```

---

# 15. Vector Store

The project uses:

```text
Sentence Transformer
        ↓
Embeddings
        ↓
FAISS
```

The embedding model used is:

```text
all-MiniLM-L6-v2
```

The generated embedding size is:

```text
384
```

Example output:

```text
[INFO] Embeddings shape: (3, 384)
```

---

# 16. Important: Rebuild FAISS After Adding Documents

Whenever a new document is added to the `data` folder, delete the old FAISS index.

Run:

```cmd
rmdir /s /q faiss_store
```

Then run the application again.

The application will rebuild the vector store from all documents.

---

# 17. RAG Search Code

The main RAG search logic is inside:

```text
src/search.py
```

The code:

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from src.vectorstore import FaissVectorStore

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir="faiss_store",
        embedding_model="all-MiniLM-L6-v2",
        llm_model="openai/gpt-oss-20b"
    ):
        self.vectorstore = FaissVectorStore(
            persist_dir,
            embedding_model
        )

        faiss_path = os.path.join(
            persist_dir,
            "faiss.index"
        )

        meta_path = os.path.join(
            persist_dir,
            "metadata.pkl"
        )

        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.vectorstore.load()

        else:
            from data_loader import load_all_documents

            docs = load_all_documents("data")

            self.vectorstore.build_from_documents(docs)

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found in .env file"
            )

        self.llm = ChatGroq(
            groq_api_key=api_key,
            model_name=llm_model
        )

        print("[INFO] Groq LLM initialized")

    def search_and_summarize(
        self,
        query,
        top_k=3
    ):

        results = self.vectorstore.query(
            query,
            top_k=top_k
        )

        context = "\n\n".join(
            r["metadata"].get("text", "")
            for r in results
            if r.get("metadata")
        )

        if not context:
            return "No relevant documents found."

        prompt = f"""
Answer the query using only the context below.

Query:
{query}

Context:
{context}

Give a simple and clear answer.
"""

        response = self.llm.invoke(prompt)

        return response.content


if __name__ == "__main__":

    rag = RAGSearch()

    query = "What is attention mechanism?"

    answer = rag.search_and_summarize(
        query,
        top_k=3
    )

    print("\nAnswer:")
    print(answer)
```

---

# 18. Run the Application

Make sure virtual environment is active:

```cmd
venv\Scripts\activate
```

Then:

```cmd
python app.py
```

---

# 19. Expected RAG Process

When the application starts, it performs:

```text
Load Documents
      ↓
Split Documents
      ↓
Generate Embeddings
      ↓
Create FAISS Index
      ↓
Save FAISS Index
      ↓
Load FAISS Index
      ↓
Search Relevant Documents
      ↓
Send Context to Groq
      ↓
Generate Final Answer
```

---

# 20. Expected Output

For:

```text
What is attention mechanism?
```

The output should be similar to:

```text
[INFO] Groq LLM initialized
[INFO] Querying vector store for: 'What is attention mechanism?'

Answer:

Attention mechanism is a technique used in deep learning and
Natural Language Processing that helps a model focus on the
most important parts of the input.

It helps models understand relationships between different
words or tokens and is an important component of Transformer
models.
```

The exact wording can be different because the LLM generates the final response.

---

# 21. Test Company Question

Change:

```python
query = "What is attention mechanism?"
```

to:

```python
query = "What is NovaTech Innovations?"
```

Run:

```cmd
python app.py
```

The RAG system should retrieve information from:

```text
company.txt
```

and generate an answer using that context.

---

# 22. Test Product Question

Change the query to:

```python
query = "What products does NovaTech offer?"
```

Run:

```cmd
python app.py
```

The system should retrieve information from:

```text
products.txt
```

---

# 23. Important RAG Concept

RAG means:

```text
Retrieval-Augmented Generation
```

It combines two main processes:

### Retrieval

FAISS searches the documents and finds relevant information.

### Generation

Groq LLM uses the retrieved information to generate the final answer.

Therefore:

```text
User Question
      ↓
Embedding
      ↓
FAISS Search
      ↓
Relevant Context
      ↓
Groq LLM
      ↓
Final Answer
```

---

# 24. Common Errors and Solutions

## Error 1: FAISS Not Found

Error:

```text
ModuleNotFoundError: No module named 'faiss'
```

Fix:

```cmd
python -m pip install faiss-cpu
```

---

## Error 2: Sentence Transformers Not Found

Error:

```text
ModuleNotFoundError: No module named 'sentence_transformers'
```

Fix:

```cmd
python -m pip install sentence-transformers
```

---

## Error 3: LangChain Not Found

Error:

```text
ModuleNotFoundError: No module named 'langchain'
```

Fix:

```cmd
python -m pip install langchain
```

---

## Error 4: LangChain Groq Not Found

Error:

```text
ModuleNotFoundError: No module named 'langchain_groq'
```

Fix:

```cmd
python -m pip install langchain-groq
```

---

## Error 5: dotenv Not Found

Fix:

```cmd
python -m pip install python-dotenv
```

---

## Error 6: GROQ API Key Not Found

Error:

```text
GROQ_API_KEY not found in .env file
```

Check `.env`:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Then verify:

```cmd
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('GROQ KEY FOUND:', bool(os.getenv('GROQ_API_KEY')))"
```

Expected:

```text
GROQ KEY FOUND: True
```

---

## Error 7: Old Groq Model

If Groq says:

```text
model_decommissioned
```

Do not use old models such as:

```text
gemma2-9b-it
```

The project currently uses:

```python
llm_model="openai/gpt-oss-20b"
```

---

## Error 8: Answer Not Found

If the output says:

```text
The provided context does not contain any information...
```

check whether the required information exists inside the `data` folder.

If a new document was added, rebuild FAISS:

```cmd
rmdir /s /q faiss_store
```

Then:

```cmd
python app.py
```

---

# 25. Hugging Face Warning

You may see:

```text
Warning: You are sending unauthenticated requests to the HF Hub.
```

This is a warning, not a project error.

If the model loads successfully:

```text
[INFO] Loaded embedding model: all-MiniLM-L6-v2
```

the application can continue running.

---

# 26. Complete Run Sequence

For a fresh setup:

```cmd
cd /d D:\RAG-Tutorials-main\RAG-Tutorials-main
```

```cmd
python -m venv venv
```

```cmd
venv\Scripts\activate
```

```cmd
python -m pip install --upgrade pip
```

```cmd
python -m pip install faiss-cpu sentence-transformers scikit-learn langchain langchain-groq langchain-text-splitters python-dotenv
```

Create:

```text
data/company.txt
data/products.txt
data/attention.txt
```

Create:

```text
.env
```

Add:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

Then:

```cmd
rmdir /s /q faiss_store
```

Finally:

```cmd
python app.py
```

---

# 27. Final Architecture

```text
                 USER
                   │
                   ▼
              QUESTION
                   │
                   ▼
        Sentence Transformer
                   │
                   ▼
              EMBEDDING
                   │
                   ▼
                FAISS
           Vector Database
                   │
                   ▼
          Relevant Documents
                   │
                   ▼
              CONTEXT
                   │
                   ▼
             GROQ LLM
                   │
                   ▼
             FINAL ANSWER
```

---

# 28. Project Status

The project successfully implements:

* Document loading
* TXT document processing
* Text chunking
* Sentence Transformer embeddings
* FAISS vector database
* Similarity search
* Groq LLM integration
* Environment variable API key
* Retrieval-Augmented Generation
* Context-based answering

The project can be extended later with:

* PDF support
* Multiple document formats
* Chat history
* Web UI
* Streamlit
* FastAPI
* React frontend
* Conversational RAG
* Source/document citations
* Docker deployment
* Cloud deployment
"# RAG-tutorials-main" 
"# RAG-tutorials-main" 
