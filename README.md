# AI Customer Support Assistant

An AI-powered customer support assistant that classifies incoming customer queries and generates grounded answers from company knowledge-base documents (policies, product catalog, technical guides) using Retrieval-Augmented Generation (RAG).

The system combines:
- A classical ML classifier (TF-IDF + Linear SVM) for fast intent detection
- A RAG pipeline (section-aware chunking + FAISS + FLAN-T5) for accurate, document-grounded responses

---

## Objectives

- **Final Goal**: Build a backend that can take a customer question, predict its category, retrieve the most relevant sections from uploaded PDFs/TXT files, and generate a short, grounded answer — or clearly say when the information is not available.
- Classify customer queries into support categories with confidence scores.
- Index and manage knowledge-base documents (product catalog, refund/cancellation policy, shipping policy, technical support guide, etc.).
- Answer questions only from the uploaded documents (no hallucination).
- Expose a clean FastAPI backend that can be used by any frontend or chatbot UI.

---

## Project Lifecycle & Tools Overview

| Lifecycle Step       | Tool(s) Used                                      |
|----------------------|---------------------------------------------------|
| **Data Preparation** | `pandas`, Jupyter Notebooks                       |
| **Classification**   | `scikit-learn` (TF-IDF + Linear SVM + Calibration)|
| **Document Parsing** | `PyMuPDF` (PDF), plain TXT                        |
| **Chunking**         | Custom section-aware chunker                      |
| **Embeddings**       | `sentence-transformers` (`all-MiniLM-L6-v2`)      |
| **Vector Store**     | `FAISS` (IndexFlatIP)                             |
| **Generation**       | `transformers` + `google/flan-t5-large`            |
| **API**              | `FastAPI` + `uvicorn` + `pydantic`                |
| **Serialization**    | `joblib`                                          |

---

## Key Features

- **Query Classification**  
  TF-IDF + calibrated Linear SVM predicts the intent category of a customer query and returns confidence + full probability distribution.

- **Document Upload & Indexing**  
  Upload one or more PDF / TXT files. The system extracts text, removes boilerplate, performs section-aware chunking, embeds the chunks, and adds them to a FAISS index.

- **Grounded RAG Answers**  
  Retrieves the most relevant chunks, builds a strict prompt, and generates an answer only from the retrieved context. Returns status:
  - `answered`
  - `not_specified`
  - `not_in_documents`
  - `no_documents`

- **Source Attribution**  
  Every answer includes the source document name, page number, section heading, and similarity score.

- **Document Management**  
  List all indexed documents and delete any document (the index is automatically rebuilt).

---

## Project Structure

```
AI_Customer_Support_Assistant/
│
├── Data/
│   ├── customer_support_dataset.csv          # Training data for the classifier
│   ├── product_catalog.pdf
│   ├── refund_cancellation_policy.pdf
│   ├── shipping_delivery_policy.pdf
│   └── technical_support_guide.pdf
│
├── notebooks/
│   ├── 01_dataset_preparation.ipynb
│   ├── 02_customer_query_classification.ipynb
│   └── 03_rag_implementation.ipynb
│
├── Output/                                   # Pre-built artifacts (optional)
│   ├── customer_query_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── index.faiss
│   └── chunks.json
│
├── backend_package/
│   └── backend/
│       ├── main.py                           # FastAPI application
│       ├── config.py                         # Central settings
│       ├── schemas.py                        # Pydantic models
│       ├── classifier_service.py             # Query classification
│       ├── rag_service.py                    # RAG pipeline
│       └── model_loaders.py                  # Embedder + LLM loaders
│
├── requirement.txt
├── LICENSE
└── README.md
```

---

## Complete Workflow

```
Customer Query
      │
      ├──────────────────► Classifier (TF-IDF + Linear SVM)
      │                         │
      │                         ▼
      │                   Predicted Category + Confidence
      │
      └──────────────────► RAG Pipeline
                                │
                                ├─ Retrieve top-k chunks (FAISS)
                                ├─ Filter by similarity threshold
                                ├─ Build grounded prompt
                                └─ Generate answer with FLAN-T5
                                      │
                                      ▼
                              Final Response
                              (answer + sources + status + category)
```

---

## API Endpoints

| Method   | Endpoint                  | Description                                      |
|----------|---------------------------|--------------------------------------------------|
| `GET`    | `/`                       | API info & available endpoints                   |
| `GET`    | `/health`                 | Health check + loaded models & document count    |
| `POST`   | `/classify-query`         | Predict category of a customer query             |
| `POST`   | `/generate-response`      | Classify + generate grounded RAG answer          |
| `POST`   | `/upload-document`        | Upload & index one or more PDF/TXT files         |
| `GET`    | `/documents`              | List all indexed documents                       |
| `DELETE` | `/documents/{filename}`   | Remove a document and rebuild the index          |

Interactive docs available at: `http://localhost:8000/docs`

---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/Rajvib-0/AI_Customer_Support_Assistant.git
cd AI_Customer_Support_Assistant
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux / macOS
# or
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirement.txt
```

### 4. Prepare the classifier model

The backend expects a trained classifier at `models/classifier.joblib` (or the path set by `CLASSIFIER_PATH`).

You can train it from the notebook or from code:

```python
from backend_package.backend.classifier_service import train_and_save

metrics = train_and_save(
    csv_path="Data/customer_support_dataset.csv",
    out_path="models/classifier.joblib"
)
print(metrics)
```

### 5. Run the FastAPI server

From the project root:

```bash
# Make sure the backend package is importable
export PYTHONPATH=$PYTHONPATH:$(pwd)/backend_package

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Test the API

- Open **http://localhost:8000/docs**
- Upload the sample PDFs from the `Data/` folder using `/upload-document`
- Try `/generate-response` with questions such as:
  - “How long does an approved refund take?”
  - “What is the shipping policy for international orders?”
  - “How do I reset my device?”

---

## Configuration

All important settings live in `backend_package/backend/config.py` and can be overridden with environment variables:

| Variable              | Default                              | Description                              |
|-----------------------|--------------------------------------|------------------------------------------|
| `EMBEDDING_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model                       |
| `LLM_MODEL`           | `google/flan-t5-large`               | Generation model                         |
| `CLASSIFIER_PATH`     | `models/classifier.joblib`           | Path to trained classifier               |
| `TOP_K`               | `4`                                  | Number of chunks to retrieve             |
| `MIN_SIMILARITY`      | `0.30`                               | Minimum cosine similarity threshold      |
| `MAX_UPLOAD_MB`       | `10`                                 | Max file size for uploads                |
| `CHUNK_MAX_WORDS`     | `90`                                 | Maximum words per chunk                  |

---

## Models Used

| Component     | Model / Method                          | Notes                                      |
|---------------|-----------------------------------------|--------------------------------------------|
| Classifier    | TF-IDF + Linear SVM (Calibrated)        | Fast, interpretable, good on short queries |
| Embeddings    | `all-MiniLM-L6-v2`                      | 384-dimensional, high quality & lightweight|
| Generator     | `google/flan-t5-large`                  | Instruction-tuned, good at following the “answer only from context” constraint |

---

## Notebooks Overview

1. **01_dataset_preparation.ipynb** – Cleaning and preparing the customer support dataset  
2. **02_customer_query_classification.ipynb** – Training & evaluating the TF-IDF + SVM classifier  
3. **03_rag_implementation.ipynb** – End-to-end RAG experiments (chunking, retrieval, generation)

---

## License

This project is licensed under the **MIT License**.
```
