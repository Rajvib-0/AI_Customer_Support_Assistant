"""FastAPI application.  Run:  uvicorn backend.main:app --host 0.0.0.0 --port 8000"""
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config as cfg
from .classifier_service import ClassifierService
from .rag_service import RAGService, extract_pages
from .schemas import (ClassificationResponse, DocumentsResponse, GenerateRequest, GenerateResponse,
                      HealthResponse, QueryRequest, UploadResponse, UploadResult)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

DESCRIPTION = """
Backend for the **AI-Powered Customer Support Assistant**.

* `POST /upload-document` - index PDF/TXT knowledge-base documents
* `POST /classify-query` - predict the intent of a customer query
* `POST /generate-response` - classify the query and answer it from the uploaded documents
"""


def create_app(classifier=None, rag=None) -> FastAPI:
    """Build the app. Pass ready-made services to skip model loading (used for tests)."""
    state = {"classifier": classifier, "rag": rag}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if state["classifier"] is None:
            log.info("Loading classifier from %s", cfg.CLASSIFIER_PATH)
            state["classifier"] = ClassifierService(cfg.CLASSIFIER_PATH)
        if state["rag"] is None:
            from .model_loaders import load_embedder, load_llm_generate
            state["rag"] = RAGService(load_embedder(), load_llm_generate())
        log.info("Backend ready")
        yield

    app = FastAPI(title="AI Customer Support Assistant API", version="1.0.0",
                  description=DESCRIPTION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    def get_classifier() -> ClassifierService:
        if state["classifier"] is None:
            raise HTTPException(503, "Classifier is not loaded yet")
        return state["classifier"]

    def get_rag() -> RAGService:
        if state["rag"] is None:
            raise HTTPException(503, "RAG service is not loaded yet")
        return state["rag"]

    # ------------------------------------------------------------ info
    @app.get("/", tags=["Info"])
    def root():
        return {"name": "AI Customer Support Assistant API", "version": "1.0.0",
                "docs": "/docs", "health": "/health",
                "endpoints": {"POST /upload-document": "Upload PDF/TXT documents",
                              "POST /classify-query": "Predict query category",
                              "POST /generate-response": "Classify + generate RAG answer",
                              "GET /documents": "List indexed documents",
                              "DELETE /documents/{filename}": "Remove a document"}}

    @app.get("/health", response_model=HealthResponse, tags=["Info"])
    def health():
        clf, r = state["classifier"], state["rag"]
        return HealthResponse(status="ok", classifier_loaded=clf is not None, rag_loaded=r is not None,
                              classifier_model=clf.name if clf else None,
                              embedding_model=cfg.EMBEDDING_MODEL, llm_model=cfg.LLM_MODEL,
                              documents=len(r.documents) if r else 0, chunks=len(r.chunks) if r else 0)

    # ------------------------------------------------------------ documents
    @app.post("/upload-document", response_model=UploadResponse, tags=["Documents"])
    def upload_document(files: List[UploadFile] = File(..., description="One or more PDF/TXT files")):
        """Extract text, chunk, embed and add the document(s) to the FAISS index (re-uploading a name replaces it)."""
        rag = get_rag()
        names = []
        for f in files:                                           # validate everything before processing anything
            name = re.sub(r"[^\w.\- ]", "_", Path(f.filename or "").name)
            if Path(name).suffix.lower() not in cfg.ALLOWED_EXTENSIONS:
                raise HTTPException(400, f"'{f.filename}': only PDF or TXT files are supported")
            names.append(name)

        results = []
        for f, name in zip(files, names):
            dest, size, too_big = cfg.UPLOAD_DIR / name, 0, False
            with open(dest, "wb") as out:
                while chunk := f.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > cfg.MAX_UPLOAD_MB * 1024 * 1024:
                        too_big = True
                        break
                    out.write(chunk)
            if too_big:
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"'{name}' is larger than {cfg.MAX_UPLOAD_MB} MB")
            try:
                pages = extract_pages(dest)
                if not pages:
                    raise ValueError("no extractable text (scanned PDF?)")
                n_chunks = rag.add_document(name, pages)
                results.append(UploadResult(filename=name, status="indexed", pages=len(pages), chunks=n_chunks))
                log.info("Indexed %s: %d pages, %d chunks", name, len(pages), n_chunks)
            except Exception as e:  # noqa: BLE001
                dest.unlink(missing_ok=True)
                results.append(UploadResult(filename=name, status="failed", error=str(e)))
        if all(r.status == "failed" for r in results):
            raise HTTPException(422, "; ".join(f"{r.filename}: {r.error}" for r in results))
        return UploadResponse(results=results, total_documents=len(rag.documents), total_chunks=len(rag.chunks))

    @app.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
    def list_documents():
        rag = get_rag()
        return DocumentsResponse(documents=rag.list_documents(), total_chunks=len(rag.chunks))

    @app.delete("/documents/{filename}", response_model=DocumentsResponse, tags=["Documents"])
    def delete_document(filename: str):
        rag = get_rag()
        if not rag.remove_document(filename):
            raise HTTPException(404, f"Document '{filename}' not found")
        (cfg.UPLOAD_DIR / Path(filename).name).unlink(missing_ok=True)
        return DocumentsResponse(documents=rag.list_documents(), total_chunks=len(rag.chunks))

    # ------------------------------------------------------------ AI endpoints
    @app.post("/classify-query", response_model=ClassificationResponse, tags=["AI"])
    def classify_query(body: QueryRequest):
        """Predict the category of a customer query with a confidence score."""
        return get_classifier().predict(body.query)

    @app.post("/generate-response", response_model=GenerateResponse, tags=["AI"])
    def generate_response(body: GenerateRequest):
        """Classify the query, retrieve relevant document sections and generate a grounded answer."""
        start = time.perf_counter()
        cls = get_classifier().predict(body.query)
        res = get_rag().answer(body.query, body.top_k)
        return GenerateResponse(query=body.query, category=cls["category"], confidence=cls["confidence"],
                                probabilities=cls["probabilities"],
                                processing_time_ms=round((time.perf_counter() - start) * 1000, 1), **res)

    return app


app = create_app()
