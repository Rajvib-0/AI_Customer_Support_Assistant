"""Module 2 - RAG: document extraction, section-aware chunking, FAISS retrieval, grounded generation."""
import json
import logging
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List

import faiss
import numpy as np
import pymupdf

from . import config as cfg

log = logging.getLogger("rag")

# ------------------------------------------------------------------ extraction
def extract_pages(path) -> List[dict]:
    """Return [{'page': int, 'text': str}] for a PDF or TXT file."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        with pymupdf.open(str(path)) as pdf:
            pages = [{"page": i + 1, "text": p.get_text("text")} for i, p in enumerate(pdf)]
    elif ext == ".txt":
        pages = [{"page": 1, "text": path.read_text(encoding="utf-8", errors="ignore")}]
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return [p for p in pages if p["text"].strip()]


# ------------------------------------------------------------------ cleaning & chunking
BULLET = re.compile(r"^\s*[•●▪◦·\-\*]\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def norm_line(line: str) -> str:
    line = re.sub(r"\s+", " ", line).strip()
    # PDF font quirk: the rupee sign is extracted as "■" or as a stray "I" before an amount ("I54,999").
    line = line.replace("■", cfg.CURRENCY_SYMBOL)
    return re.sub(r"(?<![A-Za-z0-9])I(?=\d{1,3}(?:,\d{3})+\b|\d{3,}\b)", cfg.CURRENCY_SYMBOL, line)


def find_boilerplate(documents: List[dict], min_pages: int = 3, min_fraction: float = 0.6) -> set:
    """Lines repeated on most pages (running headers/footers) are not content."""
    counts, total_pages = Counter(), sum(len(d["pages"]) for d in documents)
    for doc in documents:
        for page in doc["pages"]:
            for line in {norm_line(l).lower() for l in page["text"].splitlines() if l.strip()}:
                counts[line] += 1
    limit = max(min_pages, min_fraction * total_pages)
    return {line for line, c in counts.items() if c >= limit}


def is_noise(line: str, boilerplate: set) -> bool:
    low = line.lower()
    return (low in boilerplate
            or re.fullmatch(r"(page\s*)?\d+(\s*(of|/)\s*\d+)?", low) is not None
            or " | " in line)


def is_heading(line: str, prev_item) -> bool:
    words = line.split()
    if not (1 <= len(words) <= 9) or line[-1] in ".!?,;:":
        return False
    return prev_item is None or prev_item[-1] in ".!?:"       # otherwise it is a wrapped line


def parse_sections(pages: List[dict], boilerplate: set) -> List[dict]:
    sections, cur = [], None
    for page in pages:
        for raw in page["text"].splitlines():
            line = norm_line(raw)
            if not line or is_noise(line, boilerplate):
                continue
            is_bullet = bool(BULLET.match(line))
            line = BULLET.sub("", line)
            prev = cur["items"][-1] if cur and cur["items"] else None
            if not is_bullet and is_heading(line, prev):
                cur = {"heading": line, "page": page["page"], "items": []}
                sections.append(cur)
                continue
            if cur is None:
                cur = {"heading": "Introduction", "page": page["page"], "items": []}
                sections.append(cur)
            if is_bullet or prev is None or prev[-1] in ".!?:":
                cur["items"].append(line)
            else:
                cur["items"][-1] += " " + line                    # wrapped line -> continue previous item
    return [s for s in sections if s["items"]]


def doc_title(source: str) -> str:
    return Path(source).stem.replace("_", " ").replace("-", " ").title()


def build_chunks(doc: dict, boilerplate: set) -> List[dict]:
    title, chunks, buf = doc_title(doc["source"]), [], []
    max_words, min_words = cfg.CHUNK_MAX_WORDS, cfg.CHUNK_MIN_WORDS

    def add(text, heading, page):
        chunks.append({"source": doc["source"], "page": page, "section": heading,
                       "text": text, "embed_text": f"{title}. {text}"})

    def flush():
        if buf:
            add(" ".join(b["text"] for b in buf), " / ".join(b["heading"] for b in buf), buf[0]["page"])
            buf.clear()

    for sec in parse_sections(doc["pages"], boilerplate):
        piece = f"{sec['heading']}: " + " ".join(sec["items"])
        n_words = len(piece.split())
        if n_words > max_words:                                   # long section -> split by sentences
            flush()
            units = [s for item in sec["items"] for s in SENT_SPLIT.split(item)]
            group, count = [], 0
            for u in units:
                if group and count + len(u.split()) > max_words:
                    add(f"{sec['heading']}: " + " ".join(group), sec["heading"], sec["page"])
                    group, count = [], 0
                group.append(u)
                count += len(u.split())
            if group:
                add(f"{sec['heading']}: " + " ".join(group), sec["heading"], sec["page"])
            continue
        buf_words = sum(len(b["text"].split()) for b in buf)
        if buf and (buf_words >= min_words or buf_words + n_words > max_words):
            flush()                                               # keep distinct sections apart
        buf.append({"text": piece, "heading": sec["heading"], "page": sec["page"]})
    flush()
    return chunks


# ------------------------------------------------------------------ answer handling
NO_ANSWER_MESSAGE = ("I could not find this information in the available documents. "
                     "Please contact customer support for help.")
NO_DOCUMENTS_MESSAGE = "No documents have been uploaded yet. Please upload a PDF or TXT knowledge-base document first."

REFUSAL_STARTS = ("unanswerable", "i don't know", "i do not know", "the information is not",
                  "information is not available", "not mentioned", "no information", "not specified in the")

NOT_SPECIFIED = re.compile(r"\b(does not|doesn't|do not|did not)\s+(specify|provide|define|contain|include|mention|cover)\b"
                           r"|\bnot\s+(specified|provided|defined|mentioned|covered)\b", re.I)


def is_refusal(answer: str) -> bool:
    a = answer.lower().strip(" .\n")
    return len(a) < 2 or a.startswith(REFUSAL_STARTS)


def says_not_specified(answer: str) -> bool:
    return bool(NOT_SPECIFIED.search(SENT_SPLIT.split(answer.strip())[0]))


def build_prompt(query: str, hits: List[dict]) -> str:
    context = "\n\n".join(f"[{h['source']}, page {h['page']}] {h['text']}" for h in hits)
    return ("Answer the question using only the context below. Give a short, complete answer.\n"
            "If the context does not contain the answer, reply with exactly: unanswerable\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:")


# ------------------------------------------------------------------ the service
class RAGService:
    def __init__(self, embedder, llm_generate: Callable[[str], str], docs_file=cfg.DOCS_FILE,
                 min_similarity: float = cfg.MIN_SIMILARITY):
        self.embedder, self.llm_generate = embedder, llm_generate
        self.docs_file, self.min_similarity = Path(docs_file), min_similarity
        self._lock, self._llm_lock = threading.RLock(), threading.Lock()
        self.documents: Dict[str, List[dict]] = {}
        self.chunks: List[dict] = []
        self.index = None
        self._cache: Dict[str, np.ndarray] = {}                  # embed_text -> vector (avoid re-embedding)
        if self.docs_file.exists():                              # restore state after a server restart
            self.documents = json.loads(self.docs_file.read_text(encoding="utf-8"))
            self._rebuild()
            log.info("Restored %d documents / %d chunks", len(self.documents), len(self.chunks))

    # ---- index management
    def _rebuild(self, documents: Dict[str, List[dict]] = None):
        documents = self.documents if documents is None else documents
        docs = [{"source": s, "pages": p} for s, p in documents.items()]
        boilerplate = find_boilerplate(docs)
        chunks = []
        for doc in docs:
            for ch in build_chunks(doc, boilerplate):
                ch["chunk_id"] = len(chunks)
                chunks.append(ch)
        index = None
        if chunks:
            missing = sorted({c["embed_text"] for c in chunks} - self._cache.keys())
            if missing:
                vecs = np.asarray(self.embedder.encode(missing, convert_to_numpy=True, normalize_embeddings=True,
                                                       show_progress_bar=False), dtype="float32")
                self._cache.update(zip(missing, vecs))
            matrix = np.vstack([self._cache[c["embed_text"]] for c in chunks]).astype("float32")
            index = faiss.IndexFlatIP(matrix.shape[1])           # inner product on normalised vectors = cosine
            index.add(matrix)
        return chunks, index

    def _commit(self, documents, chunks, index):
        self.documents, self.chunks, self.index = documents, chunks, index
        self.docs_file.parent.mkdir(parents=True, exist_ok=True)
        self.docs_file.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")

    def add_document(self, source: str, pages: List[dict]) -> int:
        """Index (or replace) a document. Returns its chunk count; raises ValueError if it has no usable text."""
        with self._lock:
            documents = {**self.documents, source: pages}
            chunks, index = self._rebuild(documents)
            n = sum(c["source"] == source for c in chunks)
            if n == 0:
                raise ValueError("no usable text could be extracted")
            self._commit(documents, chunks, index)
            return n

    def remove_document(self, source: str) -> bool:
        with self._lock:
            if source not in self.documents:
                return False
            documents = {k: v for k, v in self.documents.items() if k != source}
            chunks, index = self._rebuild(documents)
            self._commit(documents, chunks, index)
            return True

    def list_documents(self) -> Dict[str, dict]:
        with self._lock:
            return {s: {"pages": len(p), "chunks": sum(c["source"] == s for c in self.chunks)}
                    for s, p in self.documents.items()}

    # ---- retrieval & generation
    def retrieve(self, query: str, top_k: int = cfg.TOP_K) -> List[dict]:
        with self._lock:
            if self.index is None:
                return []
            q = np.asarray(self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True),
                           dtype="float32")
            scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
            return [{**self.chunks[i], "similarity": float(s)} for s, i in zip(scores[0], ids[0]) if i != -1]

    def select_context(self, hits: List[dict]) -> List[dict]:
        if not hits or hits[0]["similarity"] < self.min_similarity:
            return []
        best = hits[0]["similarity"]
        keep = [h for h in hits if h["similarity"] >= max(self.min_similarity, best - cfg.RELATIVE_MARGIN)]
        return keep[:cfg.CONTEXT_CHUNKS]

    def answer(self, query: str, top_k: int = cfg.TOP_K) -> dict:
        if not self.documents:
            return {"status": "no_documents", "answer": NO_DOCUMENTS_MESSAGE, "answerable": False,
                    "top_score": 0.0, "sources": []}
        hits = self.retrieve(query, top_k)
        top_score = round(hits[0]["similarity"], 4) if hits else 0.0
        context = self.select_context(hits)
        declined = {"status": "not_in_documents", "answer": NO_ANSWER_MESSAGE, "answerable": False,
                    "top_score": top_score, "sources": []}
        if not context:                                           # weak retrieval -> do not call the LLM
            return declined
        with self._llm_lock:
            answer = self.llm_generate(build_prompt(query, context)).strip()
        if is_refusal(answer):
            return declined
        sources = [{"document": h["source"], "page": h["page"], "section": h["section"],
                    "similarity": round(h["similarity"], 4)} for h in context]
        if says_not_specified(answer):                            # the policy explicitly leaves this open
            return {"status": "not_specified", "answerable": False, "top_score": top_score, "sources": sources,
                    "answer": answer.rstrip(". ") + ". Please contact customer support for details."}
        return {"status": "answered", "answer": answer, "answerable": True, "top_score": top_score,
                "sources": sources}
