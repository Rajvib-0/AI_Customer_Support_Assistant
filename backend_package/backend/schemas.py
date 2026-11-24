"""Pydantic request / response models."""
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=1000, description="The customer's question",
                       examples=["How long does an approved refund take?"])

    @field_validator("query")
    @classmethod
    def not_blank(cls, v: str) -> str:
        v = v.strip()
        if len(v) < 2:
            raise ValueError("query must contain at least 2 non-space characters")
        return v


class GenerateRequest(QueryRequest):
    top_k: int = Field(4, ge=1, le=8, description="How many chunks to retrieve")


class ClassificationResponse(BaseModel):
    query: str
    category: str
    confidence: float = Field(..., ge=0, le=1)
    confidence_type: Literal["probability", "softmax_of_margin"]
    probabilities: Dict[str, float]
    model: str


class SourceReference(BaseModel):
    document: str
    page: int
    section: str
    similarity: float


class GenerateResponse(BaseModel):
    query: str
    category: str
    confidence: float
    probabilities: Dict[str, float]
    answer: str
    # answered | not_specified (documents explicitly say so) | not_in_documents | no_documents (nothing uploaded)
    status: Literal["answered", "not_specified", "not_in_documents", "no_documents"]
    answerable: bool
    top_score: float
    sources: List[SourceReference]
    processing_time_ms: float


class UploadResult(BaseModel):
    filename: str
    status: Literal["indexed", "failed"]
    pages: int = 0
    chunks: int = 0
    error: Optional[str] = None


class UploadResponse(BaseModel):
    results: List[UploadResult]
    total_documents: int
    total_chunks: int


class DocumentInfo(BaseModel):
    pages: int
    chunks: int


class DocumentsResponse(BaseModel):
    documents: Dict[str, DocumentInfo]
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    classifier_loaded: bool
    rag_loaded: bool
    classifier_model: Optional[str] = None
    embedding_model: str
    llm_model: str
    documents: int
    chunks: int
