"""ragkit — dataset-agnostic, pluggable RAG framework."""

from ragkit.models import Answer, Chunk, Document, EvalExample, Query, ScoredHit
from ragkit.pipeline import RAGPipeline

__version__ = "0.1.0"
__all__ = [
    "Answer",
    "Chunk",
    "Document",
    "EvalExample",
    "Query",
    "RAGPipeline",
    "ScoredHit",
    "__version__",
]
