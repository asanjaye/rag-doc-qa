from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
import uuid

from .document_processor import DocumentProcessor, DocumentChunk
from .embeddings import EmbeddingService
from .retriever import Retriever, RetrievalResult
from .generator import Generator, GenerationResult
from .escalation import EscalationHandler, EscalationRequest


@dataclass
class Citation:
    source_file: str
    section: str
    relevance_score: float
    snippet: str
    
    def __str__(self) -> str:
        return f"[{self.section}] (relevance: {self.relevance_score:.2f})"


@dataclass
class QueryResponse:
    request_id: str
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    escalated: bool
    escalation_reason: Optional[str]
    model_used: str
    tokens_used: int
    retrieval_results: list[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "question": self.question,
            "answer": self.answer,
            "citations": [
                {
                    "source_file": c.source_file,
                    "section": c.section,
                    "relevance_score": c.relevance_score,
                    "snippet": c.snippet[:200] + "..." if len(c.snippet) > 200 else c.snippet
                }
                for c in self.citations
            ],
            "confidence": self.confidence,
            "escalated": self.escalated,
            "escalation_reason": self.escalation_reason,
            "model_used": self.model_used,
            "tokens_used": self.tokens_used
        }


class RAGPipeline:
    ESCALATION_RESPONSE = """I don't have enough confident information in my knowledge base to answer this question accurately. This question has been escalated to a human expert who will follow up with you.

**Question logged for human review:** {question}
**Reason:** {reason}

A team member will respond within 24 business hours."""

    OUT_OF_SCOPE_KEYWORDS = [
        "competitor", "legal advice", "medical advice", "investment advice",
        "password", "hack", "bypass", "illegal"
    ]
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        generator: Generator,
        persist_dir: str = "./data/chroma",
        confidence_threshold: float = 0.65,
        max_chunks: int = 5,
        escalation_webhook_url: str = "",
        escalation_log_file: str = "escalations.jsonl"
    ):
        self.retriever = Retriever(embedding_service, persist_dir)
        self.generator = generator
        self.confidence_threshold = confidence_threshold
        self.max_chunks = max_chunks
        self.escalation_handler = EscalationHandler(
            webhook_url=escalation_webhook_url,
            log_file=escalation_log_file
        )
    
    def index_documents(self, directory: str, chunk_size: int = 500, chunk_overlap: int = 50) -> int:
        processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = processor.load_documents(directory)
        indexed = self.retriever.index_documents(chunks)
        return indexed
    
    def query(self, question: str) -> QueryResponse:
        request_id = str(uuid.uuid4())[:8]
        
        out_of_scope_reason = self._check_out_of_scope(question)
        if out_of_scope_reason:
            return self._create_escalation_response(
                request_id=request_id,
                question=question,
                reason=out_of_scope_reason,
                retrieval_results=[]
            )
        
        retrieval_results = self.retriever.retrieve(
            query=question,
            top_k=self.max_chunks,
            threshold=0.0
        )
        
        if not retrieval_results:
            return self._create_escalation_response(
                request_id=request_id,
                question=question,
                reason="No relevant documents found in knowledge base",
                retrieval_results=[]
            )
        
        max_confidence = max(r.score for r in retrieval_results)
        avg_confidence = sum(r.score for r in retrieval_results) / len(retrieval_results)
        
        confident_results = [r for r in retrieval_results if r.score >= self.confidence_threshold]
        
        if not confident_results or max_confidence < self.confidence_threshold:
            return self._create_escalation_response(
                request_id=request_id,
                question=question,
                reason=f"Retrieval confidence too low (max: {max_confidence:.2f}, threshold: {self.confidence_threshold})",
                retrieval_results=retrieval_results
            )
        
        context = self._build_context(confident_results)
        generation_result = self.generator.generate(question, context)
        
        citations = [
            Citation(
                source_file=r.chunk.source_file,
                section=r.chunk.source_section,
                relevance_score=r.score,
                snippet=r.chunk.content
            )
            for r in confident_results
        ]
        
        return QueryResponse(
            request_id=request_id,
            question=question,
            answer=generation_result.answer,
            citations=citations,
            confidence=max_confidence,
            escalated=False,
            escalation_reason=None,
            model_used=generation_result.model,
            tokens_used=generation_result.tokens_used,
            retrieval_results=[
                {"section": r.chunk.source_section, "score": r.score}
                for r in retrieval_results
            ]
        )
    
    def _check_out_of_scope(self, question: str) -> Optional[str]:
        question_lower = question.lower()
        for keyword in self.OUT_OF_SCOPE_KEYWORDS:
            if keyword in question_lower:
                return f"Question appears to be out of scope (detected: '{keyword}')"
        return None
    
    def _build_context(self, results: list[RetrievalResult]) -> str:
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Source {i}: {result.chunk.source_section}]\n{result.chunk.content}"
            )
        return "\n\n---\n\n".join(context_parts)
    
    def _create_escalation_response(
        self,
        request_id: str,
        question: str,
        reason: str,
        retrieval_results: list[RetrievalResult]
    ) -> QueryResponse:
        escalation_request = EscalationRequest(
            question=question,
            reason=reason,
            timestamp=datetime.now(timezone.utc).isoformat(),
            retrieval_scores=[r.score for r in retrieval_results],
            top_sources=[r.chunk.source_section for r in retrieval_results[:3]],
            request_id=request_id
        )
        self.escalation_handler.escalate(escalation_request)
        
        answer = self.ESCALATION_RESPONSE.format(
            question=question,
            reason=reason
        )
        
        return QueryResponse(
            request_id=request_id,
            question=question,
            answer=answer,
            citations=[],
            confidence=max([r.score for r in retrieval_results], default=0.0),
            escalated=True,
            escalation_reason=reason,
            model_used="none",
            tokens_used=0,
            retrieval_results=[
                {"section": r.chunk.source_section, "score": r.score}
                for r in retrieval_results
            ]
        )
    
    def get_pending_escalations(self, limit: int = 50) -> list[dict]:
        return [e.to_dict() for e in self.escalation_handler.get_pending_escalations(limit)]
    
    @property
    def document_count(self) -> int:
        return self.retriever.document_count
