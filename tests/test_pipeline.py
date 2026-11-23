"""Tests for RAG pipeline and escalation logic."""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rag.pipeline import RAGPipeline, QueryResponse, Citation
from src.rag.embeddings import EmbeddingService
from src.rag.generator import Generator, GenerationResult
from src.rag.retriever import RetrievalResult
from src.rag.document_processor import DocumentChunk


class TestRAGPipeline:
    @pytest.fixture
    def mock_embedding_service(self):
        with patch('src.rag.embeddings.LocalEmbedding') as mock:
            instance = mock.return_value
            instance.dimension = 384
            instance.embed.return_value = [[0.1] * 384]
            instance.embed_query.return_value = [0.1] * 384
            
            service = EmbeddingService(provider="local", local_model="all-MiniLM-L6-v2")
            return service
    
    @pytest.fixture
    def mock_generator(self):
        generator = Generator(provider="openai", openai_api_key="")
        return generator
    
    @pytest.fixture
    def temp_dirs(self):
        with tempfile.TemporaryDirectory() as chroma_dir:
            with tempfile.TemporaryDirectory() as kb_dir:
                test_doc = Path(kb_dir) / "test.md"
                test_doc.write_text("""# Test Document

## Test Section

This is test content about remote work policy.
Core hours are 10 AM to 3 PM.
""")
                yield chroma_dir, kb_dir
    
    def test_out_of_scope_detection(self, mock_embedding_service, mock_generator, temp_dirs):
        chroma_dir, kb_dir = temp_dirs
        
        pipeline = RAGPipeline(
            embedding_service=mock_embedding_service,
            generator=mock_generator,
            persist_dir=chroma_dir,
            escalation_log_file=str(Path(chroma_dir) / "escalations.jsonl")
        )
        
        result = pipeline._check_out_of_scope("What is your competitor pricing?")
        assert result is not None
        assert "competitor" in result.lower()
        
        result = pipeline._check_out_of_scope("What are the working hours?")
        assert result is None
    
    def test_escalation_response_format(self, mock_embedding_service, mock_generator, temp_dirs):
        chroma_dir, kb_dir = temp_dirs
        
        pipeline = RAGPipeline(
            embedding_service=mock_embedding_service,
            generator=mock_generator,
            persist_dir=chroma_dir,
            escalation_log_file=str(Path(chroma_dir) / "escalations.jsonl")
        )
        
        response = pipeline._create_escalation_response(
            request_id="test123",
            question="Test question",
            reason="Test reason",
            retrieval_results=[]
        )
        
        assert response.escalated is True
        assert response.escalation_reason == "Test reason"
        assert "human" in response.answer.lower()
        assert response.citations == []
    
    def test_context_building(self, mock_embedding_service, mock_generator, temp_dirs):
        chroma_dir, kb_dir = temp_dirs
        
        pipeline = RAGPipeline(
            embedding_service=mock_embedding_service,
            generator=mock_generator,
            persist_dir=chroma_dir
        )
        
        chunk1 = DocumentChunk(
            content="Content one",
            source_file="file1.md",
            source_section="Section One",
            chunk_index=0,
            start_char=0,
            end_char=11
        )
        
        chunk2 = DocumentChunk(
            content="Content two",
            source_file="file2.md",
            source_section="Section Two",
            chunk_index=0,
            start_char=0,
            end_char=11
        )
        
        results = [
            RetrievalResult(chunk=chunk1, score=0.9),
            RetrievalResult(chunk=chunk2, score=0.8)
        ]
        
        context = pipeline._build_context(results)
        
        assert "Section One" in context
        assert "Section Two" in context
        assert "Content one" in context
        assert "Content two" in context


class TestQueryResponse:
    def test_to_dict(self):
        citations = [
            Citation(
                source_file="test.md",
                section="Test Section",
                relevance_score=0.85,
                snippet="Short snippet"
            )
        ]
        
        response = QueryResponse(
            request_id="abc123",
            question="Test question?",
            answer="Test answer",
            citations=citations,
            confidence=0.85,
            escalated=False,
            escalation_reason=None,
            model_used="gpt-4o-mini",
            tokens_used=150
        )
        
        result = response.to_dict()
        
        assert result["request_id"] == "abc123"
        assert result["confidence"] == 0.85
        assert result["escalated"] is False
        assert len(result["citations"]) == 1
        assert result["citations"][0]["section"] == "Test Section"
    
    def test_long_snippet_truncation(self):
        long_snippet = "x" * 500
        citations = [
            Citation(
                source_file="test.md",
                section="Test",
                relevance_score=0.9,
                snippet=long_snippet
            )
        ]
        
        response = QueryResponse(
            request_id="test",
            question="q",
            answer="a",
            citations=citations,
            confidence=0.9,
            escalated=False,
            escalation_reason=None,
            model_used="test",
            tokens_used=0
        )
        
        result = response.to_dict()
        
        assert len(result["citations"][0]["snippet"]) < len(long_snippet)
        assert result["citations"][0]["snippet"].endswith("...")
