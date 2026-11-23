"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
from pathlib import Path

from src.api.app import create_app
from src.rag.pipeline import QueryResponse, Citation


@pytest.fixture
def test_client():
    with tempfile.TemporaryDirectory() as tmpdir:
        kb_dir = Path(tmpdir) / "kb"
        kb_dir.mkdir()
        
        test_doc = kb_dir / "test.md"
        test_doc.write_text("""# Test

## Section

Test content.
""")
        
        chroma_dir = Path(tmpdir) / "chroma"
        
        with patch.dict('os.environ', {
            'KNOWLEDGE_BASE_DIR': str(kb_dir),
            'CHROMA_PERSIST_DIR': str(chroma_dir),
            'EMBEDDING_PROVIDER': 'local',
            'LLM_PROVIDER': 'openai',
            'OPENAI_API_KEY': '',
            'ESCALATION_LOG_FILE': str(Path(tmpdir) / 'escalations.jsonl')
        }):
            from src.config import get_settings
            get_settings.cache_clear()
            
            app = create_app()
            yield TestClient(app)
            
            get_settings.cache_clear()


class TestHealthEndpoint:
    def test_health_returns_status(self, test_client):
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "documents_indexed" in data
        assert "llm_provider" in data


class TestQueryEndpoint:
    def test_query_requires_question(self, test_client):
        response = test_client.post("/api/query", json={})
        assert response.status_code == 422
    
    def test_query_rejects_empty_question(self, test_client):
        response = test_client.post("/api/query", json={"question": ""})
        assert response.status_code == 422
    
    def test_query_returns_response_structure(self, test_client):
        response = test_client.post(
            "/api/query",
            json={"question": "What is the test content?"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "request_id" in data
        assert "question" in data
        assert "answer" in data
        assert "citations" in data
        assert "confidence" in data
        assert "escalated" in data
        assert "model_used" in data


class TestIndexEndpoint:
    def test_index_returns_counts(self, test_client):
        response = test_client.post("/api/index", json={})
        
        assert response.status_code == 200
        data = response.json()
        
        assert "documents_indexed" in data
        assert "total_documents" in data
    
    def test_index_nonexistent_directory(self, test_client):
        response = test_client.post(
            "/api/index",
            json={"directory": "/nonexistent/path"}
        )
        assert response.status_code == 404


class TestEscalationsEndpoint:
    def test_escalations_returns_list(self, test_client):
        response = test_client.get("/api/escalations")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "escalations" in data
        assert isinstance(data["escalations"], list)
