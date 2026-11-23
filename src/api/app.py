from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import logging

from ..config import get_settings
from ..rag import EmbeddingService, Generator, RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)


class QueryResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    citations: list[dict]
    confidence: float
    escalated: bool
    escalation_reason: Optional[str]
    model_used: str
    tokens_used: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    llm_provider: str
    llm_configured: bool
    embedding_provider: str


class IndexRequest(BaseModel):
    directory: Optional[str] = None


class IndexResponse(BaseModel):
    documents_indexed: int
    total_documents: int


def create_app() -> FastAPI:
    settings = get_settings()
    
    app = FastAPI(
        title="Document Q&A Assistant",
        description="RAG-powered Q&A with citations and human escalation",
        version="1.0.0"
    )
    
    embedding_service = EmbeddingService(
        provider=settings.embedding_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_embedding_model,
        local_model=settings.local_embedding_model
    )
    
    generator = Generator(
        provider=settings.llm_provider,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_model,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_model=settings.anthropic_model
    )
    
    pipeline = RAGPipeline(
        embedding_service=embedding_service,
        generator=generator,
        persist_dir=settings.chroma_persist_dir,
        confidence_threshold=settings.confidence_threshold,
        max_chunks=settings.max_chunks,
        escalation_webhook_url=settings.escalation_webhook_url,
        escalation_log_file=settings.escalation_log_file
    )
    
    app.state.pipeline = pipeline
    app.state.settings = settings
    
    base_dir = Path(__file__).parent.parent.parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"
    
    templates_dir.mkdir(exist_ok=True)
    static_dir.mkdir(exist_ok=True)
    
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    templates = Jinja2Templates(directory=str(templates_dir))
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "llm_configured": generator.is_configured,
                "llm_provider": settings.llm_provider if generator.is_configured else "not configured",
                "documents_indexed": pipeline.document_count
            }
        )
    
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            documents_indexed=pipeline.document_count,
            llm_provider=settings.llm_provider,
            llm_configured=generator.is_configured,
            embedding_provider=settings.embedding_provider
        )
    
    @app.post("/api/query", response_model=QueryResponse)
    async def query(request: QueryRequest):
        try:
            result = pipeline.query(request.question)
            return QueryResponse(**result.to_dict())
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/api/index", response_model=IndexResponse)
    async def index_documents(request: IndexRequest):
        directory = request.directory or settings.knowledge_base_dir
        
        if not Path(directory).exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {directory}")
        
        try:
            new_docs = pipeline.index_documents(
                directory=directory,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            return IndexResponse(
                documents_indexed=new_docs,
                total_documents=pipeline.document_count
            )
        except Exception as e:
            logger.error(f"Indexing error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/escalations")
    async def get_escalations(limit: int = 50):
        return {"escalations": pipeline.get_pending_escalations(limit)}
    
    @app.on_event("startup")
    async def startup():
        kb_dir = Path(settings.knowledge_base_dir)
        if kb_dir.exists() and pipeline.document_count == 0:
            logger.info(f"Auto-indexing documents from {kb_dir}")
            new_docs = pipeline.index_documents(
                directory=str(kb_dir),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap
            )
            logger.info(f"Indexed {new_docs} document chunks")
    
    return app


app = create_app()
