#!/usr/bin/env python3
"""
Document Q&A Assistant
Run with: python main.py [--port PORT]
"""

import argparse
import uvicorn
from src.config import get_settings


def main():
    parser = argparse.ArgumentParser(description="Document Q&A Assistant Server")
    parser.add_argument("--host", default=None, help="Host to bind to")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()
    
    settings = get_settings()
    host = args.host or settings.host
    port = args.port or settings.port
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Document Q&A Assistant with Citations              ║
║              and Human Handoff                               ║
╠══════════════════════════════════════════════════════════════╣
║  Web UI:    http://{host}:{port}                             
║  API Docs:  http://{host}:{port}/docs                        
║  Health:    http://{host}:{port}/health                      
╠══════════════════════════════════════════════════════════════╣
║  LLM Provider:       {settings.llm_provider:<15}                        
║  Embedding Provider: {settings.embedding_provider:<15}                        
║  Confidence Threshold: {settings.confidence_threshold:.0%}                            
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()
