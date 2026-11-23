#!/usr/bin/env python3
"""Command-line interface for Document Q&A Assistant."""

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from .config import get_settings
from .rag import EmbeddingService, Generator, RAGPipeline

console = Console()


def create_pipeline() -> RAGPipeline:
    settings = get_settings()
    
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
    
    return RAGPipeline(
        embedding_service=embedding_service,
        generator=generator,
        persist_dir=settings.chroma_persist_dir,
        confidence_threshold=settings.confidence_threshold,
        max_chunks=settings.max_chunks,
        escalation_webhook_url=settings.escalation_webhook_url,
        escalation_log_file=settings.escalation_log_file
    )


def cmd_query(args):
    pipeline = create_pipeline()
    settings = get_settings()
    
    if pipeline.document_count == 0:
        console.print("[yellow]No documents indexed. Indexing knowledge base...[/yellow]")
        indexed = pipeline.index_documents(settings.knowledge_base_dir)
        console.print(f"[green]Indexed {indexed} document chunks[/green]\n")
    
    result = pipeline.query(args.question)
    
    if result.escalated:
        console.print(Panel(
            f"[bold yellow]⚠️ ESCALATED TO HUMAN[/bold yellow]\n\n"
            f"[yellow]Reason:[/yellow] {result.escalation_reason}\n\n"
            f"{result.answer}",
            title="Human Handoff Required",
            border_style="yellow"
        ))
    else:
        console.print(Panel(
            Markdown(result.answer),
            title=f"Answer (confidence: {result.confidence:.0%})",
            border_style="green" if result.confidence >= 0.8 else "yellow"
        ))
        
        if result.citations:
            table = Table(title="Citations")
            table.add_column("Section", style="cyan")
            table.add_column("Source", style="dim")
            table.add_column("Relevance", justify="right")
            
            for citation in result.citations:
                table.add_row(
                    citation.section,
                    Path(citation.source_file).name,
                    f"{citation.relevance_score:.0%}"
                )
            
            console.print(table)
    
    console.print(f"\n[dim]Request ID: {result.request_id} | Model: {result.model_used} | Tokens: {result.tokens_used}[/dim]")


def cmd_index(args):
    pipeline = create_pipeline()
    directory = args.directory or get_settings().knowledge_base_dir
    
    console.print(f"[blue]Indexing documents from {directory}...[/blue]")
    indexed = pipeline.index_documents(directory)
    console.print(f"[green]✓ Indexed {indexed} new chunks[/green]")
    console.print(f"[dim]Total documents in index: {pipeline.document_count}[/dim]")


def cmd_escalations(args):
    pipeline = create_pipeline()
    escalations = pipeline.get_pending_escalations(args.limit)
    
    if not escalations:
        console.print("[dim]No pending escalations[/dim]")
        return
    
    table = Table(title=f"Pending Escalations ({len(escalations)})")
    table.add_column("ID", style="cyan")
    table.add_column("Question")
    table.add_column("Reason", style="yellow")
    table.add_column("Time", style="dim")
    
    for esc in escalations:
        table.add_row(
            esc["request_id"],
            esc["question"][:50] + "..." if len(esc["question"]) > 50 else esc["question"],
            esc["reason"][:40] + "..." if len(esc["reason"]) > 40 else esc["reason"],
            esc["timestamp"][:19]
        )
    
    console.print(table)


def cmd_interactive(args):
    pipeline = create_pipeline()
    settings = get_settings()
    
    if pipeline.document_count == 0:
        console.print("[yellow]Indexing knowledge base...[/yellow]")
        indexed = pipeline.index_documents(settings.knowledge_base_dir)
        console.print(f"[green]Indexed {indexed} document chunks[/green]\n")
    
    console.print(Panel(
        "[bold]Document Q&A Assistant[/bold]\n\n"
        "Ask questions about NovaTech policies, products, and FAQs.\n"
        "Type 'exit' or 'quit' to leave.\n"
        f"\n[dim]Documents indexed: {pipeline.document_count} | LLM: {settings.llm_provider}[/dim]",
        border_style="blue"
    ))
    
    while True:
        try:
            question = console.input("\n[bold blue]You:[/bold blue] ").strip()
            
            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                console.print("[dim]Goodbye![/dim]")
                break
            
            with console.status("Thinking..."):
                result = pipeline.query(question)
            
            if result.escalated:
                console.print(f"\n[bold yellow]Assistant (ESCALATED):[/bold yellow]")
                console.print(f"[yellow]{result.answer}[/yellow]")
            else:
                console.print(f"\n[bold green]Assistant:[/bold green]")
                console.print(Markdown(result.answer))
                
                if result.citations:
                    sources = ", ".join(set(c.section for c in result.citations[:3]))
                    console.print(f"\n[dim]Sources: {sources} | Confidence: {result.confidence:.0%}[/dim]")
        
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted. Type 'exit' to quit.[/dim]")
        except EOFError:
            break


def main():
    parser = argparse.ArgumentParser(
        description="Document Q&A Assistant - RAG with citations and human handoff"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    query_parser = subparsers.add_parser("query", help="Ask a single question")
    query_parser.add_argument("question", help="The question to ask")
    query_parser.set_defaults(func=cmd_query)
    
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("-d", "--directory", help="Directory to index")
    index_parser.set_defaults(func=cmd_index)
    
    esc_parser = subparsers.add_parser("escalations", help="View pending escalations")
    esc_parser.add_argument("-l", "--limit", type=int, default=20, help="Max results")
    esc_parser.set_defaults(func=cmd_escalations)
    
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    if args.command is None:
        args.func = cmd_interactive
    
    args.func(args)


if __name__ == "__main__":
    main()
