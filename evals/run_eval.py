#!/usr/bin/env python3
"""
Evaluation script for Document Q&A Assistant.

Runs the evaluation question set and reports:
- Answer correctness (expected content present)
- Citation accuracy (expected sources cited)
- Escalation rate and correctness
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_settings
from src.rag import EmbeddingService, Generator, RAGPipeline


console = Console()


@dataclass
class EvalResult:
    question_id: str
    question: str
    category: str
    difficulty: str
    
    actual_answer: str
    actual_escalated: bool
    actual_confidence: float
    actual_sources: list[str]
    
    expected_escalate: bool
    expected_answer_contains: list[str]
    expected_sources: list[str]
    
    answer_correct: bool = False
    citation_correct: bool = False
    escalation_correct: bool = False
    
    missing_content: list[str] = field(default_factory=list)
    missing_sources: list[str] = field(default_factory=list)
    notes: str = ""


def load_eval_questions(path: str = "evals/eval_questions.yaml") -> list[dict]:
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["questions"]


def evaluate_answer(result: EvalResult) -> EvalResult:
    if result.expected_escalate:
        result.escalation_correct = result.actual_escalated
        if result.actual_escalated:
            result.answer_correct = True
            result.citation_correct = True
        else:
            result.answer_correct = False
            result.citation_correct = False
            result.notes = "Should have escalated but didn't"
        return result
    
    result.escalation_correct = not result.actual_escalated
    
    if result.actual_escalated:
        result.answer_correct = False
        result.citation_correct = False
        result.notes = "Incorrectly escalated"
        return result
    
    answer_lower = result.actual_answer.lower()
    missing = []
    for expected in result.expected_answer_contains:
        if expected.lower() not in answer_lower:
            missing.append(expected)
    
    result.missing_content = missing
    result.answer_correct = len(missing) == 0
    
    actual_sources_lower = [s.lower() for s in result.actual_sources]
    missing_sources = []
    for expected_source in result.expected_sources:
        found = any(expected_source.lower() in actual for actual in actual_sources_lower)
        if not found:
            missing_sources.append(expected_source)
    
    result.missing_sources = missing_sources
    result.citation_correct = len(missing_sources) == 0
    
    return result


def run_evaluation(pipeline: RAGPipeline, questions: list[dict]) -> list[EvalResult]:
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running evaluation...", total=len(questions))
        
        for q in questions:
            progress.update(task, description=f"Testing: {q['id']}")
            
            response = pipeline.query(q["question"])
            
            result = EvalResult(
                question_id=q["id"],
                question=q["question"],
                category=q["category"],
                difficulty=q["difficulty"],
                actual_answer=response.answer,
                actual_escalated=response.escalated,
                actual_confidence=response.confidence,
                actual_sources=[c.section for c in response.citations],
                expected_escalate=q["should_escalate"],
                expected_answer_contains=q.get("expected_answer_contains", []),
                expected_sources=q.get("expected_sources", [])
            )
            
            result = evaluate_answer(result)
            results.append(result)
            
            progress.advance(task)
    
    return results


def print_results(results: list[EvalResult]):
    correct_answers = sum(1 for r in results if r.answer_correct)
    correct_citations = sum(1 for r in results if r.citation_correct)
    correct_escalations = sum(1 for r in results if r.escalation_correct)
    
    actual_escalations = sum(1 for r in results if r.actual_escalated)
    expected_escalations = sum(1 for r in results if r.expected_escalate)
    
    total = len(results)
    
    console.print("\n")
    console.print(Panel(
        f"""[bold]Evaluation Results[/bold]

📊 [cyan]Overall Metrics[/cyan]
   • Answer Correctness:    {correct_answers}/{total} ({correct_answers/total*100:.1f}%)
   • Citation Accuracy:     {correct_citations}/{total} ({correct_citations/total*100:.1f}%)
   • Escalation Accuracy:   {correct_escalations}/{total} ({correct_escalations/total*100:.1f}%)

📤 [yellow]Escalation Stats[/yellow]
   • Expected Escalations:  {expected_escalations}
   • Actual Escalations:    {actual_escalations}
   • Escalation Rate:       {actual_escalations/total*100:.1f}%
""",
        title="Summary",
        border_style="blue"
    ))
    
    by_category = {}
    for r in results:
        if r.category not in by_category:
            by_category[r.category] = []
        by_category[r.category].append(r)
    
    table = Table(title="Results by Category")
    table.add_column("Category", style="cyan")
    table.add_column("Questions", justify="right")
    table.add_column("Answer %", justify="right")
    table.add_column("Citation %", justify="right")
    table.add_column("Escalation %", justify="right")
    
    for cat, cat_results in sorted(by_category.items()):
        n = len(cat_results)
        ans = sum(1 for r in cat_results if r.answer_correct)
        cit = sum(1 for r in cat_results if r.citation_correct)
        esc = sum(1 for r in cat_results if r.escalation_correct)
        
        table.add_row(
            cat,
            str(n),
            f"{ans/n*100:.0f}%",
            f"{cit/n*100:.0f}%",
            f"{esc/n*100:.0f}%"
        )
    
    console.print(table)
    
    failures = [r for r in results if not (r.answer_correct and r.citation_correct and r.escalation_correct)]
    
    if failures:
        console.print("\n[bold red]Failed Questions:[/bold red]\n")
        
        fail_table = Table()
        fail_table.add_column("ID", style="cyan")
        fail_table.add_column("Question")
        fail_table.add_column("Issues", style="red")
        fail_table.add_column("Confidence")
        
        for r in failures:
            issues = []
            if not r.answer_correct:
                if r.missing_content:
                    issues.append(f"Missing: {', '.join(r.missing_content[:2])}")
                else:
                    issues.append("Answer incorrect")
            if not r.citation_correct:
                issues.append(f"Missing sources: {', '.join(r.missing_sources[:2])}")
            if not r.escalation_correct:
                issues.append("Wrong escalation")
            if r.notes:
                issues.append(r.notes)
            
            fail_table.add_row(
                r.question_id,
                r.question[:40] + "..." if len(r.question) > 40 else r.question,
                "\n".join(issues),
                f"{r.actual_confidence:.0%}"
            )
        
        console.print(fail_table)
    
    return {
        "total": total,
        "answer_correct": correct_answers,
        "citation_correct": correct_citations,
        "escalation_correct": correct_escalations,
        "actual_escalations": actual_escalations,
        "expected_escalations": expected_escalations,
        "answer_accuracy": correct_answers / total,
        "citation_accuracy": correct_citations / total,
        "escalation_accuracy": correct_escalations / total,
        "escalation_rate": actual_escalations / total
    }


def save_results(results: list[EvalResult], metrics: dict, output_dir: str = "evals/results"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    detailed = []
    for r in results:
        detailed.append({
            "question_id": r.question_id,
            "question": r.question,
            "category": r.category,
            "difficulty": r.difficulty,
            "actual_answer": r.actual_answer[:500],
            "actual_escalated": r.actual_escalated,
            "actual_confidence": r.actual_confidence,
            "actual_sources": r.actual_sources,
            "answer_correct": r.answer_correct,
            "citation_correct": r.citation_correct,
            "escalation_correct": r.escalation_correct,
            "missing_content": r.missing_content,
            "missing_sources": r.missing_sources,
            "notes": r.notes
        })
    
    output = {
        "timestamp": timestamp,
        "metrics": metrics,
        "results": detailed
    }
    
    output_path = Path(output_dir) / f"eval_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
    return output_path


def main():
    console.print(Panel(
        "[bold]Document Q&A Assistant - Evaluation Runner[/bold]\n\n"
        "This script evaluates the RAG pipeline against a set of test questions,\n"
        "measuring answer correctness, citation accuracy, and escalation behavior.",
        border_style="blue"
    ))
    
    settings = get_settings()
    
    console.print("\n[blue]Initializing pipeline...[/blue]")
    
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
        max_chunks=settings.max_chunks
    )
    
    if pipeline.document_count == 0:
        console.print("[yellow]Indexing knowledge base...[/yellow]")
        indexed = pipeline.index_documents(settings.knowledge_base_dir)
        console.print(f"[green]Indexed {indexed} document chunks[/green]")
    else:
        console.print(f"[green]Using existing index ({pipeline.document_count} chunks)[/green]")
    
    console.print(f"[dim]LLM Provider: {settings.llm_provider} (configured: {generator.is_configured})[/dim]")
    
    questions = load_eval_questions()
    console.print(f"\n[blue]Running {len(questions)} evaluation questions...[/blue]\n")
    
    results = run_evaluation(pipeline, questions)
    
    metrics = print_results(results)
    
    save_results(results, metrics)
    
    all_pass = metrics["answer_accuracy"] >= 0.7 and metrics["escalation_accuracy"] >= 0.8
    
    if all_pass:
        console.print("\n[bold green]✓ Evaluation PASSED[/bold green]")
        return 0
    else:
        console.print("\n[bold red]✗ Evaluation FAILED[/bold red]")
        console.print("[dim]Target: 70% answer accuracy, 80% escalation accuracy[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
