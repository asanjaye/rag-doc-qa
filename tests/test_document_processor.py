"""Tests for document processing and chunking."""

import pytest
from pathlib import Path
import tempfile

from src.rag.document_processor import DocumentProcessor, DocumentChunk


class TestDocumentChunk:
    def test_chunk_id_generation(self):
        chunk = DocumentChunk(
            content="Test content",
            source_file="test.md",
            source_section="Test Section",
            chunk_index=0,
            start_char=0,
            end_char=12
        )
        assert chunk.chunk_id is not None
        assert "test_0_" in chunk.chunk_id
    
    def test_citation_format(self):
        chunk = DocumentChunk(
            content="Test content",
            source_file="docs/test.md",
            source_section="Test Section",
            chunk_index=0,
            start_char=0,
            end_char=12
        )
        assert chunk.citation == "[Test Section](docs/test.md)"


class TestDocumentProcessor:
    def test_split_by_headers(self):
        processor = DocumentProcessor()
        content = """# Main Title

Introduction text.

## Section One

Section one content.

### Subsection

Subsection content.
"""
        sections = processor._split_by_headers(content)
        
        assert len(sections) >= 3
        section_titles = [s[0] for s in sections]
        assert "Main Title" in section_titles
        assert "Section One" in section_titles
        assert "Subsection" in section_titles
    
    def test_chunk_text_respects_size(self):
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=0)
        
        text = "Short paragraph.\n\n" * 10
        chunks = list(processor._chunk_text(text))
        
        for chunk in chunks:
            assert len(chunk["text"]) <= 100 + 50  # allow some overflow for complete sentences
    
    def test_chunk_text_handles_empty(self):
        processor = DocumentProcessor()
        chunks = list(processor._chunk_text(""))
        assert len(chunks) == 0
    
    def test_load_documents_from_directory(self):
        processor = DocumentProcessor(chunk_size=200)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.md"
            test_file.write_text("""# Test Document

## Section A

This is section A content with enough text to form a chunk.

## Section B

This is section B content with different information.
""")
            
            chunks = processor.load_documents(tmpdir)
            
            assert len(chunks) >= 2
            assert all(isinstance(c, DocumentChunk) for c in chunks)
            assert any("Section A" in c.source_section for c in chunks)
    
    def test_long_paragraph_splitting(self):
        processor = DocumentProcessor(chunk_size=50)
        
        long_para = "This is a very long sentence that exceeds the chunk size. " * 5
        chunks = list(processor._split_long_paragraph(long_para, 0))
        
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk["text"]) <= 100
