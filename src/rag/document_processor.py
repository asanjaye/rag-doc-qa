import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Generator
import hashlib


@dataclass
class DocumentChunk:
    content: str
    source_file: str
    source_section: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)
    
    @property
    def chunk_id(self) -> str:
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{Path(self.source_file).stem}_{self.chunk_index}_{content_hash}"
    
    @property
    def citation(self) -> str:
        return f"[{self.source_section}]({self.source_file})"


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_documents(self, directory: str) -> list[DocumentChunk]:
        chunks = []
        dir_path = Path(directory)
        
        for file_path in dir_path.glob("**/*.md"):
            file_chunks = list(self.process_markdown(file_path))
            chunks.extend(file_chunks)
        
        return chunks
    
    def process_markdown(self, file_path: Path) -> Generator[DocumentChunk, None, None]:
        content = file_path.read_text(encoding="utf-8")
        sections = self._split_by_headers(content)
        
        chunk_index = 0
        for section_title, section_content in sections:
            for chunk in self._chunk_text(section_content):
                yield DocumentChunk(
                    content=chunk["text"],
                    source_file=str(file_path),
                    source_section=section_title,
                    chunk_index=chunk_index,
                    start_char=chunk["start"],
                    end_char=chunk["end"],
                    metadata={
                        "file_name": file_path.name,
                        "section": section_title,
                    }
                )
                chunk_index += 1
    
    def _split_by_headers(self, content: str) -> list[tuple[str, str]]:
        header_pattern = r'^(#{1,3})\s+(.+)$'
        lines = content.split('\n')
        sections = []
        current_section = "Introduction"
        current_content = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                if current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        if current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        return sections
    
    def _chunk_text(self, text: str) -> Generator[dict, None, None]:
        text = text.strip()
        if not text:
            return
        
        paragraphs = re.split(r'\n\n+', text)
        current_chunk = ""
        current_start = 0
        char_pos = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = char_pos
            else:
                if current_chunk:
                    yield {
                        "text": current_chunk,
                        "start": current_start,
                        "end": char_pos - 2
                    }
                
                if len(para) > self.chunk_size:
                    for sub_chunk in self._split_long_paragraph(para, char_pos):
                        yield sub_chunk
                    current_chunk = ""
                else:
                    current_chunk = para
                    current_start = char_pos
            
            char_pos += len(para) + 2
        
        if current_chunk:
            yield {
                "text": current_chunk,
                "start": current_start,
                "end": char_pos
            }
    
    def _split_long_paragraph(self, para: str, start_pos: int) -> Generator[dict, None, None]:
        sentences = re.split(r'(?<=[.!?])\s+', para)
        current_chunk = ""
        chunk_start = start_pos
        pos = start_pos
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk = (current_chunk + " " + sentence).strip()
            else:
                if current_chunk:
                    yield {
                        "text": current_chunk,
                        "start": chunk_start,
                        "end": pos
                    }
                current_chunk = sentence
                chunk_start = pos
            pos += len(sentence) + 1
        
        if current_chunk:
            yield {
                "text": current_chunk,
                "start": chunk_start,
                "end": pos
            }
