import chromadb
from pathlib import Path
from dataclasses import dataclass

from .document_processor import DocumentChunk
from .embeddings import EmbeddingService


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float
    
    @property
    def is_confident(self) -> bool:
        return self.score >= 0.65


class Retriever:
    COLLECTION_NAME = "knowledge_base"
    
    def __init__(
        self,
        embedding_service: EmbeddingService,
        persist_dir: str = "./data/chroma"
    ):
        self.embedding_service = embedding_service
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_documents(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source_file": chunk.source_file,
                "source_section": chunk.source_section,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
            for chunk in chunks
        ]
        
        embeddings = self.embedding_service.embed_documents(documents)
        
        existing_ids = set(self.collection.get()["ids"])
        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]
        
        if new_indices:
            self.collection.add(
                ids=[ids[i] for i in new_indices],
                documents=[documents[i] for i in new_indices],
                embeddings=[embeddings[i] for i in new_indices],
                metadatas=[metadatas[i] for i in new_indices]
            )
        
        return len(new_indices)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0
    ) -> list[RetrievalResult]:
        if self.collection.count() == 0:
            return []
        
        query_embedding = self.embedding_service.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )
        
        retrieval_results = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1 - distance
            
            if score < threshold:
                continue
            
            chunk = DocumentChunk(
                content=doc,
                source_file=metadata["source_file"],
                source_section=metadata["source_section"],
                chunk_index=metadata["chunk_index"],
                start_char=metadata["start_char"],
                end_char=metadata["end_char"],
                metadata={
                    k: v for k, v in metadata.items()
                    if k not in ["source_file", "source_section", "chunk_index", "start_char", "end_char"]
                }
            )
            
            retrieval_results.append(RetrievalResult(chunk=chunk, score=score))
        
        return retrieval_results
    
    def clear_index(self):
        self.client.delete_collection(self.COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    @property
    def document_count(self) -> int:
        return self.collection.count()
