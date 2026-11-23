from abc import ABC, abstractmethod
import numpy as np
from typing import Union


class BaseEmbedding(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        pass


class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self.dimension = self._dimension_map.get(model, 1536)
    
    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class EmbeddingService:
    def __init__(
        self,
        provider: str = "local",
        openai_api_key: str = "",
        openai_model: str = "text-embedding-3-small",
        local_model: str = "all-MiniLM-L6-v2"
    ):
        if provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI embeddings")
            self._embedding = OpenAIEmbedding(openai_api_key, openai_model)
        else:
            self._embedding = LocalEmbedding(local_model)
        
        self.provider = provider
        self.dimension = self._embedding.dimension
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedding.embed(texts)
    
    def embed_query(self, text: str) -> list[float]:
        return self._embedding.embed_query(text)
