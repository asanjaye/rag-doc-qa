from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class GenerationResult:
    answer: str
    model: str
    tokens_used: int


class BaseGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str
    ) -> GenerationResult:
        pass


class OpenAIGenerator(BaseGenerator):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str
    ) -> GenerationResult:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
            max_tokens=1000
        )
        
        return GenerationResult(
            answer=response.choices[0].message.content,
            model=self.model,
            tokens_used=response.usage.total_tokens
        )


class AnthropicGenerator(BaseGenerator):
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str
    ) -> GenerationResult:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        return GenerationResult(
            answer=response.content[0].text,
            model=self.model,
            tokens_used=response.usage.input_tokens + response.usage.output_tokens
        )


class MockGenerator(BaseGenerator):
    def generate(
        self,
        query: str,
        context: str,
        system_prompt: str
    ) -> GenerationResult:
        return GenerationResult(
            answer="[Mock response - LLM not configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY]",
            model="mock",
            tokens_used=0
        )


class Generator:
    SYSTEM_PROMPT = """You are a helpful assistant for NovaTech Solutions. Answer questions based ONLY on the provided context. 

IMPORTANT RULES:
1. Use ONLY information from the provided context to answer
2. When citing information, use inline citations in the format [Source: section_name] 
3. If the context doesn't contain enough information to fully answer the question, say so clearly
4. Never make up information that isn't in the context
5. Keep answers concise but complete
6. If multiple sources contain relevant information, cite each one

Format your response with clear paragraphs and use the citations naturally within your answer."""
    
    def __init__(
        self,
        provider: str = "openai",
        openai_api_key: str = "",
        openai_model: str = "gpt-4o-mini",
        anthropic_api_key: str = "",
        anthropic_model: str = "claude-3-haiku-20240307"
    ):
        self.provider = provider
        
        if provider == "openai" and openai_api_key:
            self._generator = OpenAIGenerator(openai_api_key, openai_model)
        elif provider == "anthropic" and anthropic_api_key:
            self._generator = AnthropicGenerator(anthropic_api_key, anthropic_model)
        else:
            self._generator = MockGenerator()
            self.provider = "mock"
    
    def generate(self, query: str, context: str) -> GenerationResult:
        return self._generator.generate(query, context, self.SYSTEM_PROMPT)
    
    @property
    def is_configured(self) -> bool:
        return self.provider != "mock"
