"""Ollama LLM implementation."""

import logging
from collections.abc import AsyncIterator
from typing import Any, Optional

import httpx

from openrag.config import OllamaConfig
from openrag.core.base import LLM

logger = logging.getLogger(__name__)


class OllamaLLM(LLM):
    """Ollama LLM implementation."""

    def __init__(self, config: OllamaConfig) -> None:
        """Initialize Ollama client.

        Args:
            config: Ollama configuration
        """
        self.config = config
        self.base_url = f"{config.host}:{config.port}"
        logger.info(f"Initialized Ollama client with model: {config.model} at {self.base_url}")

    async def generate(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get("model", self.config.model),
                    "messages": messages,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()

            content = data.get("message", {}).get("content", "")
            logger.debug(f"Generated {len(content)} characters")
            return content

    async def generate_stream(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate text with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": kwargs.get("model", self.config.model),
                    "messages": messages,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    },
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        import json

                        try:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                content = data["message"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

    async def list_models(self) -> list[str]:
        """List available models on Ollama server.

        Returns:
            List of model names
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
            except httpx.RequestError as e:
                logger.error(f"Failed to list models: {e}")
                return []

    async def pull_model(self, model: Optional[str] = None) -> bool:
        """Pull a model from Ollama library.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        model_name = model or self.config.model
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                )
                response.raise_for_status()
                return True
            except httpx.RequestError as e:
                logger.error(f"Failed to pull model {model_name}: {e}")
                return False
