"""HuggingFace LLM implementation."""

import logging
from collections.abc import AsyncIterator
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from openrag.config import HuggingFaceConfig
from openrag.core.base import LLM

logger = logging.getLogger(__name__)


class HuggingFaceLLM(LLM):
    """HuggingFace LLM implementation."""

    def __init__(self, config: HuggingFaceConfig) -> None:
        """Initialize HuggingFace model.

        Args:
            config: HuggingFace configuration
        """
        self.config = config
        logger.info(f"Loading HuggingFace model: {config.model_name}")

        # Configure quantization
        quantization_config = None
        if config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif config.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if config.device == "cuda" else None,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        )

        if config.device == "cpu":
            self.model = self.model.to("cpu")

        logger.info(f"Loaded model on device: {config.device}")

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> str:
        """Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        # Construct full prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # Tokenize
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the input prompt from output
        if generated_text.startswith(full_prompt):
            generated_text = generated_text[len(full_prompt) :].strip()

        logger.debug(f"Generated {len(generated_text)} characters")
        return generated_text

    async def generate_stream(
        self, prompt: str, system_prompt: str | None = None, **kwargs: Any
    ) -> AsyncIterator[str]:
        """Generate text with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Yields:
            Text chunks
        """
        # For simplicity, yield the full response at once
        # True streaming would require TextIteratorStreamer
        response = await self.generate(prompt, system_prompt, **kwargs)
        yield response
