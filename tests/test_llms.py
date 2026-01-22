"""Tests for LLM implementations."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openrag.config import HuggingFaceConfig, OllamaConfig, OpenAIConfig
from openrag.llms import HuggingFaceLLM, OllamaLLM, OpenAILLM


class TestOpenAILLM:
    """Tests for OpenAI LLM."""

    def test_openai_llm_initialization(self):
        """Test OpenAI LLM initialization."""
        config = OpenAIConfig(api_key="test-key", model="gpt-4")
        llm = OpenAILLM(config)
        assert llm.config.model == "gpt-4"
        assert llm.config.api_key == "test-key"

    @pytest.mark.asyncio
    async def test_openai_llm_generate(self):
        """Test OpenAI LLM generate method."""
        config = OpenAIConfig(api_key="test-key")
        llm = OpenAILLM(config)

        # Mock the client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        llm.client.chat.completions.create = AsyncMock(return_value=mock_response)

        response = await llm.generate("Hello")
        assert response == "Test response"


class TestOllamaLLM:
    """Tests for Ollama LLM."""

    def test_ollama_llm_initialization(self):
        """Test Ollama LLM initialization."""
        config = OllamaConfig(host="http://localhost", port=11434, model="llama3")
        llm = OllamaLLM(config)
        assert llm.config.model == "llama3"
        assert llm.base_url == "http://localhost:11434"

    def test_ollama_llm_default_values(self):
        """Test Ollama LLM default configuration."""
        config = OllamaConfig()
        llm = OllamaLLM(config)
        assert llm.config.host == "http://localhost"
        assert llm.config.port == 11434
        assert llm.config.model == "llama3"
        assert llm.config.temperature == 0.7
        assert llm.config.max_tokens == 1000
        assert llm.config.timeout == 60

    @pytest.mark.asyncio
    async def test_ollama_llm_generate(self):
        """Test Ollama LLM generate method."""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        # Mock httpx client
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Test response from Ollama"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await llm.generate("Hello, world!")
            assert response == "Test response from Ollama"

    @pytest.mark.asyncio
    async def test_ollama_llm_generate_with_system_prompt(self):
        """Test Ollama LLM generate with system prompt."""
        config = OllamaConfig(model="llama3")
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"message": {"content": "Response with context"}}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            response = await llm.generate(
                "What is Python?", system_prompt="You are a programming expert."
            )
            assert response == "Response with context"

            # Verify the messages include both system and user
            call_args = mock_client.post.call_args
            json_data = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert len(json_data["messages"]) == 2
            assert json_data["messages"][0]["role"] == "system"
            assert json_data["messages"][0]["content"] == "You are a programming expert."
            assert json_data["messages"][1]["role"] == "user"
            assert json_data["messages"][1]["content"] == "What is Python?"

    @pytest.mark.asyncio
    async def test_ollama_llm_list_models(self):
        """Test Ollama LLM list_models method."""
        config = OllamaConfig()
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.return_value = mock_response
            mock_client_class.return_value = mock_client

            models = await llm.list_models()
            assert models == ["llama3", "mistral", "codellama"]

    @pytest.mark.asyncio
    async def test_ollama_llm_list_models_error(self):
        """Test Ollama LLM list_models method with error."""
        config = OllamaConfig()
        llm = OllamaLLM(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.get.side_effect = Exception("Connection error")
            mock_client_class.return_value = mock_client

            models = await llm.list_models()
            assert models == []

    @pytest.mark.asyncio
    async def test_ollama_llm_pull_model(self):
        """Test Ollama LLM pull_model method."""
        config = OllamaConfig()
        llm = OllamaLLM(config)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.return_value = mock_response
            mock_client_class.return_value = mock_client

            result = await llm.pull_model("llama3")
            assert result is True

            call_args = mock_client.post.call_args
            json_data = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert json_data["name"] == "llama3"

    @pytest.mark.asyncio
    async def test_ollama_llm_pull_model_error(self):
        """Test Ollama LLM pull_model method with error."""
        config = OllamaConfig()
        llm = OllamaLLM(config)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client.post.side_effect = Exception("Pull failed")
            mock_client_class.return_value = mock_client

            result = await llm.pull_model("llama3")
            assert result is False


class TestHuggingFaceLLM:
    """Tests for HuggingFace LLM."""

    def test_huggingface_llm_initialization(self):
        """Test HuggingFace LLM initialization."""
        config = HuggingFaceConfig(model_name="gpt2")
        llm = HuggingFaceLLM(config)
        assert llm.config.model_name == "gpt2"
