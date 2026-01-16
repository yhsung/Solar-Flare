"""
Unit tests for the LLM Provider Factory module.

Tests the create_llm factory function and LLMProvider enum.
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from solar_flare.llm_providers import (
    create_llm,
    LLMProvider,
    list_providers,
    get_default_model,
    DEFAULT_MODELS,
    DEFAULT_BASE_URLS,
)


class TestLLMProviderEnum:
    """Tests for the LLMProvider enum."""

    def test_provider_values(self):
        """Test that provider enum has expected values."""
        assert LLMProvider.OPENAI.value == "openai"
        assert LLMProvider.ANTHROPIC.value == "anthropic"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.LMSTUDIO.value == "lmstudio"

    def test_string_conversion(self):
        """Test that providers can be created from strings."""
        assert LLMProvider("openai") == LLMProvider.OPENAI
        assert LLMProvider("anthropic") == LLMProvider.ANTHROPIC
        assert LLMProvider("ollama") == LLMProvider.OLLAMA
        assert LLMProvider("lmstudio") == LLMProvider.LMSTUDIO


class TestListProviders:
    """Tests for the list_providers function."""

    def test_list_providers_returns_all(self):
        """Test that list_providers returns all provider names."""
        providers = list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
        assert "lmstudio" in providers
        assert len(providers) == 4


class TestGetDefaultModel:
    """Tests for the get_default_model function."""

    def test_get_default_model_by_enum(self):
        """Test getting default model using enum."""
        assert get_default_model(LLMProvider.OPENAI) == "gpt-4o"
        assert get_default_model(LLMProvider.ANTHROPIC) == "claude-3-5-sonnet-20241022"
        assert get_default_model(LLMProvider.OLLAMA) == "llama3.1"
        assert get_default_model(LLMProvider.LMSTUDIO) == "local-model"

    def test_get_default_model_by_string(self):
        """Test getting default model using string."""
        assert get_default_model("openai") == "gpt-4o"
        assert get_default_model("ollama") == "llama3.1"


class TestCreateLLM:
    """Tests for the create_llm factory function."""

    def test_invalid_provider_raises_error(self):
        """Test that an invalid provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            create_llm(provider="invalid_provider")

    def test_provider_enum_is_accepted(self):
        """Test that LLMProvider enum values are accepted."""
        # We can't actually create the LLM without mocking, but we can
        # verify the enum is properly handled by checking error handling
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()
            result = create_llm(provider=LLMProvider.OPENAI)
            assert result is not None


class TestCreateOpenAILLM:
    """Tests for OpenAI LLM creation."""

    def test_creates_chat_openai_with_api_key(self):
        """Test that ChatOpenAI is created with correct parameters."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            result = create_llm(
                provider="openai",
                model="gpt-4o",
                temperature=0.5,
                api_key="test-api-key",
            )

            mock_chat.assert_called_once_with(
                model="gpt-4o",
                temperature=0.5,
                api_key="test-api-key",
            )
            assert result is not None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"}, clear=False)
    def test_reads_api_key_from_environment(self):
        """Test that API key is read from environment when not provided."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            create_llm(provider="openai", model="gpt-4o")

            # Verify the call was made with the env key
            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["api_key"] == "env-api-key"


class TestCreateAnthropicLLM:
    """Tests for Anthropic LLM creation."""

    def test_creates_chat_anthropic(self):
        """Test that ChatAnthropic is created with correct parameters."""
        with patch("langchain_anthropic.ChatAnthropic") as mock_chat:
            mock_chat.return_value = MagicMock()

            result = create_llm(
                provider="anthropic",
                model="claude-3",
                temperature=0.3,
                api_key="test-key",
            )

            mock_chat.assert_called_once_with(
                model="claude-3",
                temperature=0.3,
                api_key="test-key",
            )
            assert result is not None


class TestCreateOllamaLLM:
    """Tests for Ollama LLM creation."""

    def test_creates_chat_ollama_with_base_url(self):
        """Test that ChatOllama is created with correct parameters."""
        with patch("langchain_ollama.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()

            result = create_llm(
                provider="ollama",
                model="llama3.1",
                temperature=0.3,
                base_url="http://custom:11434",
            )

            mock_chat.assert_called_once_with(
                model="llama3.1",
                temperature=0.3,
                base_url="http://custom:11434",
            )
            assert result is not None

    @patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://env-ollama:11434"}, clear=False)
    def test_reads_base_url_from_environment(self):
        """Test that base_url is read from environment when not provided."""
        with patch("langchain_ollama.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()

            create_llm(provider="ollama", model="llama3.1")

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["base_url"] == "http://env-ollama:11434"

    def test_uses_default_base_url(self):
        """Test that default base URL is used when not specified."""
        with patch("langchain_ollama.ChatOllama") as mock_chat:
            mock_chat.return_value = MagicMock()
            
            # Clear OLLAMA_BASE_URL if it exists
            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("OLLAMA_BASE_URL", None)
                create_llm(provider="ollama", model="llama3.1")

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["base_url"] == "http://localhost:11434"


class TestCreateLMStudioLLM:
    """Tests for LM Studio LLM creation."""

    def test_creates_with_custom_base_url(self):
        """Test that ChatOpenAI is created with LM Studio base URL."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            result = create_llm(
                provider="lmstudio",
                model="local-model",
                temperature=0.3,
                base_url="http://localhost:1234/v1",
            )

            mock_chat.assert_called_once_with(
                model="local-model",
                temperature=0.3,
                base_url="http://localhost:1234/v1",
                api_key="lm-studio",  # Default dummy key
            )
            assert result is not None

    @patch.dict(os.environ, {"LMSTUDIO_API_KEY": "custom-key"}, clear=False)
    def test_uses_custom_api_key_from_env(self):
        """Test that custom API key from env is used."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            create_llm(
                provider="lmstudio",
                base_url="http://localhost:1234/v1",
            )

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["api_key"] == "custom-key"

    def test_uses_default_base_url_from_env(self):
        """Test that base URL is read from environment."""
        with patch.dict(os.environ, {"LMSTUDIO_BASE_URL": "http://custom:5000/v1"}, clear=False):
            with patch("langchain_openai.ChatOpenAI") as mock_chat:
                mock_chat.return_value = MagicMock()

                create_llm(provider="lmstudio")

                call_kwargs = mock_chat.call_args[1]
                assert call_kwargs["base_url"] == "http://custom:5000/v1"


class TestProviderDefaults:
    """Tests for default values."""

    def test_default_models_dict(self):
        """Test that DEFAULT_MODELS has all providers."""
        assert LLMProvider.OPENAI in DEFAULT_MODELS
        assert LLMProvider.ANTHROPIC in DEFAULT_MODELS
        assert LLMProvider.OLLAMA in DEFAULT_MODELS
        assert LLMProvider.LMSTUDIO in DEFAULT_MODELS

    def test_default_base_urls_dict(self):
        """Test that DEFAULT_BASE_URLS has local providers."""
        assert LLMProvider.OLLAMA in DEFAULT_BASE_URLS
        assert LLMProvider.LMSTUDIO in DEFAULT_BASE_URLS
        assert DEFAULT_BASE_URLS[LLMProvider.OLLAMA] == "http://localhost:11434"
        assert DEFAULT_BASE_URLS[LLMProvider.LMSTUDIO] == "http://localhost:1234/v1"

    @patch.dict(os.environ, {"OPENAI_MODEL": "gpt-4-turbo"}, clear=False)
    def test_model_from_environment_variable(self):
        """Test that model can be read from environment variable."""
        with patch("langchain_openai.ChatOpenAI") as mock_chat:
            mock_chat.return_value = MagicMock()

            create_llm(provider="openai")

            call_kwargs = mock_chat.call_args[1]
            assert call_kwargs["model"] == "gpt-4-turbo"
