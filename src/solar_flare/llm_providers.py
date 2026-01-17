"""
LLM Provider Factory for Solar-Flare.

This module provides a unified interface for creating LLM instances
across multiple providers including cloud-based (OpenAI, Anthropic)
and local/self-hosted options (Ollama, LM Studio).
"""

import os
from enum import Enum
from typing import Optional

from langchain_core.language_models import BaseChatModel


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    LMSTUDIO = "lmstudio"
    DEEPSEEK = "deepseek"


# Default models for each provider
DEFAULT_MODELS = {
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.LMSTUDIO: "local-model",
    LLMProvider.DEEPSEEK: "deepseek-chat",
}

# Default base URLs for local providers
DEFAULT_BASE_URLS = {
    LLMProvider.OLLAMA: "http://localhost:11434",
    LLMProvider.LMSTUDIO: "http://localhost:1234/v1",
    LLMProvider.DEEPSEEK: "https://api.deepseek.com",
}


def create_llm(
    provider: str | LLMProvider = LLMProvider.OPENAI,
    model: Optional[str] = None,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs,
) -> BaseChatModel:
    """
    Factory function to create LLM instances for any supported provider.

    This provides a unified interface for creating LLM instances, abstracting
    away provider-specific configuration details while allowing customization.

    Args:
        provider: LLM provider - one of 'openai', 'anthropic', 'ollama', 'lmstudio'.
                  Can be a string or LLMProvider enum value.
        model: Model name. If not specified, uses provider-specific defaults:
               - openai: 'gpt-4o'
               - anthropic: 'claude-3-5-sonnet-20241022'
               - ollama: 'llama3.1'
               - lmstudio: 'local-model'
        temperature: Model temperature (default: 0.3).
        base_url: Custom API base URL. Required for lmstudio, optional for ollama.
                  Defaults are read from environment variables or built-in defaults.
        api_key: API key. Required for openai/anthropic, optional for local providers.
                 If not specified, reads from environment variables.
        **kwargs: Additional provider-specific parameters passed to the model constructor.

    Returns:
        Configured BaseChatModel instance ready for use with Solar-Flare agents.

    Raises:
        ValueError: If an unsupported provider is specified.
        ImportError: If required provider package is not installed.

    Examples:
        >>> # OpenAI (default)
        >>> llm = create_llm()

        >>> # Anthropic Claude
        >>> llm = create_llm(provider="anthropic", model="claude-3-5-sonnet-20241022")

        >>> # Ollama (local)
        >>> llm = create_llm(provider="ollama", model="llama3.1")

        >>> # LM Studio (OpenAI-compatible local server)
        >>> llm = create_llm(
        ...     provider="lmstudio",
        ...     base_url="http://localhost:1234/v1"
        ... )

        >>> # DeepSeek
        >>> llm = create_llm(provider="deepseek", model="deepseek-chat")
    """
    # Normalize provider to enum
    if isinstance(provider, str):
        try:
            provider = LLMProvider(provider.lower())
        except ValueError:
            valid_providers = [p.value for p in LLMProvider]
            raise ValueError(
                f"Unsupported provider: '{provider}'. "
                f"Valid providers are: {valid_providers}"
            )

    # Get temperature from env if using default
    env_temp = os.getenv("LLM_TEMPERATURE", "0.3")
    try:
        temperature = float(env_temp)
    except ValueError:
        pass  # Use default if invalid

    # Get default model if not specified
    if model is None:
        model = os.getenv(
            f"{provider.value.upper()}_MODEL", DEFAULT_MODELS.get(provider)
        )

    # Create provider-specific LLM instance
    if provider == LLMProvider.OPENAI:
        return _create_openai_llm(model, temperature, api_key, **kwargs)
    elif provider == LLMProvider.ANTHROPIC:
        return _create_anthropic_llm(model, temperature, api_key, **kwargs)
    elif provider == LLMProvider.OLLAMA:
        return _create_ollama_llm(model, temperature, base_url, **kwargs)
    elif provider == LLMProvider.LMSTUDIO:
        return _create_lmstudio_llm(model, temperature, base_url, api_key, **kwargs)
    elif provider == LLMProvider.DEEPSEEK:
        return _create_deepseek_llm(model, temperature, base_url, api_key, **kwargs)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _create_openai_llm(
    model: str,
    temperature: float,
    api_key: Optional[str],
    **kwargs,
) -> BaseChatModel:
    """Create an OpenAI ChatModel instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for OpenAI provider. "
            "Install it with: pip install langchain-openai"
        )

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _create_anthropic_llm(
    model: str,
    temperature: float,
    api_key: Optional[str],
    **kwargs,
) -> BaseChatModel:
    """Create an Anthropic ChatModel instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic package is required for Anthropic provider. "
            "Install it with: pip install langchain-anthropic"
        )

    if api_key is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        api_key=api_key,
        **kwargs,
    )


def _create_ollama_llm(
    model: str,
    temperature: float,
    base_url: Optional[str],
    **kwargs,
) -> BaseChatModel:
    """Create an Ollama ChatModel instance."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama package is required for Ollama provider. "
            "Install it with: pip install langchain-ollama"
        )

    if base_url is None:
        base_url = os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URLS[LLMProvider.OLLAMA])

    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        **kwargs,
    )


def _create_lmstudio_llm(
    model: str,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    **kwargs,
) -> BaseChatModel:
    """
    Create an LM Studio ChatModel instance.

    LM Studio provides an OpenAI-compatible API, so we use ChatOpenAI
    with a custom base_url pointing to the local LM Studio server.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for LM Studio provider. "
            "Install it with: pip install langchain-openai"
        )

    if base_url is None:
        base_url = os.getenv(
            "LMSTUDIO_BASE_URL", DEFAULT_BASE_URLS[LLMProvider.LMSTUDIO]
        )

    # LM Studio doesn't require an API key, but ChatOpenAI expects one
    # Use a dummy key if not provided
    if api_key is None:
        api_key = os.getenv("LMSTUDIO_API_KEY", "lm-studio")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


def _create_deepseek_llm(
    model: str,
    temperature: float,
    base_url: Optional[str],
    api_key: Optional[str],
    **kwargs,
) -> BaseChatModel:
    """
    Create a DeepSeek ChatModel instance.

    DeepSeek provides an OpenAI-compatible API, so we use ChatOpenAI
    with a custom base_url pointing to api.deepseek.com.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai package is required for DeepSeek provider. "
            "Install it with: pip install langchain-openai"
        )

    if base_url is None:
        base_url = os.getenv(
            "DEEPSEEK_BASE_URL", DEFAULT_BASE_URLS[LLMProvider.DEEPSEEK]
        )

    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise ValueError(
            "DeepSeek API key is required. "
            "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


def list_providers() -> list[str]:
    """
    Return a list of all supported LLM provider names.

    Returns:
        List of provider names as strings.
    """
    return [p.value for p in LLMProvider]


def get_default_model(provider: str | LLMProvider) -> str:
    """
    Get the default model name for a provider.

    Args:
        provider: LLM provider name or enum value.

    Returns:
        Default model name for the provider.
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider.lower())
    return DEFAULT_MODELS.get(provider, "")
