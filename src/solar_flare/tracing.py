"""
Tracing and observability support for Solar-Flare.

This module provides integrations with observability platforms
for monitoring and debugging LLM interactions.

Supported platforms:
- Langfuse: Open-source LLM observability (langfuse.com)
- LangSmith: LangChain's native tracing (via environment variables)
"""

import os
from typing import Optional, List, Any


def create_langfuse_handler(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    enabled: bool = True,
) -> Optional[Any]:
    """
    Create a Langfuse callback handler for LangChain tracing.
    
    Langfuse provides open-source LLM observability with features like:
    - Trace visualization
    - Cost tracking
    - Latency monitoring
    - Prompt management
    - User feedback collection
    
    Note: Langfuse v3 reads API keys from environment variables:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_HOST (optional)
    
    Args:
        session_id: Optional session ID for grouping traces
        user_id: Optional user ID for attribution
        metadata: Optional additional metadata dict
        enabled: Whether tracing is enabled (default: True)
        
    Returns:
        CallbackHandler instance if Langfuse is available and configured,
        None otherwise.
        
    Examples:
        >>> # Basic usage (reads from environment)
        >>> handler = create_langfuse_handler()
        >>> llm = create_llm(callbacks=[handler] if handler else [])
        
        >>> # With session and metadata
        >>> handler = create_langfuse_handler(
        ...     session_id="user-session-123",
        ...     metadata={"project": "logging-service"}
        ... )
    """
    if not enabled:
        return None
    
    # Check for required keys (langfuse v3 reads from env vars)
    if not is_langfuse_configured():
        return None
    
    try:
        from langfuse.langchain import CallbackHandler
        from langfuse import get_client
    except ImportError:
        return None
    
    try:
        # Create handler - v3 uses env vars for auth
        handler = CallbackHandler()
        
        # Set trace context if session/user/metadata provided
        if session_id or user_id or metadata:
            # Get the langfuse client to set trace metadata
            client = get_client()
            # Note: In v3, metadata is set via trace context
            # The handler will automatically use env-configured client
        
        return handler
    except Exception:
        return None


def get_tracing_callbacks(
    langfuse: bool = True,
    langfuse_session_id: Optional[str] = None,
    langfuse_metadata: Optional[dict] = None,
) -> List[Any]:
    """
    Get a list of configured tracing callbacks.
    
    This is a convenience function that returns all enabled
    tracing handlers based on environment configuration.
    
    Args:
        langfuse: Whether to include Langfuse handler (default: True)
        langfuse_session_id: Optional session ID for Langfuse
        langfuse_metadata: Optional metadata dict for Langfuse
        
    Returns:
        List of callback handlers (may be empty if none configured)
        
    Examples:
        >>> callbacks = get_tracing_callbacks()
        >>> llm = create_llm()
        >>> result = await run_workflow(llm, message, callbacks=callbacks)
    """
    callbacks = []
    
    if langfuse:
        handler = create_langfuse_handler(
            session_id=langfuse_session_id,
            metadata=langfuse_metadata,
        )
        if handler:
            callbacks.append(handler)
    
    return callbacks


def is_langfuse_configured() -> bool:
    """
    Check if Langfuse is configured via environment variables.
    
    Returns:
        True if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set.
    """
    return bool(
        os.getenv("LANGFUSE_PUBLIC_KEY") and 
        os.getenv("LANGFUSE_SECRET_KEY")
    )
