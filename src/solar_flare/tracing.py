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
    public_key: Optional[str] = None,
    secret_key: Optional[str] = None,
    host: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    trace_name: Optional[str] = None,
    release: Optional[str] = None,
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
    
    Args:
        public_key: Langfuse public key (or LANGFUSE_PUBLIC_KEY env var)
        secret_key: Langfuse secret key (or LANGFUSE_SECRET_KEY env var)
        host: Langfuse host URL (or LANGFUSE_HOST env var, default: cloud)
        session_id: Optional session ID for grouping traces
        user_id: Optional user ID for attribution
        trace_name: Optional name for the trace
        release: Optional release/version identifier
        metadata: Optional additional metadata dict
        enabled: Whether tracing is enabled (default: True)
        
    Returns:
        CallbackHandler instance if Langfuse is available and configured,
        None otherwise.
        
    Examples:
        >>> # Basic usage (reads from environment)
        >>> handler = create_langfuse_handler()
        >>> llm = create_llm(callbacks=[handler] if handler else [])
        
        >>> # With explicit configuration
        >>> handler = create_langfuse_handler(
        ...     session_id="user-session-123",
        ...     trace_name="requirements_analysis",
        ...     metadata={"project": "logging-service"}
        ... )
    """
    if not enabled:
        return None
    
    # Check for required keys
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    
    if not public_key or not secret_key:
        return None
    
    try:
        from langfuse.callback import CallbackHandler
    except ImportError:
        return None
    
    # Get optional configuration
    host = host or os.getenv("LANGFUSE_HOST")
    
    # Build handler kwargs
    handler_kwargs = {
        "public_key": public_key,
        "secret_key": secret_key,
    }
    
    if host:
        handler_kwargs["host"] = host
    if session_id:
        handler_kwargs["session_id"] = session_id
    if user_id:
        handler_kwargs["user_id"] = user_id
    if trace_name:
        handler_kwargs["trace_name"] = trace_name
    if release:
        handler_kwargs["release"] = release
    if metadata:
        handler_kwargs["metadata"] = metadata
    
    return CallbackHandler(**handler_kwargs)


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
