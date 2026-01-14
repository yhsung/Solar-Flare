"""
Conversation memory management for multi-turn interactions.

Provides session-based conversation tracking and persistence
for the Solar-Flare multi-agent system.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field


class ConversationSession(BaseModel):
    """Represents a conversation session with its history and metadata."""

    session_id: str = Field(description="Unique session identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class ConversationMemory:
    """
    Manages conversation history and session state.

    Features:
    - Session-based conversation tracking
    - Message persistence with timestamps
    - Context window management
    - Integration with LangGraph checkpointing

    Attributes:
        max_messages: Maximum messages to retain per session
        sessions: Active session storage
        checkpointer: LangGraph checkpointer for state persistence
    """

    def __init__(
        self,
        max_messages: int = 50,
    ):
        """
        Initialize conversation memory.

        Args:
            max_messages: Maximum messages to retain per session
        """
        self.max_messages = max_messages
        self.checkpointer = MemorySaver()
        self.sessions: Dict[str, ConversationSession] = {}

    def create_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSession:
        """
        Create a new conversation session.

        Args:
            session_id: Unique identifier for the session
            metadata: Optional metadata (e.g., user info, project context)

        Returns:
            Newly created ConversationSession
        """
        session = ConversationSession(
            session_id=session_id,
            metadata=metadata or {},
        )
        self.sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ConversationSession]:
        """
        Retrieve an existing session.

        Args:
            session_id: Session identifier

        Returns:
            ConversationSession if found, None otherwise
        """
        return self.sessions.get(session_id)

    def get_or_create_session(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationSession:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier
            metadata: Optional metadata for new session

        Returns:
            Existing or newly created ConversationSession
        """
        session = self.sessions.get(session_id)
        if session is None:
            session = self.create_session(session_id, metadata)
        return session

    def add_message(
        self,
        session_id: str,
        message: BaseMessage,
    ) -> None:
        """
        Add a message to a session.

        Args:
            session_id: Session to add message to
            message: Message to add
        """
        session = self.get_or_create_session(session_id)

        # Serialize message
        msg_dict = {
            "type": type(message).__name__,
            "content": message.content,
            "timestamp": datetime.now().isoformat(),
        }

        # Add additional fields if present
        if hasattr(message, "additional_kwargs"):
            msg_dict["additional_kwargs"] = message.additional_kwargs

        session.messages.append(msg_dict)
        session.last_updated = datetime.now()

        # Trim if exceeds max
        if len(session.messages) > self.max_messages:
            session.messages = session.messages[-self.max_messages :]

    def add_human_message(self, session_id: str, content: str) -> None:
        """Add a human message to the session."""
        self.add_message(session_id, HumanMessage(content=content))

    def add_ai_message(self, session_id: str, content: str) -> None:
        """Add an AI message to the session."""
        self.add_message(session_id, AIMessage(content=content))

    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[BaseMessage]:
        """
        Get messages from a session as BaseMessage objects.

        Args:
            session_id: Session to get messages from
            limit: Optional limit on number of messages (most recent)

        Returns:
            List of BaseMessage objects
        """
        session = self.sessions.get(session_id)
        if session is None:
            return []

        messages = session.messages
        if limit:
            messages = messages[-limit:]

        # Convert back to BaseMessage objects
        result = []
        for msg in messages:
            msg_type = msg.get("type", "HumanMessage")
            content = msg.get("content", "")

            if msg_type == "HumanMessage":
                result.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                result.append(AIMessage(content=content))
            elif msg_type == "SystemMessage":
                result.append(SystemMessage(content=content))
            else:
                # Default to HumanMessage for unknown types
                result.append(HumanMessage(content=content))

        return result

    def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        session = self.sessions.get(session_id)
        return len(session.messages) if session else 0

    def clear_session(self, session_id: str) -> bool:
        """
        Clear all messages from a session.

        Args:
            session_id: Session to clear

        Returns:
            True if session existed and was cleared
        """
        session = self.sessions.get(session_id)
        if session:
            session.messages = []
            session.last_updated = datetime.now()
            return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session entirely.

        Args:
            session_id: Session to delete

        Returns:
            True if session existed and was deleted
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return list(self.sessions.keys())

    def get_checkpointer(self) -> MemorySaver:
        """
        Get the checkpointer for LangGraph integration.

        Returns:
            MemorySaver instance for workflow checkpointing
        """
        return self.checkpointer

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a summary of a session.

        Args:
            session_id: Session to summarize

        Returns:
            Dictionary with session statistics
        """
        session = self.sessions.get(session_id)
        if session is None:
            return {"exists": False}

        return {
            "exists": True,
            "session_id": session.session_id,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat(),
            "message_count": len(session.messages),
            "metadata": session.metadata,
        }
