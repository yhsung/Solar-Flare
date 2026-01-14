"""
State checkpointing module for Solar-Flare workflow persistence.

This module provides checkpointing capabilities for persisting workflow
state across executions, enabling resumption and state inspection.
"""

import os
import pickle
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from pydantic import BaseModel, Field


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint behavior."""

    backend: str = Field(
        default="memory",
        description="Checkpoint backend: 'memory', 'sqlite', or 'file'",
    )
    persist_directory: str = Field(
        default="./data/checkpoints",
        description="Directory for file-based persistence",
    )
    database_path: str = Field(
        default="./data/conversations.db",
        description="Path for SQLite database",
    )
    max_checkpoints: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum checkpoints to retain per thread",
    )
    auto_save: bool = Field(
        default=True,
        description="Automatically save checkpoints after each step",
    )


class CheckpointMetadata(BaseModel):
    """Metadata about a checkpoint."""

    checkpoint_id: str = Field(description="Unique checkpoint identifier")
    thread_id: str = Field(description="Thread/session identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    step: int = Field(description="Workflow step number")
    phase: str = Field(description="Current workflow phase")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class SolarFlareCheckpointer:
    """
    Checkpointer for Solar-Flare workflow state persistence.

    Supports multiple backends:
    - memory: In-memory checkpointing (ephemeral)
    - sqlite: SQLite database persistence
    - file: File-based persistence with pickle

    Attributes:
        config: Checkpoint configuration
        saver: LangGraph checkpoint saver instance
    """

    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize the checkpointer.

        Args:
            config: Checkpoint configuration
        """
        self.config = config or CheckpointConfig()
        self.saver = self._create_saver()
        self._checkpoint_metadata: Dict[str, List[CheckpointMetadata]] = {}

        # Create directories if needed
        if self.config.backend == "file":
            Path(self.config.persist_directory).mkdir(parents=True, exist_ok=True)
        elif self.config.backend == "sqlite":
            Path(self.config.database_path).parent.mkdir(parents=True, exist_ok=True)

    def _create_saver(self) -> BaseCheckpointSaver:
        """
        Create the appropriate checkpoint saver.

        Returns:
            BaseCheckpointSaver instance
        """
        if self.config.backend == "memory":
            return MemorySaver()
        elif self.config.backend == "sqlite":
            return SqliteSaver.from_conn_string(self.config.database_path)
        elif self.config.backend == "file":
            # File-based uses custom implementation
            return FileCheckpointSaver(self.config.persist_directory)
        else:
            raise ValueError(f"Unknown checkpoint backend: {self.config.backend}")

    def get_saver(self) -> BaseCheckpointSaver:
        """
        Get the LangGraph checkpoint saver.

        Returns:
            BaseCheckpointSaver instance
        """
        return self.saver

    def add_metadata(
        self,
        checkpoint_id: str,
        thread_id: str,
        step: int,
        phase: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add metadata for a checkpoint.

        Args:
            checkpoint_id: Checkpoint identifier
            thread_id: Thread/session identifier
            step: Workflow step number
            phase: Current workflow phase
            metadata: Additional metadata
        """
        checkpoint_meta = CheckpointMetadata(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            step=step,
            phase=phase,
            metadata=metadata or {},
        )

        if thread_id not in self._checkpoint_metadata:
            self._checkpoint_metadata[thread_id] = []

        self._checkpoint_metadata[thread_id].append(checkpoint_meta)

        # Trim if exceeding max
        if len(self._checkpoint_metadata[thread_id]) > self.config.max_checkpoints:
            self._checkpoint_metadata[thread_id] = self._checkpoint_metadata[thread_id][
                -self.config.max_checkpoints :
            ]

    def get_metadata(
        self,
        thread_id: str,
        limit: Optional[int] = None,
    ) -> List[CheckpointMetadata]:
        """
        Get metadata for checkpoints in a thread.

        Args:
            thread_id: Thread identifier
            limit: Optional limit on number of checkpoints

        Returns:
            List of checkpoint metadata
        """
        metadata = self._checkpoint_metadata.get(thread_id, [])
        if limit:
            return metadata[-limit:]
        return metadata

    def get_latest_metadata(self, thread_id: str) -> Optional[CheckpointMetadata]:
        """
        Get the most recent checkpoint metadata for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Latest CheckpointMetadata or None
        """
        metadata = self._checkpoint_metadata.get(thread_id)
        if metadata:
            return metadata[-1]
        return None

    def list_threads(self) -> List[str]:
        """
        List all thread IDs with checkpoints.

        Returns:
            List of thread IDs
        """
        return list(self._checkpoint_metadata.keys())

    def clear_thread(self, thread_id: str) -> bool:
        """
        Clear all checkpoints for a thread.

        Args:
            thread_id: Thread to clear

        Returns:
            True if thread existed and was cleared
        """
        if thread_id in self._checkpoint_metadata:
            del self._checkpoint_metadata[thread_id]
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpointer statistics.

        Returns:
            Dictionary with checkpointer stats
        """
        return {
            "backend": self.config.backend,
            "total_threads": len(self._checkpoint_metadata),
            "total_checkpoints": sum(
                len(metas) for metas in self._checkpoint_metadata.values()
            ),
            "persist_directory": self.config.persist_directory,
            "database_path": self.config.database_path,
            "max_checkpoints": self.config.max_checkpoints,
        }


class FileCheckpointSaver(BaseCheckpointSaver):
    """
    File-based checkpoint saver for environments without database support.

    Saves checkpoints as individual files with metadata tracking.
    """

    def __init__(
        self,
        persist_directory: str = "./data/checkpoints",
    ):
        """
        Initialize the file checkpoint saver.

        Args:
            persist_directory: Directory to store checkpoint files
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, Dict[str, str]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the checkpoint index from disk."""
        index_path = self.persist_directory / "index.json"
        if index_path.exists():
            with open(index_path, "r") as f:
                self._index = json.load(f)
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save the checkpoint index to disk."""
        index_path = self.persist_directory / "index.json"
        with open(index_path, "w") as f:
            json.dump(self._index, f)

    def _get_checkpoint_path(self, thread_id: str, checkpoint_id: str) -> Path:
        """Get the file path for a checkpoint."""
        thread_dir = self.persist_directory / thread_id
        thread_dir.mkdir(exist_ok=True)
        return thread_dir / f"{checkpoint_id}.pkl"

    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Save a checkpoint.

        Args:
            config: Configuration with thread_id
            checkpoint: Checkpoint data
            metadata: Optional metadata

        Returns:
            The saved checkpoint configuration
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = checkpoint.get("id", datetime.now().timestamp())

        # Save checkpoint to file
        checkpoint_path = self._get_checkpoint_path(thread_id, str(checkpoint_id))
        with open(checkpoint_path, "wb") as f:
            pickle.dump(checkpoint, f)

        # Update index
        if thread_id not in self._index:
            self._index[thread_id] = {}
        self._index[thread_id][str(checkpoint_id)] = {
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        self._save_index()

        return config

    def get(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve a checkpoint.

        Args:
            config: Configuration with thread_id and checkpoint_id

        Returns:
            Checkpoint data or None if not found
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        checkpoint_id = config.get("configurable", {}).get("checkpoint_id")

        if not checkpoint_id:
            # Get latest checkpoint
            thread_checkpoints = self._index.get(thread_id, {})
            if thread_checkpoints:
                checkpoint_id = list(thread_checkpoints.keys())[-1]
            else:
                return None

        checkpoint_path = self._get_checkpoint_path(thread_id, str(checkpoint_id))

        if checkpoint_path.exists():
            with open(checkpoint_path, "rb") as f:
                return pickle.load(f)

        return None

    def list(
        self,
        config: Optional[Dict[str, Any]] = None,
        before: Optional[Any] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        List checkpoints.

        Args:
            config: Optional configuration with thread_id
            before: Optional timestamp filter
            limit: Optional limit on results

        Returns:
            List of checkpoint configurations
        """
        thread_id = None
        if config:
            thread_id = config.get("configurable", {}).get("thread_id")

        results = []

        threads = [thread_id] if thread_id else self._index.keys()

        for tid in threads:
            thread_checkpoints = self._index.get(tid, {})
            for checkpoint_id, info in thread_checkpoints.items():
                results.append(
                    {
                        "configurable": {
                            "thread_id": tid,
                            "checkpoint_id": checkpoint_id,
                        },
                        "timestamp": info.get("timestamp"),
                    }
                )

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if limit:
            results = results[:limit]

        return results


class CheckpointInspector:
    """
    Utility for inspecting checkpointed workflow states.

    Provides methods to analyze and debug workflow execution.
    """

    def __init__(self, checkpointer: SolarFlareCheckpointer):
        """
        Initialize the inspector.

        Args:
            checkpointer: The checkpointer to inspect
        """
        self.checkpointer = checkpointer

    def get_thread_timeline(
        self,
        thread_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get a timeline of checkpoints for a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            List of checkpoint information in chronological order
        """
        metadata = self.checkpointer.get_metadata(thread_id)

        return [
            {
                "checkpoint_id": m.checkpoint_id,
                "timestamp": m.timestamp.isoformat(),
                "step": m.step,
                "phase": m.phase,
                "metadata": m.metadata,
            }
            for m in metadata
        ]

    def get_phase_summary(
        self,
        thread_id: str,
    ) -> Dict[str, Any]:
        """
        Get a summary of phases executed in a thread.

        Args:
            thread_id: Thread identifier

        Returns:
            Dictionary with phase statistics
        """
        metadata = self.checkpointer.get_metadata(thread_id)

        phases: Dict[str, int] = {}
        total_steps = len(metadata)

        for m in metadata:
            phases[m.phase] = phases.get(m.phase, 0) + 1

        return {
            "thread_id": thread_id,
            "total_steps": total_steps,
            "phases": phases,
            "first_phase": metadata[0].phase if metadata else None,
            "last_phase": metadata[-1].phase if metadata else None,
        }

    def find_long_steps(
        self,
        thread_id: str,
        threshold_ms: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Find steps that took longer than threshold.

        Args:
            thread_id: Thread identifier
            threshold_ms: Time threshold in milliseconds

        Returns:
            List of step information that exceeded threshold
        """
        metadata = self.checkpointer.get_metadata(thread_id)
        long_steps = []

        for i in range(1, len(metadata)):
            duration_ms = (
                metadata[i].timestamp - metadata[i - 1].timestamp
            ).total_seconds() * 1000

            if duration_ms > threshold_ms:
                long_steps.append(
                    {
                        "step": metadata[i].step,
                        "phase": metadata[i].phase,
                        "duration_ms": duration_ms,
                    }
                )

        return long_steps


def create_checkpointer(
    backend: str = "memory",
    persist_directory: str = "./data/checkpoints",
    database_path: str = "./data/conversations.db",
) -> SolarFlareCheckpointer:
    """
    Create a checkpointer with the specified configuration.

    Args:
        backend: Checkpoint backend ('memory', 'sqlite', or 'file')
        persist_directory: Directory for file-based persistence
        database_path: Path for SQLite database

    Returns:
        Configured SolarFlareCheckpointer
    """
    config = CheckpointConfig(
        backend=backend,
        persist_directory=persist_directory,
        database_path=database_path,
    )

    return SolarFlareCheckpointer(config)
