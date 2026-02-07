"""
Checkpoint: Save and restore execution state for resumability.

Enables resuming failed grid searches from the last completed task
without recomputing everything.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .task_graph import TaskGraph


@dataclass
class Checkpoint:
    """
    Execution checkpoint for resumability.
    
    Contains the full TaskGraph state including which tasks are completed,
    their results, and the current execution status.
    """
    
    graph: TaskGraph
    timestamp: datetime
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def save(self, path: Path) -> None:
        """
        Save checkpoint to disk.
        
        Args:
            path: Path to checkpoint file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize graph to JSON
        graph_json = self.graph.to_json()
        
        # Build checkpoint data
        checkpoint_data = {
            "version": "1.0",
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "graph": graph_json,
        }
        
        # Write atomically using temp file
        temp_path = path.with_suffix('.tmp')
        try:
            import json
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Atomic rename
            temp_path.replace(path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
    
    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        """
        Load checkpoint from disk.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Restored Checkpoint
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        version = data.get("version", "1.0")
        if version != "1.0":
            raise ValueError(f"Unsupported checkpoint version: {version}")
        
        # Restore graph
        graph = TaskGraph.from_json(data["graph"])
        
        # Restore timestamp
        timestamp = datetime.fromisoformat(data["timestamp"])
        
        # Restore metadata
        metadata = data.get("metadata", {})
        
        return cls(
            graph=graph,
            timestamp=timestamp,
            metadata=metadata,
        )
    
    def get_progress(self) -> Dict[str, any]:
        """Get execution progress statistics."""
        stats = self.graph.get_stats()
        
        completed_pct = (stats["completed"] / stats["total"] * 100) if stats["total"] > 0 else 0
        
        return {
            "total_tasks": stats["total"],
            "completed": stats["completed"],
            "pending": stats["pending"],
            "running": stats["running"],
            "failed": stats["failed"],
            "completion_pct": completed_pct,
            "timestamp": self.timestamp.isoformat(),
        }
    
    def __repr__(self) -> str:
        progress = self.get_progress()
        return (
            f"Checkpoint({progress['completed']}/{progress['total_tasks']} tasks, "
            f"{progress['completion_pct']:.1f}% complete, "
            f"saved {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')})"
        )


def find_latest_checkpoint(directory: Path, pattern: str = "checkpoint_*.json") -> Optional[Path]:
    """
    Find the most recent checkpoint file in a directory.
    
    Args:
        directory: Directory to search
        pattern: Glob pattern for checkpoint files
        
    Returns:
        Path to latest checkpoint or None if none found
    """
    if not directory.exists():
        return None
    
    checkpoints = list(directory.glob(pattern))
    if not checkpoints:
        return None
    
    # Sort by modification time, most recent first
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return checkpoints[0]
