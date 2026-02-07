"""
Tests for Checkpoint: save/restore execution state.
"""
import pytest
from datetime import datetime
from pathlib import Path
from core.orchestration.checkpoint import Checkpoint, find_latest_checkpoint
from core.orchestration.task_graph import TaskGraph, TaskNode


class TestCheckpoint:
    """Tests for Checkpoint save and restore."""
    
    def test_create_checkpoint(self):
        """Test creating a checkpoint."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        
        timestamp = datetime.now()
        checkpoint = Checkpoint(graph=graph, timestamp=timestamp)
        
        assert checkpoint.graph == graph
        assert checkpoint.timestamp == timestamp
        assert checkpoint.metadata == {}
    
    def test_checkpoint_with_metadata(self):
        """Test checkpoint with custom metadata."""
        graph = TaskGraph()
        timestamp = datetime.now()
        metadata = {"run_id": "12345", "user": "test"}
        
        checkpoint = Checkpoint(graph=graph, timestamp=timestamp, metadata=metadata)
        
        assert checkpoint.metadata == metadata
    
    def test_save_checkpoint(self, tmp_path):
        """Test saving checkpoint to disk."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={"key": "value"}))
        graph.mark_completed("task1", {"output": "result"})
        
        checkpoint = Checkpoint(graph=graph, timestamp=datetime.now())
        checkpoint_path = tmp_path / "checkpoint.json"
        
        checkpoint.save(checkpoint_path)
        
        assert checkpoint_path.exists()
    
    def test_load_checkpoint(self, tmp_path):
        """Test loading checkpoint from disk."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={"key": "value"}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.mark_completed("task1", {"output": "result"})
        
        timestamp = datetime.now()
        checkpoint = Checkpoint(graph=graph, timestamp=timestamp)
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint.save(checkpoint_path)
        
        # Load
        loaded = Checkpoint.load(checkpoint_path)
        
        assert len(loaded.graph) == 2
        assert loaded.graph.get_status("task1") == "completed"
        assert loaded.graph.get_task("task1").result == {"output": "result"}
        assert loaded.graph.get_status("task2") == "pending"
    
    def test_load_missing_checkpoint_raises(self, tmp_path):
        """Test that loading missing checkpoint raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Checkpoint.load(tmp_path / "nonexistent.json")
    
    def test_checkpoint_roundtrip_preserves_state(self, tmp_path):
        """Test that save/load roundtrip preserves graph state."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={"instruments": ["AAPL"]}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        
        graph.mark_completed("task1", {"data_dir": "/path/to/data"})
        graph.mark_status("task2", "running")
        
        checkpoint = Checkpoint(
            graph=graph,
            timestamp=datetime.now(),
            metadata={"run_id": "test123"}
        )
        
        checkpoint_path = tmp_path / "checkpoint.json"
        checkpoint.save(checkpoint_path)
        
        loaded = Checkpoint.load(checkpoint_path)
        
        assert len(loaded.graph) == 3
        assert loaded.graph.get_status("task1") == "completed"
        assert loaded.graph.get_task("task1").result == {"data_dir": "/path/to/data"}
        assert loaded.graph.get_status("task2") == "running"
        assert loaded.graph.get_status("task3") == "pending"
        assert loaded.metadata == {"run_id": "test123"}
    
    def test_get_progress(self):
        """Test getting execution progress."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        graph.add_task(TaskNode(id="task4", task_type="simulation", payload={}, depends_on=["task3"]))
        
        graph.mark_completed("task1", {})
        graph.mark_completed("task2", {})
        
        checkpoint = Checkpoint(graph=graph, timestamp=datetime.now())
        progress = checkpoint.get_progress()
        
        assert progress["total_tasks"] == 4
        assert progress["completed"] == 2
        assert progress["pending"] == 2
        assert progress["completion_pct"] == 50.0
    
    def test_atomic_save(self, tmp_path):
        """Test that checkpoint save is atomic (uses temp file + rename)."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        
        checkpoint = Checkpoint(graph=graph, timestamp=datetime.now())
        checkpoint_path = tmp_path / "checkpoint.json"
        
        checkpoint.save(checkpoint_path)
        
        # Verify no .tmp file left behind
        assert checkpoint_path.exists()
        assert not checkpoint_path.with_suffix('.tmp').exists()


class TestFindLatestCheckpoint:
    """Tests for finding latest checkpoint."""
    
    def test_find_latest_checkpoint_empty_dir(self, tmp_path):
        """Test finding checkpoint in empty directory."""
        result = find_latest_checkpoint(tmp_path)
        assert result is None
    
    def test_find_latest_checkpoint_nonexistent_dir(self, tmp_path):
        """Test finding checkpoint in nonexistent directory."""
        result = find_latest_checkpoint(tmp_path / "nonexistent")
        assert result is None
    
    def test_find_latest_checkpoint_single(self, tmp_path):
        """Test finding single checkpoint."""
        checkpoint_path = tmp_path / "checkpoint_001.json"
        checkpoint_path.write_text("{}")
        
        result = find_latest_checkpoint(tmp_path)
        
        assert result == checkpoint_path
    
    def test_find_latest_checkpoint_multiple(self, tmp_path):
        """Test finding most recent checkpoint among multiple."""
        import time
        
        # Create checkpoints with different timestamps
        cp1 = tmp_path / "checkpoint_001.json"
        cp1.write_text("{}")
        time.sleep(0.1)
        
        cp2 = tmp_path / "checkpoint_002.json"
        cp2.write_text("{}")
        time.sleep(0.1)
        
        cp3 = tmp_path / "checkpoint_003.json"
        cp3.write_text("{}")
        
        result = find_latest_checkpoint(tmp_path)
        
        # Should return the most recently modified
        assert result == cp3
    
    def test_find_latest_checkpoint_custom_pattern(self, tmp_path):
        """Test finding checkpoint with custom pattern."""
        # Create files with different patterns
        (tmp_path / "checkpoint_001.json").write_text("{}")
        (tmp_path / "backup_001.json").write_text("{}")
        
        # Default pattern should find checkpoint_*
        result = find_latest_checkpoint(tmp_path)
        assert result.name.startswith("checkpoint_")
        
        # Custom pattern should find backup_*
        result = find_latest_checkpoint(tmp_path, pattern="backup_*.json")
        assert result.name.startswith("backup_")
