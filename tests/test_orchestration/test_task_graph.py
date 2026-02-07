"""
Tests for TaskGraph: dependency management, topological sort, serialization.
"""
import pytest
from core.orchestration.task_graph import TaskGraph, TaskNode


class TestTaskNode:
    """Tests for TaskNode dataclass."""
    
    def test_create_node(self):
        """Test creating a task node."""
        node = TaskNode(
            id="task1",
            task_type="data",
            payload={"key": "value"},
            depends_on=["task0"],
        )
        
        assert node.id == "task1"
        assert node.task_type == "data"
        assert node.payload == {"key": "value"}
        assert node.depends_on == ["task0"]
        assert node.status == "pending"
        assert node.result is None
    
    def test_node_serialization(self):
        """Test node to_dict and from_dict."""
        node = TaskNode(
            id="task1",
            task_type="data",
            payload={"key": "value"},
            depends_on=["task0"],
            status="completed",
            result={"output": "path"},
        )
        
        data = node.to_dict()
        restored = TaskNode.from_dict(data)
        
        assert restored.id == node.id
        assert restored.task_type == node.task_type
        assert restored.payload == node.payload
        assert restored.depends_on == node.depends_on
        assert restored.status == node.status
        assert restored.result == node.result


class TestTaskGraph:
    """Tests for TaskGraph DAG management."""
    
    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = TaskGraph()
        assert len(graph) == 0
        assert graph.is_complete()
        assert not graph.has_failed_tasks()
    
    def test_add_single_task(self):
        """Test adding a single task."""
        graph = TaskGraph()
        node = TaskNode(id="task1", task_type="data", payload={})
        
        graph.add_task(node)
        
        assert len(graph) == 1
        assert "task1" in graph.nodes
        assert graph.get_task("task1") == node
    
    def test_add_duplicate_task_raises(self):
        """Test that duplicate task IDs raise ValueError."""
        graph = TaskGraph()
        node1 = TaskNode(id="task1", task_type="data", payload={})
        node2 = TaskNode(id="task1", task_type="indicators", payload={})
        
        graph.add_task(node1)
        
        with pytest.raises(ValueError, match="already exists"):
            graph.add_task(node2)
    
    def test_add_task_with_missing_dependency_raises(self):
        """Test that tasks with missing dependencies raise ValueError."""
        graph = TaskGraph()
        node = TaskNode(id="task1", task_type="data", payload={}, depends_on=["nonexistent"])
        
        with pytest.raises(ValueError, match="non-existent task"):
            graph.add_task(node)
    
    def test_add_tasks_with_dependencies(self):
        """Test adding tasks with proper dependencies."""
        graph = TaskGraph()
        
        node1 = TaskNode(id="task1", task_type="data", payload={})
        node2 = TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"])
        node3 = TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"])
        
        graph.add_task(node1)
        graph.add_task(node2)
        graph.add_task(node3)
        
        assert len(graph) == 3
    
    def test_topological_sort_simple(self):
        """Test topological sort with simple linear dependencies."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        
        order = graph.get_execution_order()
        
        assert order == ["task1", "task2", "task3"]
    
    def test_topological_sort_parallel(self):
        """Test topological sort with parallel tasks."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="data", task_type="data", payload={}))
        graph.add_task(TaskNode(id="ind1", task_type="indicators", payload={}, depends_on=["data"]))
        graph.add_task(TaskNode(id="ind2", task_type="indicators", payload={}, depends_on=["data"]))
        graph.add_task(TaskNode(id="sig", task_type="signals", payload={}, depends_on=["ind1", "ind2"]))
        
        order = graph.get_execution_order()
        
        assert order[0] == "data"
        assert set(order[1:3]) == {"ind1", "ind2"}
        assert order[3] == "sig"
    
    def test_topological_levels(self):
        """Test topological level grouping for parallel execution."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="data", task_type="data", payload={}))
        graph.add_task(TaskNode(id="ind1", task_type="indicators", payload={}, depends_on=["data"]))
        graph.add_task(TaskNode(id="ind2", task_type="indicators", payload={}, depends_on=["data"]))
        graph.add_task(TaskNode(id="sig", task_type="signals", payload={}, depends_on=["ind1", "ind2"]))
        
        levels = graph.get_topological_levels()
        
        assert len(levels) == 3
        assert levels[0] == ["data"]
        assert set(levels[1]) == {"ind1", "ind2"}
        assert levels[2] == ["sig"]
    
    def test_cycle_detection(self):
        """Test that cycles are detected and raise ValueError."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        
        # Manually create a cycle by modifying the graph structure (simulates corruption)
        # task2 -> task1, task3 -> task2, task2 -> task3 (cycle)
        graph.nodes["task2"].depends_on.append("task3")
        graph._topological_order = None  # Invalidate cache to force recomputation
        
        with pytest.raises(ValueError, match="cycle"):
            graph.get_execution_order()
    
    def test_status_management(self):
        """Test task status tracking."""
        graph = TaskGraph()
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        
        assert graph.get_status("task1") == "pending"
        
        graph.mark_status("task1", "running")
        assert graph.get_status("task1") == "running"
        
        graph.mark_completed("task1", {"output": "result"})
        assert graph.get_status("task1") == "completed"
        assert graph.get_task("task1").result == {"output": "result"}
        
        graph.mark_failed("task1")
        assert graph.get_status("task1") == "failed"
    
    def test_get_ready_tasks(self):
        """Test getting tasks that are ready to run."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        
        # Initially only task1 is ready
        ready = graph.get_ready_tasks()
        assert ready == ["task1"]
        
        # After task1 completes, task2 is ready
        graph.mark_completed("task1", {})
        ready = graph.get_ready_tasks()
        assert ready == ["task2"]
        
        # After task2 completes, task3 is ready
        graph.mark_completed("task2", {})
        ready = graph.get_ready_tasks()
        assert ready == ["task3"]
        
        # After task3 completes, no tasks are ready
        graph.mark_completed("task3", {})
        ready = graph.get_ready_tasks()
        assert ready == []
    
    def test_is_complete(self):
        """Test checking if all tasks are completed."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        
        assert not graph.is_complete()
        
        graph.mark_completed("task1", {})
        assert not graph.is_complete()
        
        graph.mark_completed("task2", {})
        assert graph.is_complete()
    
    def test_has_failed_tasks(self):
        """Test checking for failed tasks."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        
        assert not graph.has_failed_tasks()
        
        graph.mark_completed("task1", {})
        assert not graph.has_failed_tasks()
        
        graph.mark_failed("task2")
        assert graph.has_failed_tasks()
    
    def test_get_stats(self):
        """Test getting task statistics."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.add_task(TaskNode(id="task3", task_type="signals", payload={}, depends_on=["task2"]))
        
        stats = graph.get_stats()
        assert stats["total"] == 3
        assert stats["pending"] == 3
        assert stats["completed"] == 0
        
        graph.mark_status("task1", "running")
        stats = graph.get_stats()
        assert stats["running"] == 1
        assert stats["pending"] == 2
        
        graph.mark_completed("task1", {})
        graph.mark_completed("task2", {})
        stats = graph.get_stats()
        assert stats["completed"] == 2
        assert stats["pending"] == 1
    
    def test_serialization_roundtrip(self):
        """Test serializing and deserializing a graph."""
        graph = TaskGraph()
        
        graph.add_task(TaskNode(id="task1", task_type="data", payload={"key": "value"}))
        graph.add_task(TaskNode(id="task2", task_type="indicators", payload={}, depends_on=["task1"]))
        graph.mark_completed("task1", {"output": "result"})
        
        # Serialize
        json_str = graph.to_json()
        
        # Deserialize
        restored = TaskGraph.from_json(json_str)
        
        assert len(restored) == 2
        assert "task1" in restored.nodes
        assert "task2" in restored.nodes
        assert restored.get_task("task1").status == "completed"
        assert restored.get_task("task1").result == {"output": "result"}
        assert restored.get_task("task2").depends_on == ["task1"]
