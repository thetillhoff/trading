"""
TaskGraph: Declarative DAG for task orchestration.

Provides dependency management, topological sorting, and serialization
for checkpoint/resume functionality.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Literal, Optional, Set
from collections import deque


TaskStatus = Literal["pending", "running", "completed", "failed"]


@dataclass
class TaskNode:
    """A single task in the execution graph."""
    
    id: str
    task_type: str
    payload: Dict
    depends_on: List[str] = field(default_factory=list)
    status: TaskStatus = "pending"
    result: Optional[Dict] = None
    fingerprint: Optional[str] = None
    was_cached: bool = False  # True if result came from cache
    compute_time_s: float = 0.0  # Computation time in seconds
    
    def to_dict(self) -> Dict:
        """Serialize to dict for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> TaskNode:
        """Deserialize from dict."""
        return cls(**data)


class TaskGraph:
    """
    Directed Acyclic Graph (DAG) of tasks with dependency management.
    
    Provides:
    - Add tasks with explicit dependencies
    - Topological sort for execution order
    - Topological levels for parallel execution
    - Cycle detection
    - Status tracking
    - Serialization for checkpointing
    """
    
    def __init__(self):
        self.nodes: Dict[str, TaskNode] = {}
        self._topological_order: Optional[List[str]] = None
        self._topological_levels: Optional[List[List[str]]] = None
    
    def add_task(self, node: TaskNode) -> None:
        """
        Add a task to the graph.
        
        Args:
            node: TaskNode to add
            
        Raises:
            ValueError: If task ID already exists or creates a cycle
        """
        if node.id in self.nodes:
            raise ValueError(f"Task ID already exists: {node.id}")
        
        # Verify dependencies exist
        for dep_id in node.depends_on:
            if dep_id not in self.nodes:
                raise ValueError(f"Task {node.id} depends on non-existent task: {dep_id}")
        
        self.nodes[node.id] = node
        
        # Invalidate cached topological order
        self._topological_order = None
        self._topological_levels = None
        
        # Verify no cycles
        try:
            self.get_execution_order()
        except ValueError as e:
            # Remove node if it creates a cycle
            del self.nodes[node.id]
            raise e
    
    def get_task(self, task_id: str) -> TaskNode:
        """Get task by ID."""
        if task_id not in self.nodes:
            raise ValueError(f"Task not found: {task_id}")
        return self.nodes[task_id]
    
    def get_status(self, task_id: str) -> TaskStatus:
        """Get task status."""
        return self.get_task(task_id).status
    
    def mark_status(self, task_id: str, status: TaskStatus) -> None:
        """Update task status."""
        self.get_task(task_id).status = status
    
    def mark_completed(self, task_id: str, result: Dict) -> None:
        """Mark task as completed with result."""
        task = self.get_task(task_id)
        task.status = "completed"
        task.result = result
    
    def mark_failed(self, task_id: str) -> None:
        """Mark task as failed."""
        self.get_task(task_id).status = "failed"
    
    def get_execution_order(self) -> List[str]:
        """
        Get topologically sorted task execution order.
        
        Returns:
            List of task IDs in execution order
            
        Raises:
            ValueError: If graph contains a cycle
        """
        if self._topological_order is not None:
            return self._topological_order
        
        # Kahn's algorithm for topological sort
        in_degree = {task_id: 0 for task_id in self.nodes}
        
        # Calculate in-degrees
        for node in self.nodes.values():
            for dep_id in node.depends_on:
                in_degree[dep_id] += 0  # Ensure dep exists
            for dep_id in node.depends_on:
                if dep_id in in_degree:
                    in_degree[node.id] = in_degree.get(node.id, 0) + 1
        
        # Queue of nodes with no dependencies
        queue = deque([task_id for task_id, deg in in_degree.items() if deg == 0])
        order = []
        
        while queue:
            task_id = queue.popleft()
            order.append(task_id)
            
            # Find tasks that depend on this one
            for node in self.nodes.values():
                if task_id in node.depends_on:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        queue.append(node.id)
        
        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle")
        
        self._topological_order = order
        return order
    
    def get_topological_levels(self) -> List[List[str]]:
        """
        Get tasks grouped by dependency level for parallel execution.
        
        Level 0: Tasks with no dependencies
        Level 1: Tasks depending only on level 0
        Level N: Tasks depending only on levels 0..N-1
        
        Returns:
            List of lists, where each inner list contains task IDs at that level
        """
        if self._topological_levels is not None:
            return self._topological_levels
        
        levels: List[List[str]] = []
        assigned_level: Dict[str, int] = {}
        
        # Get topological order first (validates no cycles)
        execution_order = self.get_execution_order()
        
        for task_id in execution_order:
            node = self.nodes[task_id]
            
            if not node.depends_on:
                # No dependencies: level 0
                level = 0
            else:
                # Level is max(dependency levels) + 1
                level = max(assigned_level[dep_id] for dep_id in node.depends_on) + 1
            
            assigned_level[task_id] = level
            
            # Ensure levels list is large enough
            while len(levels) <= level:
                levels.append([])
            
            levels[level].append(task_id)
        
        self._topological_levels = levels
        return levels
    
    def get_ready_tasks(self) -> List[str]:
        """
        Get tasks that are ready to run (dependencies satisfied, status=pending).
        
        Returns:
            List of task IDs ready to execute
        """
        ready = []
        for task_id, node in self.nodes.items():
            if node.status != "pending":
                continue
            
            # Check if all dependencies are completed
            deps_satisfied = all(
                self.nodes[dep_id].status == "completed"
                for dep_id in node.depends_on
            )
            
            if deps_satisfied:
                ready.append(task_id)
        
        return ready
    
    def is_complete(self) -> bool:
        """Check if all tasks are completed."""
        return all(node.status == "completed" for node in self.nodes.values())
    
    def has_failed_tasks(self) -> bool:
        """Check if any tasks have failed."""
        return any(node.status == "failed" for node in self.nodes.values())
    
    def get_stats(self) -> Dict[str, int]:
        """Get task statistics by status."""
        stats = {
            "total": len(self.nodes),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
        }
        for node in self.nodes.values():
            stats[node.status] += 1
        return stats
    
    def to_json(self) -> str:
        """Serialize graph to JSON for checkpointing."""
        data = {
            "nodes": {task_id: node.to_dict() for task_id, node in self.nodes.items()}
        }
        return json.dumps(data, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_str: str) -> TaskGraph:
        """Deserialize graph from JSON."""
        data = json.loads(json_str)
        graph = cls()
        
        # Add nodes in topological order to avoid dependency errors
        nodes_data = data["nodes"]
        added = set()
        
        while len(added) < len(nodes_data):
            made_progress = False
            
            for task_id, node_data in nodes_data.items():
                if task_id in added:
                    continue
                
                # Check if all dependencies are added
                deps = node_data.get("depends_on", [])
                if all(dep_id in added for dep_id in deps):
                    node = TaskNode.from_dict(node_data)
                    graph.nodes[node.id] = node
                    added.add(task_id)
                    made_progress = True
            
            if not made_progress:
                raise ValueError("Cannot deserialize graph: unresolved dependencies or cycle")
        
        return graph
    
    def __len__(self) -> int:
        """Number of tasks in the graph."""
        return len(self.nodes)
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TaskGraph({stats['total']} tasks: "
            f"{stats['completed']} completed, "
            f"{stats['pending']} pending, "
            f"{stats['running']} running, "
            f"{stats['failed']} failed)"
        )
