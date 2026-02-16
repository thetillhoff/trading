"""
Executor: Parallel task execution engine with dependency management.

Executes tasks from a TaskGraph in topological order, maximizing parallelism
while respecting dependencies and resource constraints.
"""
from __future__ import annotations

import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Dict, Optional, List, Any
from datetime import datetime

from .task_graph import TaskGraph, TaskNode
from . import tasks as task_module


def _execute_task_worker(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker function for executing a single task.
    
    This runs in a separate process. Must be a module-level function
    for pickling by ProcessPoolExecutor.
    
    Args:
        task_type: Type of task (data, indicators, signals, simulation, outputs, grid_report)
        payload: Task payload dictionary
        
    Returns:
        Task result dictionary (paths and metadata) with _task_compute_time_s added
    """
    import time
    import traceback
    import gc
    
    # Import here to avoid circular imports and ensure fresh imports in worker
    from . import orchestrator as orch
    
    try:
        # Track actual computation time (excludes queue wait)
        compute_start = time.perf_counter()
        result = orch.run_task(task_type, payload)
        compute_elapsed = time.perf_counter() - compute_start
        
        # Add computation time to result
        result["_task_compute_time_s"] = compute_elapsed
        
        # Force garbage collection after memory-intensive tasks
        if task_type in ("outputs", "grid_report"):
            gc.collect()
        
        return result
    except Exception as e:
        # Capture full traceback before process dies
        error_detail = {
            "_task_compute_time_s": 0.0,
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_traceback": traceback.format_exc(),
        }
        # Re-raise with enhanced error message
        raise RuntimeError(f"Task {task_type} failed: {e}") from e


class Executor:
    """
    Parallel executor for TaskGraph.
    
    Executes tasks respecting dependencies, maximizing parallelism within
    each topological level. Provides progress tracking and error handling.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        verbose: bool = True,
        stream=None,
        progress_file: Optional[Path] = None,
        workers_per_level: Optional[Dict[int, int]] = None,
    ):
        """
        Initialize executor.
        
        Args:
            max_workers: Maximum parallel workers (default: cpu_count - 1, min 1)
            verbose: Print progress messages
            stream: Output stream for messages (default: sys.stdout)
            progress_file: Optional JSON file to write progress updates
            workers_per_level: Optional dict mapping level index to worker count
                              (e.g., {1: 8, 2: 2} = 8 workers for level 1, 2 for level 2)
        """
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count()
            max_workers = max(1, cpu_count - 1 if cpu_count > 1 else 1)
        
        self.max_workers = max_workers
        self.verbose = verbose
        self.stream = stream or sys.stdout
        self.progress_file = progress_file
        self.workers_per_level = workers_per_level or {}
        self._task_timings = {}  # Track task execution times
    
    def execute(
        self,
        graph: TaskGraph,
        cache: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
    ) -> Dict[str, Dict]:
        """
        Execute all tasks in the graph.
        
        Args:
            graph: TaskGraph to execute
            cache: Optional TaskCache for fingerprint-based caching
            checkpoint_path: Optional path to save checkpoints (also used to load if exists)
            
        Returns:
            Dictionary mapping task_id to result
            
        Raises:
            RuntimeError: If any task fails
        """
        if len(graph) == 0:
            return {}
        
        # Try to load checkpoint if it exists
        if checkpoint_path and checkpoint_path.exists():
            try:
                import json
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                # Track old fingerprints for cleanup
                old_fingerprints_by_task = {}
                
                # Only restore completed tasks (not failed ones)
                checkpoint_graph = TaskGraph.from_json(checkpoint_data)
                restored_count = 0
                
                for task_id, checkpoint_node in checkpoint_graph.nodes.items():
                    # Track old fingerprint for potential cleanup
                    if checkpoint_node.fingerprint and task_id in graph.nodes:
                        old_fp = checkpoint_node.fingerprint
                        new_fp = graph.nodes[task_id].fingerprint
                        
                        # If fingerprint changed, mark old cache entry for deletion
                        if old_fp != new_fp:
                            old_fingerprints_by_task[task_id] = old_fp
                    
                    if checkpoint_node.status == "completed" and task_id in graph.nodes:
                        # Restore completed task state
                        graph.nodes[task_id].status = "completed"
                        graph.nodes[task_id].result = checkpoint_node.result
                        restored_count += 1
                
                # Clean up invalidated cache entries
                if cache and old_fingerprints_by_task:
                    deleted_count = 0
                    for task_id, old_fp in old_fingerprints_by_task.items():
                        if cache.delete(old_fp):
                            deleted_count += 1
                    
                    if deleted_count > 0 and self.verbose:
                        self.stream.write(f"Cleaned up {deleted_count} invalidated cache entries (fingerprint changed)\n")
                        self.stream.flush()
                
                if restored_count > 0 and self.verbose:
                    self.stream.write(f"\nRestored {restored_count} completed task(s) from checkpoint\n")
                    failed_count = sum(1 for n in checkpoint_graph.nodes.values() if n.status == "failed")
                    if failed_count > 0:
                        self.stream.write(f"Retrying {failed_count} previously failed task(s)\n")
                    self.stream.flush()
            except Exception as e:
                if self.verbose:
                    self.stream.write(f"Warning: Could not load checkpoint: {e}\n")
                    self.stream.flush()
        
        # Save graph for DAG visualization BEFORE execution
        # This allows the outputs task to read the current graph
        workspace_root = None
        try:
            import pickle
            from pathlib import Path
            
            # Look for 'root' or 'data_root' in any task payload
            for node in graph.nodes.values():
                if "root" in node.payload:
                    workspace_root = Path(node.payload["root"])
                    break
                elif "data_root" in node.payload:
                    workspace_root = Path(node.payload["data_root"])
                    break
            
            if workspace_root:
                graph_path = workspace_root / "task_graph.pkl"
                # Save initial state (will be updated with cache info during execution)
                with open(graph_path, "wb") as f:
                    pickle.dump(graph, f)
        except Exception:
            pass  # Don't fail if we can't save graph
        
        start_time = time.perf_counter()
        
        if self.verbose:
            stats = graph.get_stats()
            self.stream.write(f"\nExecuting TaskGraph: {stats['total']} tasks\n")
            self.stream.write(f"  Workers: {self.max_workers}\n")
            if cache:
                self.stream.write(f"  Cache: enabled\n")
            if checkpoint_path:
                self.stream.write(f"  Checkpoint: {checkpoint_path}\n")
            self.stream.flush()
        
        # Get topological levels for parallel execution
        levels = graph.get_topological_levels()
        
        if self.verbose:
            self.stream.write(f"\nExecution Plan:\n")
            for level_idx, level_tasks in enumerate(levels):
                if level_tasks:
                    task_types = {}
                    for tid in level_tasks:
                        ttype = graph.get_task(tid).task_type
                        task_types[ttype] = task_types.get(ttype, 0) + 1
                    type_str = ", ".join(f"{count} {ttype}" for ttype, count in sorted(task_types.items()))
                    self.stream.write(f"  Level {level_idx}: {len(level_tasks)} tasks ({type_str})\n")
            self.stream.write(f"\n")
            self.stream.flush()
        
        all_results: Dict[str, Dict] = {}
        
        # Execute level by level
        for level_idx, level_tasks in enumerate(levels):
            if not level_tasks:
                continue
            
            level_start = time.perf_counter()
            
            if self.verbose:
                self.stream.write(f"\n{'='*60}\n")
                self.stream.write(f"Level {level_idx}: {len(level_tasks)} tasks\n")
                self.stream.write(f"{'='*60}\n")
                self.stream.flush()
            
            # Check cache for tasks in this level
            tasks_to_run = []
            cached_count = 0
            checkpoint_count = 0
            
            for task_id in level_tasks:
                node = graph.get_task(task_id)
                
                # Check if already completed (from checkpoint)
                if node.status == "completed":
                    checkpoint_count += 1
                    all_results[task_id] = node.result or {}
                    continue
                
                # Check cache if available
                cache_hit = False
                if cache and node.fingerprint:
                    cached_result = cache.get(node.fingerprint)
                    if cached_result is not None:
                        graph.mark_completed(task_id, cached_result)
                        node.was_cached = True  # Mark as cache hit
                        all_results[task_id] = cached_result
                        cached_count += 1
                        cache_hit = True
                
                if not cache_hit:
                    tasks_to_run.append(task_id)
            
            # Summary of skipped tasks
            if self.verbose and (cached_count > 0 or checkpoint_count > 0):
                self.stream.write(
                    f"Skipping {cached_count + checkpoint_count} tasks "
                    f"(cache: {cached_count}, checkpoint: {checkpoint_count})\n"
                )
                self.stream.flush()
            
            if not tasks_to_run:
                level_elapsed = time.perf_counter() - level_start
                if self.verbose:
                    self.stream.write(f"  Level {level_idx} complete (all cached/skipped): {level_elapsed:.1f}s\n")
                    self.stream.flush()
                continue
            
            # Execute tasks in parallel
            level_workers = self.workers_per_level.get(level_idx, self.max_workers)
            
            # Check if this level has memory-intensive tasks
            task_types = {graph.get_task(tid).task_type for tid in tasks_to_run}
            if any(tt in ("outputs", "grid_report") for tt in task_types):
                if level_idx not in self.workers_per_level:
                    level_workers = 1  # Force serial for these tasks unless overridden
            
            level_results = self._execute_level(graph, tasks_to_run, cache, level_workers, level_idx)
            all_results.update(level_results)
            
            # Save graph after each level (for DAG visualization of in-progress execution)
            if workspace_root:
                try:
                    graph_path = workspace_root / "task_graph.pkl"
                    with open(graph_path, "wb") as f:
                        pickle.dump(graph, f)
                except Exception:
                    pass
            
            # Save checkpoint after each level
            if checkpoint_path:
                self._save_checkpoint(graph, checkpoint_path)
            
            level_elapsed = time.perf_counter() - level_start
            if self.verbose:
                self.stream.write(f"  Level {level_idx} complete: {level_elapsed:.1f}s\n")
                self.stream.flush()
        
        total_elapsed = time.perf_counter() - start_time
        
        # Update saved graph with final cache hit information
        if workspace_root:
            try:
                graph_path = workspace_root / "task_graph.pkl"
                with open(graph_path, "wb") as f:
                    pickle.dump(graph, f)
                if self.verbose:
                    self.stream.write(f"Task graph saved to: {graph_path}\n")
            except Exception as e:
                if self.verbose:
                    self.stream.write(f"Warning: Could not save task graph: {e}\n")
        
        if self.verbose:
            stats = graph.get_stats()
            self.stream.write(f"\nExecution complete: {total_elapsed:.1f}s\n")
            self.stream.write(f"  Completed: {stats['completed']}/{stats['total']}\n")
            self.stream.write(f"  Failed: {stats['failed']}\n")
            
            # Log indicator cache stats
            try:
                from ..indicators.disk_cache import get_cache_stats
                cache_stats = get_cache_stats()
                self.stream.write(f"\nIndicator Cache Stats:\n")
                self.stream.write(f"  Cache Directory: {cache_stats['cache_dir']}\n")
                self.stream.write(f"  Cached Files: {cache_stats['files']}\n")
                self.stream.write(f"  Total Size: {cache_stats['size_mb']:.2f} MB\n")
            except Exception:
                pass  # Ignore if cache stats unavailable
            
            self.stream.flush()
        
        if graph.has_failed_tasks():
            raise RuntimeError(f"{graph.get_stats()['failed']} tasks failed")
        
        return all_results
    
    def _execute_level(
        self,
        graph: TaskGraph,
        task_ids: List[str],
        cache: Optional[Any] = None,
        level_workers: Optional[int] = None,
        level_idx: Optional[int] = None,
    ) -> Dict[str, Dict]:
        """Execute all tasks in a single level in parallel."""
        results: Dict[str, Dict] = {}
        
        # Calculate max task ID length for alignment
        max_task_id_len = max((len(tid) for tid in task_ids), default=0)
        
        # Use provided level_workers or fall back to max_workers
        workers = level_workers if level_workers is not None else self.max_workers
        
        # Detect memory-intensive task types
        task_types = {graph.get_task(tid).task_type for tid in task_ids}
        forced_serial = any(tt in ("outputs", "grid_report") for tt in task_types)
        
        # Log worker configuration
        if self.verbose and level_idx is not None:
            if level_idx in self.workers_per_level:
                self.stream.write(f"Using {workers} worker(s) for this level (configured)\n")
            elif forced_serial and workers == 1:
                self.stream.write(f"  Note: Running {', '.join(task_types & {'outputs', 'grid_report'})} serially (1 at a time) to prevent OOM\n")
            self.stream.flush()
        
        # Submit all tasks to executor
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Map future to task_id and track start times
            future_to_task: Dict[Future, str] = {}
            task_start_times: Dict[str, float] = {}
            
            if self.verbose:
                self.stream.write(f"Submitting {len(task_ids)} tasks to {workers} workers...\n")
                self.stream.flush()
            
            for task_id in task_ids:
                node = graph.get_task(task_id)
                graph.mark_status(task_id, "running")
                task_start_times[task_id] = time.perf_counter()
                
                future = executor.submit(
                    _execute_task_worker,
                    node.task_type,
                    node.payload,
                )
                future_to_task[future] = task_id
            
            if self.verbose:
                self.stream.write(f"All tasks submitted. Waiting for completion...\n")
                self.stream.flush()
            
            # Track completion progress
            completed_count = 0
            total_in_level = len(task_ids)
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                task_id = future_to_task[future]
                node = graph.get_task(task_id)
                elapsed = time.perf_counter() - task_start_times[task_id]
                completed_count += 1
                
                try:
                    result = future.result()
                    graph.mark_completed(task_id, result)
                    results[task_id] = result
                    
                    # Extract computation time and store in node
                    compute_time = result.get("_task_compute_time_s", elapsed)
                    node.compute_time_s = compute_time
                    
                    # Check if this was a disk cache hit (for indicators)
                    if result.get("_was_disk_cached", False):
                        node.was_cached = True
                    
                    wait_time = elapsed - compute_time
                    
                    # Track timing
                    self._task_timings[task_id] = compute_time  # Track compute time for ETA
                    
                    # Cache result if fingerprint available
                    if cache and node.fingerprint:
                        cache.put(node.fingerprint, result)
                    
                    # Write progress update
                    self._write_progress_update(graph)
                    
                    if self.verbose:
                        # Format with aligned columns
                        progress_str = f"[{completed_count}/{total_in_level}]"
                        task_id_padded = task_id.ljust(max_task_id_len)
                        
                        # Show progress with compute vs wait time (aligned)
                        if wait_time > 0.5:  # Show wait time if significant
                            self.stream.write(
                                f"  {progress_str} {task_id_padded}  "
                                f"compute: {compute_time:>6.1f}s  wait: {wait_time:>6.1f}s\n"
                            )
                        else:
                            self.stream.write(
                                f"  {progress_str} {task_id_padded}  "
                                f"compute: {compute_time:>6.1f}s\n"
                            )
                        self.stream.flush()
                
                except Exception as e:
                    graph.mark_failed(task_id)
                    
                    # Enhanced error reporting
                    error_type = type(e).__name__
                    error_msg = str(e)
                    
                    # Check if it's an OOM-related crash
                    if "BrokenProcessPool" in error_type or "BrokenPipeError" in error_type:
                        error_msg = f"{error_msg} (likely OOM - worker killed, try reducing --workers or Docker memory)"
                    
                    if self.verbose:
                        progress_str = f"[{completed_count}/{total_in_level}]"
                        task_id_padded = task_id.ljust(max_task_id_len)
                        self.stream.write(
                            f"  {progress_str} {task_id_padded} FAILED after {elapsed:.1f}s: {error_type}: {error_msg}\n"
                        )
                        self.stream.flush()
                    # Continue to mark other tasks, will raise at end
        
        return results
    
    def _save_checkpoint(self, graph: TaskGraph, checkpoint_path: Path) -> None:
        """Save execution checkpoint."""
        try:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'w') as f:
                f.write(graph.to_json())
        except Exception as e:
            if self.verbose:
                self.stream.write(f"  Warning: Failed to save checkpoint: {e}\n")
                self.stream.flush()
    
    def _write_progress_update(self, graph: TaskGraph) -> None:
        """Write progress update to JSON file if configured."""
        if not self.progress_file:
            return
        
        try:
            import json
            from datetime import datetime
            
            stats = graph.get_stats()
            
            # Calculate average task time
            avg_task_time = (
                sum(self._task_timings.values()) / len(self._task_timings)
                if self._task_timings else 0
            )
            
            # Estimate time remaining
            remaining_tasks = stats["pending"] + stats["running"]
            eta_seconds = remaining_tasks * avg_task_time if avg_task_time > 0 else 0
            
            progress = {
                "timestamp": datetime.now().isoformat(),
                "total_tasks": stats["total"],
                "completed": stats["completed"],
                "running": stats["running"],
                "pending": stats["pending"],
                "failed": stats["failed"],
                "completion_pct": (
                    stats["completed"] / stats["total"] * 100 
                    if stats["total"] > 0 else 0
                ),
                "avg_task_time_s": round(avg_task_time, 2),
                "eta_seconds": round(eta_seconds, 0),
                "eta_minutes": round(eta_seconds / 60, 1),
            }
            
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        
        except Exception:
            # Silently ignore progress write errors
            pass
