"""
TaskCache: Fingerprint-based caching for task results.

Provides content-addressable caching where tasks are cached by a hash
of their inputs (payload, dependencies). Enables reuse across runs.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List


def compute_fingerprint(
    task_type: str,
    payload: Dict[str, Any],
    dep_fingerprints: Optional[List[str]] = None,
) -> str:
    """
    Compute content-addressable fingerprint for a task.
    
    The fingerprint is a hash of:
    - task_type: Type of task (data, indicators, signals, etc.)
    - payload: Task parameters (sorted for stability)
    - dep_fingerprints: Fingerprints of dependency tasks
    
    Args:
        task_type: Type of task
        payload: Task payload dictionary
        dep_fingerprints: List of dependency fingerprints (optional)
        
    Returns:
        SHA256 hex string (first 16 chars)
    """
    # Build fingerprint input
    fp_input = {
        "task_type": task_type,
        "payload": _normalize_payload(payload, task_type),
    }
    
    if dep_fingerprints:
        fp_input["dependencies"] = sorted(dep_fingerprints)
    
    # Serialize to JSON (sorted keys for stability)
    fp_json = json.dumps(fp_input, sort_keys=True, default=str)
    
    # Hash
    h = hashlib.sha256(fp_json.encode()).hexdigest()
    
    # Return first 16 chars for readability
    return h[:16]


def _normalize_payload(payload: Dict[str, Any], task_type: str = None) -> Dict[str, Any]:
    """
    Normalize payload for fingerprinting.
    
    - Exclude workspace-dependent paths (root, data_root, result_path, output_dir, config_path)
      for all tasks EXCEPT "data" (which must write to specific workspace)
    - Convert Path objects to strings
    - Sort lists where order doesn't matter
    - Handle special cases per task type
    
    Workspace-dependent keys are excluded because they change on every run (temp directories)
    but don't affect task output. This enables caching across grid search runs.
    
    Data tasks are special: they write files to the workspace, so they MUST include "root"
    in their fingerprint to avoid cross-workspace cache hits.
    """
    # Keys to exclude from fingerprinting (workspace-dependent, don't affect results)
    # EXCEPT for "data" tasks which need workspace-specific execution
    EXCLUDED_KEYS = {
        "root",          # Temp workspace path (e.g., /tmp/grid_search_abc123/)
        "data_root",     # Temp workspace path
        "result_path",   # Temp workspace path for intermediate results
        "output_dir",    # Output directory (doesn't affect computation)
        "config_path",   # Temp workspace path (use config_id + config fingerprint instead)
    }
    
    # Data tasks need root in their fingerprint (they write files to workspace)
    if task_type == "data":
        EXCLUDED_KEYS = {"output_dir"}  # Only exclude output_dir for data tasks
    
    normalized = {}
    
    for key, value in payload.items():
        if key in EXCLUDED_KEYS:
            continue  # Skip workspace-dependent keys
            
        if isinstance(value, Path):
            normalized[key] = str(value)
        elif isinstance(value, list) and key in ("instruments",):
            # Sort instrument lists for stability
            normalized[key] = sorted(str(v) for v in value)
        elif isinstance(value, dict):
            normalized[key] = _normalize_payload(value, task_type)
        else:
            normalized[key] = value
    
    return normalized


def compute_config_fingerprint(config_path: Path) -> str:
    """
    Compute fingerprint for a config file by hashing its contents.
    
    Handles both text files (YAML) and binary files (pickle).
    
    Args:
        config_path: Path to config file (YAML or pickle)
        
    Returns:
        SHA256 hex string (first 16 chars)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Read as binary to handle both text and pickle files
    content = config_path.read_bytes()
    h = hashlib.sha256(content).hexdigest()
    return h[:16]


class TaskCache:
    """
    Persistent cache for task results.
    
    Stores task results on disk indexed by fingerprint. Results persist
    across runs, enabling fast re-execution when inputs haven't changed.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize task cache.
        
        Args:
            cache_dir: Directory for cache storage (default: ~/.cache/trading/orchestration)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "trading" / "orchestration"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Track stats
        self.hits = 0
        self.misses = 0
    
    def get(self, fingerprint: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached result by fingerprint.
        
        Args:
            fingerprint: Task fingerprint
            
        Returns:
            Cached result dict or None if not found
        """
        cache_file = self._cache_path(fingerprint)
        
        if not cache_file.exists():
            self.misses += 1
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            self.hits += 1
            return result
        except Exception:
            # Cache corrupted, treat as miss
            self.misses += 1
            return None
    
    def put(self, fingerprint: str, result: Dict[str, Any]) -> None:
        """
        Store result in cache.
        
        Args:
            fingerprint: Task fingerprint
            result: Task result dictionary
        """
        cache_file = self._cache_path(fingerprint)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception:
            # Ignore cache write errors
            pass
    
    def clear(self) -> None:
        """Clear all cached results."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def delete(self, fingerprint: str) -> bool:
        """
        Delete a specific cached result by fingerprint.
        
        Args:
            fingerprint: Task fingerprint
            
        Returns:
            True if file was deleted, False if it didn't exist
        """
        cache_file = self._cache_path(fingerprint)
        
        if cache_file.exists():
            try:
                cache_file.unlink()
                return True
            except Exception:
                return False
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate_pct": hit_rate,
            "cache_dir": str(self.cache_dir),
        }
    
    def _cache_path(self, fingerprint: str) -> Path:
        """Get cache file path for a fingerprint."""
        # Use first 2 chars as subdirectory for better filesystem performance
        subdir = fingerprint[:2]
        return self.cache_dir / subdir / f"{fingerprint}.pkl"
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"TaskCache(dir={self.cache_dir}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"hit_rate={stats['hit_rate_pct']:.1f}%)"
        )
