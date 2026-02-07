"""
Orchestrator for the pipeline: runs tasks in order; payloads are path references only.

Does not hold result data or input data in memory—only path references.
Caller (single-eval CLI or grid-search) builds the list of task specs and runs them.

Supports both legacy tuple-based task lists and new TaskGraph-based execution.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from . import tasks
from .task_graph import TaskGraph, TaskNode
from .cache import compute_fingerprint, compute_config_fingerprint
from .executor import Executor
from .checkpoint import Checkpoint, find_latest_checkpoint


def _task_description(task_type: str, payload: Dict[str, Any]) -> str:
    """One-line description for progress log."""
    if task_type == "data":
        inst = payload.get("instruments", [])
        n = len(inst) if isinstance(inst, list) else 0
        return f"instruments={n} ({payload.get('start_date', '')}–{payload.get('end_date', '')})"
    if task_type == "indicators":
        return f"{payload.get('instrument', '?')} {payload.get('indicator_type', '?')} {payload.get('spec_key', '')[:8]}"
    if task_type == "signals":
        year_suffix = f"_{payload['year']}" if payload.get("year") else ""
        return f"{payload.get('config_id', '?')} / {payload.get('instrument', '?')}{year_suffix}"
    if task_type == "merge_signals":
        return f"merge {payload.get('config_id', '?')} / {payload.get('instrument', '?')}"
    if task_type == "simulation":
        return payload.get("config_id", "?")
    if task_type == "outputs":
        return Path(payload.get("output_dir", "")).name or "?"
    if task_type == "grid_report":
        return f"{len(payload.get('result_paths', []))} configs"
    return task_type


def run_task(task_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one task by type. Payload contains path refs and small params; returns path refs.

    Task types: data, indicators, signals, simulation, outputs, grid_report.
    """
    if task_type == "data":
        data_dir_path = tasks.run_data_task(
            root=Path(payload["root"]),
            instruments=payload["instruments"],
            start_date=payload["start_date"],
            end_date=payload["end_date"],
            lookback_days=payload["lookback_days"],
            step_days=payload["step_days"],
            min_history_days=payload["min_history_days"],
            column=payload.get("column", "Close"),
        )
        return {"data_dir": str(data_dir_path)}
    if task_type == "indicators":
        result = tasks.run_indicators_task(
            root=Path(payload["root"]),
            instrument=payload["instrument"],
            indicator_type=payload["indicator_type"],
            spec_key=payload["spec_key"],
            params=payload["params"],
        )
        # result is now a dict with indicator_path and _was_disk_cached
        return result
    if task_type == "signals":
        out = tasks.run_signals_task(
            root=Path(payload["root"]),
            config_id=payload["config_id"],
            instrument=payload["instrument"],
            config=payload.get("config"),
            config_path=Path(payload["config_path"]) if payload.get("config_path") else None,
            date_range_start=payload.get("date_range_start"),
            date_range_end=payload.get("date_range_end"),
            year=payload.get("year"),
        )
        return {"signals_path": str(out)}
    if task_type == "merge_signals":
        out = tasks.run_merge_signals_task(
            root=Path(payload["root"]),
            config_id=payload["config_id"],
            instrument=payload["instrument"],
        )
        return {"signals_path": str(out)}
    if task_type == "simulation":
        out = tasks.run_simulation_task(
            root=Path(payload["root"]),
            config_id=payload["config_id"],
            instruments=payload["instruments"],
            config=payload.get("config"),
            config_path=Path(payload["config_path"]) if payload.get("config_path") else None,
        )
        return {"result_path": str(out)}
    if task_type == "outputs":
        out = tasks.run_outputs_task(
            result_path=Path(payload["result_path"]),
            output_dir=Path(payload["output_dir"]),
            data_root=Path(payload["data_root"]) if payload.get("data_root") else None,
        )
        return {"output_dir": str(out)}
    if task_type == "grid_report":
        from .tasks import run_grid_report_task
        out = run_grid_report_task(
            result_paths=[Path(p) for p in payload["result_paths"]],
            summary_dir=Path(payload["summary_dir"]),
        )
        return {"summary_dir": str(out)}
    raise ValueError(f"Unknown task_type: {task_type}")


def build_data_task(
    workspace_root: Path,
    config: Any,
    min_history_days: int = 100,
    instruments: List[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Build the single Data task. If instruments is provided, use it; else config.instruments."""
    inst_list = instruments if instruments is not None else config.instruments
    return (
        "data",
        {
            "root": str(Path(workspace_root)),
            "instruments": inst_list,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "lookback_days": config.lookback_days,
            "step_days": config.step_days,
            "min_history_days": min_history_days,
            "column": getattr(config, "column", "Close"),
        },
    )


def build_single_eval_tasks_after_data(
    workspace_root: Path,
    output_dir: Path,
    config: Any,
    config_id: str = None,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build task list for Indicators -> Signals -> Simulation -> Outputs (after Data task has run).

    Requires workspace_root to already contain data/prep_manifest.pkl. Writes config to
    workspace_root/configs/<config_id>/config.pkl.
    """
    import pickle
    from .contract import config_staging_dir, config_result_path
    from .indicator_spec import indicator_specs_needed_for_config
    from .tasks import load_prep_manifest, _params_for_indicator_type

    workspace_root = Path(workspace_root)
    output_dir = Path(output_dir)
    config_id = config_id or config.name
    staging = config_staging_dir(workspace_root, config_id)
    staging.mkdir(parents=True, exist_ok=True)
    config_pkl = staging / "config.pkl"
    with open(config_pkl, "wb") as f:
        pickle.dump(config, f)
    manifest = load_prep_manifest(workspace_root)
    prep = manifest["prep"]
    tasks_list = []
    for indicator_type, spec_key in indicator_specs_needed_for_config(config):
        params = _params_for_indicator_type(config, indicator_type)
        for inst in prep.instruments:
            tasks_list.append(
                (
                    "indicators",
                    {
                        "root": str(workspace_root),
                        "instrument": inst,
                        "indicator_type": indicator_type,
                        "spec_key": spec_key,
                        "params": params,
                    },
                )
            )
    for inst in prep.instruments:
        tasks_list.append(
            (
                "signals",
                {
                    "root": str(workspace_root),
                    "config_id": config_id,
                    "instrument": inst,
                    "config_path": str(config_pkl),
                },
            )
        )
    tasks_list.append(
        (
            "simulation",
            {
                "root": str(workspace_root),
                "config_id": config_id,
                "instruments": prep.instruments,
                "config_path": str(config_pkl),
            },
        )
    )
    result_path = config_result_path(workspace_root, config_id)
    tasks_list.append(
        (
            "outputs",
            {
                "result_path": str(result_path),
                "output_dir": str(output_dir),
                "data_root": str(workspace_root),
            },
        )
    )
    return tasks_list


def build_grid_search_tasks_after_data(
    workspace_root: Path,
    configs: List[Any],
    results_root: Path,
    summary_dir: Path,
    config_file_map: Dict[str, Path],
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Build task list for grid search after Data task has run: Indicators (deduplicated),
    then per-config Signals -> Simulation -> Outputs, then GridReport.

    config_file_map: config.name -> relative path from configs/ (for output dir).
    """
    import pickle
    from .contract import config_staging_dir, config_result_path
    from .indicator_spec import indicator_specs_needed_for_config
    from .tasks import load_prep_manifest, _params_for_indicator_type

    workspace_root = Path(workspace_root)
    results_root = Path(results_root)
    summary_dir = Path(summary_dir)
    manifest = load_prep_manifest(workspace_root)
    prep = manifest["prep"]
    tasks_list = []
    seen_indicator = set()
    for config in configs:
        staging = config_staging_dir(workspace_root, config.name)
        staging.mkdir(parents=True, exist_ok=True)
        with open(staging / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        for indicator_type, spec_key in indicator_specs_needed_for_config(config):
            params = _params_for_indicator_type(config, indicator_type)
            for inst in prep.instruments:
                key = (inst, indicator_type, spec_key)
                if key in seen_indicator:
                    continue
                seen_indicator.add(key)
                tasks_list.append(
                    (
                        "indicators",
                        {
                            "root": str(workspace_root),
                            "instrument": inst,
                            "indicator_type": indicator_type,
                            "spec_key": spec_key,
                            "params": params,
                        },
                    )
                )
    for config in configs:
        config_pkl = config_staging_dir(workspace_root, config.name) / "config.pkl"
        for inst in prep.instruments:
            tasks_list.append(
                (
                    "signals",
                    {
                        "root": str(workspace_root),
                        "config_id": config.name,
                        "instrument": inst,
                        "config_path": str(config_pkl),
                    },
                )
            )
        tasks_list.append(
            (
                "simulation",
                {
                    "root": str(workspace_root),
                    "config_id": config.name,
                    "instruments": prep.instruments,
                    "config_path": str(config_pkl),
                },
            )
        )
        rel_path = config_file_map.get(config.name, Path(config.name))
        if getattr(config, "_source_path", None) is not None:
            rel_path = config._source_path
        output_dir = results_root / rel_path
        result_path = config_result_path(workspace_root, config.name)
        tasks_list.append(
            (
                "outputs",
                {
                    "result_path": str(result_path),
                    "output_dir": str(output_dir),
                    "data_root": str(workspace_root),
                },
            )
        )
    result_paths = [str(config_result_path(workspace_root, c.name)) for c in configs]
    tasks_list.append(
        (
            "grid_report",
            {"result_paths": result_paths, "summary_dir": str(summary_dir)},
        )
    )
    return tasks_list


# ============================================================================
# TaskGraph Builders (New API)
# ============================================================================


def build_data_task_graph(
    workspace_root: Path,
    config: Any,
    min_history_days: int = 100,
    instruments: List[str] = None,
) -> TaskGraph:
    """
    Build TaskGraph for single Data task.
    
    Args:
        workspace_root: Workspace root directory
        config: Strategy config
        min_history_days: Minimum history days
        instruments: Instrument list (or use config.instruments)
        
    Returns:
        TaskGraph with single data task
    """
    graph = TaskGraph()
    
    inst_list = instruments if instruments is not None else config.instruments
    
    payload = {
        "root": str(workspace_root),
        "instruments": inst_list,
        "start_date": config.start_date,
        "end_date": config.end_date,
        "lookback_days": config.lookback_days,
        "step_days": config.step_days,
        "min_history_days": min_history_days,
        "column": getattr(config, "column", "Close"),
    }
    
    fingerprint = compute_fingerprint("data", payload, [])
    
    node = TaskNode(
        id="data",
        task_type="data",
        payload=payload,
        depends_on=[],
        fingerprint=fingerprint,
    )
    
    graph.add_task(node)
    return graph


def build_single_eval_task_graph(
    workspace_root: Path,
    output_dir: Path,
    config: Any,
    config_id: str = None,
) -> TaskGraph:
    """
    Build TaskGraph for single evaluation: Indicators → Signals → Simulation → Outputs.
    
    Requires workspace_root to already contain data/prep_manifest.pkl.
    
    Args:
        workspace_root: Workspace root with prepared data
        output_dir: Output directory for results
        config: Strategy config
        config_id: Config identifier (default: config.name)
        
    Returns:
        TaskGraph with all tasks and dependencies
    """
    import pickle
    from .contract import config_staging_dir, config_result_path
    from .indicator_spec import indicator_specs_needed_for_config
    from .tasks import load_prep_manifest, _params_for_indicator_type
    
    workspace_root = Path(workspace_root)
    output_dir = Path(output_dir)
    config_id = config_id or config.name
    
    # Save config
    staging = config_staging_dir(workspace_root, config_id)
    staging.mkdir(parents=True, exist_ok=True)
    config_pkl = staging / "config.pkl"
    with open(config_pkl, "wb") as f:
        pickle.dump(config, f)
    
    # Load manifest
    manifest = load_prep_manifest(workspace_root)
    prep = manifest["prep"]
    
    graph = TaskGraph()
    
    # Indicator tasks
    indicator_task_ids = []
    for indicator_type, spec_key in indicator_specs_needed_for_config(config):
        params = _params_for_indicator_type(config, indicator_type)
        
        for inst in prep.instruments:
            task_id = f"ind_{inst}_{spec_key}"
            payload = {
                "root": str(workspace_root),
                "instrument": inst,
                "indicator_type": indicator_type,
                "spec_key": spec_key,
                "params": params,
            }
            
            fingerprint = compute_fingerprint("indicators", payload, [])
            
            node = TaskNode(
                id=task_id,
                task_type="indicators",
                payload=payload,
                depends_on=[],
                fingerprint=fingerprint,
            )
            
            graph.add_task(node)
            indicator_task_ids.append(task_id)
    
    # Signal tasks (depend on indicators) - split by year for better parallelism
    import pandas as pd
    
    signal_task_ids = []
    merge_task_ids = []
    
    # Calculate year splits
    start_year = prep.start_date.year
    end_year = prep.end_date.year
    years = list(range(start_year, end_year + 1))
    
    for inst in prep.instruments:
        yearly_signal_tasks = []
        
        # Create one signal task per year
        for year in years:
            # Calculate year boundaries
            year_start = max(prep.start_date, pd.Timestamp(f"{year}-01-01"))
            year_end = min(prep.end_date, pd.Timestamp(f"{year}-12-31"))
            
            # Skip if year is out of range
            if year_start > year_end:
                continue
            
            task_id = f"sig_{config_id}_{inst}_{year}"
            payload = {
                "root": str(workspace_root),
                "config_id": config_id,
                "instrument": inst,
                "config_path": str(config_pkl),
                "date_range_start": year_start.strftime("%Y-%m-%d"),
                "date_range_end": year_end.strftime("%Y-%m-%d"),
                "year": year,
            }
            
            # Config fingerprint
            config_fp = compute_config_fingerprint(config_pkl) if config_pkl.exists() else "none"
            
            # Signal tasks depend on their indicators
            relevant_indicators = [tid for tid in indicator_task_ids if inst in tid]
            
            fingerprint = compute_fingerprint("signals", payload, relevant_indicators + [config_fp])
            
            node = TaskNode(
                id=task_id,
                task_type="signals",
                payload=payload,
                depends_on=relevant_indicators,
                fingerprint=fingerprint,
            )
            
            graph.add_task(node)
            yearly_signal_tasks.append(task_id)
        
        # Create merge task for this instrument
        if len(yearly_signal_tasks) > 1:
            merge_task_id = f"merge_{config_id}_{inst}"
            merge_payload = {
                "root": str(workspace_root),
                "config_id": config_id,
                "instrument": inst,
                "yearly_task_ids": yearly_signal_tasks,
            }
            
            merge_fingerprint = compute_fingerprint("merge_signals", merge_payload, yearly_signal_tasks)
            
            merge_node = TaskNode(
                id=merge_task_id,
                task_type="merge_signals",
                payload=merge_payload,
                depends_on=yearly_signal_tasks,
                fingerprint=merge_fingerprint,
            )
            
            graph.add_task(merge_node)
            merge_task_ids.append(merge_task_id)
            signal_task_ids.append(merge_task_id)
        else:
            # Only one year, no merge needed
            signal_task_ids.extend(yearly_signal_tasks)
    
    # Simulation task (depends on all signals)
    sim_task_id = f"sim_{config_id}"
    sim_payload = {
        "root": str(workspace_root),
        "config_id": config_id,
        "instruments": prep.instruments,
        "config_path": str(config_pkl),
    }
    
    config_fp = compute_config_fingerprint(config_pkl) if config_pkl.exists() else "none"
    sim_fingerprint = compute_fingerprint("simulation", sim_payload, signal_task_ids + [config_fp])
    
    sim_node = TaskNode(
        id=sim_task_id,
        task_type="simulation",
        payload=sim_payload,
        depends_on=signal_task_ids,
        fingerprint=sim_fingerprint,
    )
    
    graph.add_task(sim_node)
    
    # Outputs task (depends on simulation)
    result_path = config_result_path(workspace_root, config_id)
    out_task_id = f"out_{config_id}"
    out_payload = {
        "result_path": str(result_path),
        "output_dir": str(output_dir),
        "data_root": str(workspace_root),
    }
    
    out_fingerprint = compute_fingerprint("outputs", out_payload, [sim_fingerprint])
    
    out_node = TaskNode(
        id=out_task_id,
        task_type="outputs",
        payload=out_payload,
        depends_on=[sim_task_id],
        fingerprint=out_fingerprint,
    )
    
    graph.add_task(out_node)
    
    return graph


def build_grid_search_task_graph(
    workspace_root: Path,
    configs: List[Any],
    results_root: Path,
    summary_dir: Path,
    config_file_map: Dict[str, Path],
) -> TaskGraph:
    """
    Build TaskGraph for grid search: Indicators (deduped) → per-config Signals → Simulation → Outputs → GridReport.
    
    Args:
        workspace_root: Workspace root with prepared data
        configs: List of strategy configs
        results_root: Root directory for per-config results
        summary_dir: Directory for grid-level summary
        config_file_map: Mapping of config.name to relative path
        
    Returns:
        TaskGraph with all tasks and dependencies
    """
    import pickle
    from .contract import config_staging_dir, config_result_path
    from .indicator_spec import indicator_specs_needed_for_config
    from .tasks import load_prep_manifest, _params_for_indicator_type
    
    workspace_root = Path(workspace_root)
    results_root = Path(results_root)
    summary_dir = Path(summary_dir)
    
    manifest = load_prep_manifest(workspace_root)
    prep = manifest["prep"]
    
    graph = TaskGraph()
    
    # Save all configs and collect config fingerprints
    config_fingerprints = {}
    for config in configs:
        staging = config_staging_dir(workspace_root, config.name)
        staging.mkdir(parents=True, exist_ok=True)
        config_pkl = staging / "config.pkl"
        with open(config_pkl, "wb") as f:
            pickle.dump(config, f)
        config_fingerprints[config.name] = compute_config_fingerprint(config_pkl)
    
    # Deduplicated indicator tasks
    seen_indicator = set()
    indicator_tasks_by_key = {}
    
    for config in configs:
        for indicator_type, spec_key in indicator_specs_needed_for_config(config):
            params = _params_for_indicator_type(config, indicator_type)
            
            for inst in prep.instruments:
                key = (inst, indicator_type, spec_key)
                if key in seen_indicator:
                    continue
                
                seen_indicator.add(key)
                
                task_id = f"ind_{inst}_{spec_key}"
                payload = {
                    "root": str(workspace_root),
                    "instrument": inst,
                    "indicator_type": indicator_type,
                    "spec_key": spec_key,
                    "params": params,
                }
                
                fingerprint = compute_fingerprint("indicators", payload, [])
                
                node = TaskNode(
                    id=task_id,
                    task_type="indicators",
                    payload=payload,
                    depends_on=[],
                    fingerprint=fingerprint,
                )
                
                graph.add_task(node)
                indicator_tasks_by_key[key] = task_id
    
    # Per-config: Signals → Simulation → Outputs
    simulation_task_ids = []
    
    for config in configs:
        config_pkl = config_staging_dir(workspace_root, config.name) / "config.pkl"
        config_fp = config_fingerprints[config.name]
        
        # Signal tasks
        signal_task_ids = []
        for inst in prep.instruments:
            task_id = f"sig_{config.name}_{inst}"
            payload = {
                "root": str(workspace_root),
                "config_id": config.name,
                "instrument": inst,
                "config_path": str(config_pkl),
            }
            
            # Find relevant indicator dependencies
            relevant_indicators = []
            for indicator_type, spec_key in indicator_specs_needed_for_config(config):
                key = (inst, indicator_type, spec_key)
                if key in indicator_tasks_by_key:
                    relevant_indicators.append(indicator_tasks_by_key[key])
            
            fingerprint = compute_fingerprint("signals", payload, relevant_indicators + [config_fp])
            
            node = TaskNode(
                id=task_id,
                task_type="signals",
                payload=payload,
                depends_on=relevant_indicators,
                fingerprint=fingerprint,
            )
            
            graph.add_task(node)
            signal_task_ids.append(task_id)
        
        # Simulation task
        sim_task_id = f"sim_{config.name}"
        sim_payload = {
            "root": str(workspace_root),
            "config_id": config.name,
            "instruments": prep.instruments,
            "config_path": str(config_pkl),
        }
        
        sim_fingerprint = compute_fingerprint("simulation", sim_payload, signal_task_ids + [config_fp])
        
        sim_node = TaskNode(
            id=sim_task_id,
            task_type="simulation",
            payload=sim_payload,
            depends_on=signal_task_ids,
            fingerprint=sim_fingerprint,
        )
        
        graph.add_task(sim_node)
        simulation_task_ids.append(sim_task_id)
        
        # Outputs task
        rel_path = config_file_map.get(config.name, Path(config.name))
        if getattr(config, "_source_path", None) is not None:
            rel_path = config._source_path
        output_dir = results_root / rel_path
        result_path = config_result_path(workspace_root, config.name)
        
        out_task_id = f"out_{config.name}"
        out_payload = {
            "result_path": str(result_path),
            "output_dir": str(output_dir),
            "data_root": str(workspace_root),
        }
        
        out_fingerprint = compute_fingerprint("outputs", out_payload, [sim_fingerprint])
        
        out_node = TaskNode(
            id=out_task_id,
            task_type="outputs",
            payload=out_payload,
            depends_on=[sim_task_id],
            fingerprint=out_fingerprint,
        )
        
        graph.add_task(out_node)
    
    # Grid report task (depends on all simulations)
    result_paths = [str(config_result_path(workspace_root, c.name)) for c in configs]
    report_payload = {
        "result_paths": result_paths,
        "summary_dir": str(summary_dir),
    }
    
    report_fingerprint = compute_fingerprint("grid_report", report_payload, simulation_task_ids)
    
    report_node = TaskNode(
        id="grid_report",
        task_type="grid_report",
        payload=report_payload,
        depends_on=simulation_task_ids,
        fingerprint=report_fingerprint,
    )
    
    graph.add_task(report_node)
    
    return graph


def run_tasks(
    task_specs: Union[List[Tuple[str, Dict[str, Any]]], TaskGraph],
    verbose: bool = True,
    stream=None,
    max_workers: Optional[int] = None,
    cache_enabled: bool = False,
    checkpoint_path: Optional[Path] = None,
    progress_file: Optional[Path] = None,
) -> Union[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Run tasks in order (legacy mode) or with TaskGraph (new mode).
    
    Args:
        task_specs: Either legacy list of (task_type, payload) tuples or TaskGraph
        verbose: Print progress messages
        stream: Output stream
        max_workers: Max parallel workers (TaskGraph mode only)
        cache_enabled: Enable task caching (TaskGraph mode only)
        checkpoint_path: Path for checkpointing (TaskGraph mode only)
        progress_file: JSON file for progress monitoring (TaskGraph mode only)
        
    Returns:
        Legacy mode: List of result dicts
        TaskGraph mode: Dict mapping task_id to result dict
    """
    if stream is None:
        stream = sys.stdout
    
    # TaskGraph mode: use parallel executor
    if isinstance(task_specs, TaskGraph):
        from .cache import TaskCache
        
        cache = TaskCache() if cache_enabled else None
        executor = Executor(
            max_workers=max_workers,
            verbose=verbose,
            stream=stream,
            progress_file=progress_file,
        )
        
        return executor.execute(task_specs, cache=cache, checkpoint_path=checkpoint_path)
    
    # Legacy mode: sequential execution
    outputs = []
    current_phase = None
    phase_total = 0
    phase_index = 0
    phase_start = None

    for idx, (task_type, payload) in enumerate(task_specs):
        if task_type != current_phase:
            if current_phase is not None and verbose:
                elapsed = time.perf_counter() - phase_start if phase_start else 0
                stream.write(f"  Phase {current_phase}: {phase_total} tasks in {elapsed:.1f}s\n")
                stream.flush()
            current_phase = task_type
            phase_total = sum(1 for t, _ in task_specs[idx:] if t == task_type)
            phase_index = 0
            phase_start = time.perf_counter()
            if verbose:
                stream.write(f"\nPhase: {current_phase} ({phase_total} task{'s' if phase_total != 1 else ''})\n")
                stream.flush()

        phase_index += 1
        desc = _task_description(task_type, payload)
        if verbose:
            stream.write(f"  [{phase_index}/{phase_total}] {desc} ... ")
            stream.flush()
        t0 = time.perf_counter()
        try:
            out = run_task(task_type, payload)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            if verbose:
                stream.write(f"failed ({elapsed:.1f}s): {type(e).__name__}\n")
                stream.flush()
            raise
        elapsed = time.perf_counter() - t0
        outputs.append(out)
        if verbose:
            stream.write(f"done ({elapsed:.1f}s)\n")
            stream.flush()

    if current_phase is not None and verbose and phase_start is not None:
        elapsed = time.perf_counter() - phase_start
        stream.write(f"  Phase {current_phase}: {phase_total} tasks in {elapsed:.1f}s\n")
        stream.flush()
    return outputs
