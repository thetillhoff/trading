"""
DAG visualization for task graphs.

Generates graphical representation of task dependencies, parallelization
opportunities, and cache hit status.
"""
from pathlib import Path
from typing import Dict, List, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection

from .task_graph import TaskGraph


def visualize_task_dag(
    graph: TaskGraph,
    output_path: Path,
    title: str = "Task Execution DAG",
) -> None:
    """
    Visualize task DAG showing dependencies, parallelization, and cache hits.
    
    Excludes outputs tasks since they generate the DAG and can't show their own timing.
    
    Args:
        graph: TaskGraph to visualize
        output_path: Path to save the visualization
        title: Chart title
    """
    # Filter out outputs tasks (they generate the DAG, can't show their own timing)
    filtered_nodes = {
        task_id: node 
        for task_id, node in graph.nodes.items() 
        if node.task_type != "outputs"
    }
    
    if not filtered_nodes:
        return
    
    # Get levels and filter to only include non-outputs tasks
    all_levels = graph.get_topological_levels()
    levels = []
    for level in all_levels:
        filtered_level = [tid for tid in level if tid in filtered_nodes]
        if filtered_level:
            levels.append(filtered_level)
    
    if not levels:
        return
    
    # Calculate layout (left to right)
    max_height = max(len(level) for level in levels)
    width = len(levels)
    
    # Create figure with appropriate size (swapped for horizontal layout)
    fig_width = max(12, min(24, width * 2.5))
    fig_height = max(8, min(16, max_height * 1.2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Position nodes (left to right)
    node_positions: Dict[str, tuple] = {}
    x_spacing = 1.0
    
    for level_idx, level in enumerate(levels):
        x = level_idx * x_spacing  # Left to right
        num_nodes = len(level)
        
        # Center nodes vertically
        if num_nodes == 1:
            y_positions = [max_height / 2]
        else:
            # Spread evenly, leaving margins
            margin = 0.5
            total_height = max_height - 2 * margin
            y_positions = [margin + i * (total_height / (num_nodes - 1)) for i in range(num_nodes)]
        
        for i, task_id in enumerate(level):
            node_positions[task_id] = (x, y_positions[i])
    
    # Draw edges (dependencies) - horizontal arrows
    for task_id, (x, y) in node_positions.items():
        node = graph.get_task(task_id)
        
        for dep_id in node.depends_on:
            if dep_id in node_positions:
                dep_x, dep_y = node_positions[dep_id]
                
                # Draw arrow from dependency to task (left to right)
                arrow = FancyArrowPatch(
                    (dep_x + 0.35, dep_y),
                    (x - 0.35, y),
                    arrowstyle='->,head_width=0.15,head_length=0.2',
                    color='gray',
                    alpha=0.5,
                    linewidth=1.5,
                    zorder=1,
                )
                ax.add_patch(arrow)
    
    # Draw nodes
    for task_id, (x, y) in node_positions.items():
        node = graph.get_task(task_id)
        
        # Determine color based on task type
        if node.task_type == 'data':
            color = '#ADD8E6'  # Light blue
        elif node.task_type == 'indicators':
            color = '#FFE4B5'  # Moccasin (light orange)
        elif node.task_type == 'signals':
            color = '#DDA0DD'  # Plum (light purple)
        elif node.task_type == 'merge_signals':
            color = '#F0E68C'  # Khaki (light yellow)
        elif node.task_type == 'simulation':
            color = '#FFA07A'  # Light salmon
        elif node.task_type == 'outputs':
            color = '#98FB98'  # Pale green
        else:
            color = '#D3D3D3'  # Light gray
        
        # Determine border: cached tasks get a thick green border, others get no visible border
        if node.was_cached:
            edge_color = '#228B22'  # Forest green
            edge_width = 3
        else:
            edge_color = color  # Same as fill (invisible border)
            edge_width = 1
        
        # Draw box for task
        box = FancyBboxPatch(
            (x - 0.35, y - 0.15),
            0.7,
            0.30,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=2,
        )
        ax.add_patch(box)
        
        # Shorten task ID for display
        display_id = _shorten_task_id(task_id)
        
        # Add text label (task ID)
        ax.text(
            x, y + 0.04, display_id,
            ha='center', va='center',
            fontsize=7,
            fontweight='bold',
            zorder=3,
        )
        
        # Add computation time below task ID
        if node.compute_time_s > 0:
            time_str = f"{node.compute_time_s:.1f}s"
        else:
            time_str = "0.0s"
        
        ax.text(
            x, y - 0.07, time_str,
            ha='center', va='center',
            fontsize=6,
            color='#555555',
            style='italic',
            zorder=3,
        )
    
    # Add level labels on the top with computation times
    for level_idx in range(len(levels)):
        x = level_idx * x_spacing
        
        # Calculate total computation time for this level
        level_compute_time = sum(
            graph.get_task(tid).compute_time_s 
            for tid in levels[level_idx]
        )
        
        ax.text(
            x, max_height + 0.5,
            f"Level {level_idx}",
            ha='center', va='bottom',
            fontsize=10,
            fontweight='bold',
            color='#555555',
        )
        
        # Add task count and total computation time
        num_tasks = len(levels[level_idx])
        time_str = f"{level_compute_time:.2f}s total"
        
        ax.text(
            x, max_height + 0.8,
            f"({num_tasks} task{'s' if num_tasks != 1 else ''}, {time_str})",
            ha='center', va='bottom',
            fontsize=8,
            color='#888888',
        )
    
    # Set axis limits and remove axes
    ax.set_xlim(-0.5, width * x_spacing + 0.5)
    ax.set_ylim(-0.5, max_height + 1.5)
    ax.axis('off')
    
    # Add title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        # Task types (fill colors only)
        Patch(facecolor='#ADD8E6', edgecolor='#ADD8E6', label='Data'),
        Patch(facecolor='#FFE4B5', edgecolor='#FFE4B5', label='Indicators'),
        Patch(facecolor='#DDA0DD', edgecolor='#DDA0DD', label='Signals'),
        Patch(facecolor='#F0E68C', edgecolor='#F0E68C', label='Merge Signals'),
        Patch(facecolor='#FFA07A', edgecolor='#FFA07A', label='Simulation'),
        # Separator
        Line2D([0], [0], color='none', label=''),
        # Cache indicator (green border)
        Patch(facecolor='#DDDDDD', edgecolor='#228B22', linewidth=3, label='Cached (green border)'),
    ]
    
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        fontsize=9,
        framealpha=0.9,
        title='Task Types',
        title_fontsize=10,
    )
    
    # Add explanation text
    explanation = (
        "Nodes in the same level can run in parallel (theoretical maximum).\n"
        "Arrows show dependencies (task waits for all arrows pointing to it).\n"
        "Green border indicates task was loaded from cache (no computation).\n"
        "Times shown: individual task computation + level total (sum of all tasks in level)."
    )
    fig.text(
        0.5, 0.02,
        explanation,
        ha='center',
        fontsize=8,
        style='italic',
        color='#555555',
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def _shorten_task_id(task_id: str) -> str:
    """
    Shorten task ID for display in graph nodes.
    
    Examples:
        'ind_sp500_technical_d0dc9429' -> 'ind_sp500_tech'
        'sig_baseline_sp500_2010' -> 'sig_sp500_2010'
        'merge_baseline_sp500' -> 'merge_sp500'
    """
    parts = task_id.split('_')
    
    if len(parts) <= 3:
        return task_id
    
    # Shorten indicator type
    if 'technical' in parts:
        parts = [p.replace('technical', 'tech') for p in parts]
    if 'elliott' in parts:
        parts = [p.replace('elliott', 'ew') for p in parts]
    
    # Remove hash suffixes (long hex strings)
    filtered = []
    for part in parts:
        if len(part) > 12 and all(c in '0123456789abcdef' for c in part):
            continue  # Skip hash
        filtered.append(part)
    
    # Keep first 4 parts max
    return '_'.join(filtered[:4])
