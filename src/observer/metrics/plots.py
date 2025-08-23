import matplotlib.pyplot as plt
import pandas as pd
import sqlite3
import numpy as np
import seaborn as sns
from typing import List, Optional, Dict, Any


def plot_entropy_over_time(db_path="data/sessions.sqlite", session_id=None):
    """Plot input entropy over time for a session."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT step, input FROM interactions"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No data found.")
        return

    from collections import Counter

    def calc_entropy(inputs):
        counts = Counter(inputs)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    entropies = []
    for i in range(1, len(df)+1):
        entropy = calc_entropy(df["input"].iloc[:i])
        entropies.append(entropy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(entropies)+1), entropies, marker='o', linewidth=2, markersize=6)
    plt.title("Input Entropy Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Entropy (bits)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_metrics_dashboard(
    db_path: str = "data/sessions.sqlite",
    session_id: Optional[str] = None,
    metrics: Optional[List[str]] = None
):
    """
    Create a comprehensive dashboard of all evaluation metrics.
    
    Args:
        db_path: Path to database
        session_id: Optional specific session to plot
        metrics: Optional list of specific metrics to include
    """
    # Get evaluation data
    conn = sqlite3.connect(db_path)
    
    if session_id:
        query = "SELECT * FROM evaluation_results WHERE session_id = ? ORDER BY step"
        df = pd.read_sql_query(query, conn, params=(session_id,))
    else:
        query = "SELECT * FROM evaluation_results ORDER BY timestamp"
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if df.empty:
        print("No evaluation data found.")
        return
    
    # Default metrics to plot
    if metrics is None:
        metrics = [
            'character_entropy', 'task_completion', 'instruction_following',
            'factual_accuracy', 'reasoning_quality', 'helpfulness', 'safety'
        ]
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("No valid metrics found in data.")
        return
    
    # Create subplots
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        if session_id:
            # Time series for single session
            ax.plot(df['step'], df[metric], marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Step')
            ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        else:
            # Distribution across all sessions
            ax.hist(df[metric].dropna(), bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score')
            ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Value')
    
    # Hide empty subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_metrics_correlation_matrix(
    db_path: str = "data/sessions.sqlite",
    session_ids: Optional[List[str]] = None
):
    """
    Plot correlation matrix between different metrics.
    
    Args:
        db_path: Path to database
        session_ids: Optional list of specific sessions to include
    """
    conn = sqlite3.connect(db_path)
    
    if session_ids:
        placeholders = ','.join('?' * len(session_ids))
        query = f"SELECT * FROM evaluation_results WHERE session_id IN ({placeholders})"
        df = pd.read_sql_query(query, conn, params=session_ids)
    else:
        query = "SELECT * FROM evaluation_results"
        df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    if df.empty:
        print("No evaluation data found.")
        return
    
    # Select numeric columns (metrics)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    metric_cols = [col for col in numeric_cols if col not in ['id', 'step']]
    
    if len(metric_cols) < 2:
        print("Need at least 2 metrics for correlation analysis.")
        return
    
    # Calculate correlation matrix
    corr_matrix = df[metric_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_performance_radar(
    db_path: str = "data/sessions.sqlite",
    session_id: str,
    comparison_session: Optional[str] = None
):
    """
    Create a radar chart showing performance across different metric categories.
    
    Args:
        db_path: Path to database
        session_id: Session to plot
        comparison_session: Optional second session for comparison
    """
    conn = sqlite3.connect(db_path)
    
    # Get data for main session
    query = "SELECT * FROM evaluation_results WHERE session_id = ?"
    df1 = pd.read_sql_query(query, conn, params=(session_id,))
    
    if df1.empty:
        print(f"No data found for session {session_id}")
        return
    
    # Get data for comparison session if provided
    df2 = None
    if comparison_session:
        df2 = pd.read_sql_query(query, conn, params=(comparison_session,))
        if df2.empty:
            print(f"No data found for comparison session {comparison_session}")
            df2 = None
    
    conn.close()
    
    # Define metric categories
    categories = {
        'Entropy': ['character_entropy'],
        'Capabilities': ['task_completion', 'factual_accuracy', 'reasoning_quality'],
        'Alignment': ['instruction_following', 'helpfulness', 'safety'],
        'Information Theory': ['information_gain', 'empowerment']
    }
    
    # Calculate average scores for each category
    category_scores1 = {}
    category_scores2 = {} if df2 is not None else None
    
    for category, metrics in categories.items():
        available_metrics = [m for m in metrics if m in df1.columns]
        if available_metrics:
            category_scores1[category] = df1[available_metrics].mean().mean()
            if df2 is not None:
                category_scores2[category] = df2[available_metrics].mean().mean()
    
    if not category_scores1:
        print("No valid metrics found for radar chart.")
        return
    
    # Prepare data for radar chart
    categories_list = list(category_scores1.keys())
    values1 = list(category_scores1.values())
    values2 = list(category_scores2.values()) if category_scores2 else None
    
    # Number of variables
    N = len(categories_list)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Close the plot
    values1 += values1[:1]
    if values2:
        values2 += values2[:1]
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot main session
    ax.plot(angles, values1, 'o-', linewidth=2, label=f'Session {session_id}', color='blue')
    ax.fill(angles, values1, alpha=0.25, color='blue')
    
    # Plot comparison session if available
    if values2:
        ax.plot(angles, values2, 'o-', linewidth=2, label=f'Session {comparison_session}', color='red')
        ax.fill(angles, values2, alpha=0.25, color='red')
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_list)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    
    # Add title and legend
    plt.title('Performance Radar Chart', size=16, fontweight='bold', pad=20)
    if values2:
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.show()


def plot_metric_trends(
    db_path: str = "data/sessions.sqlite",
    metric: str = "task_completion",
    window_size: int = 5
):
    """
    Plot trends for a specific metric across all sessions with moving average.
    
    Args:
        db_path: Path to database
        metric: Metric to plot
        window_size: Window size for moving average
    """
    conn = sqlite3.connect(db_path)
    query = f"SELECT timestamp, {metric} FROM evaluation_results WHERE {metric} IS NOT NULL ORDER BY timestamp"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print(f"No data found for metric {metric}")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate moving average
    df['moving_avg'] = df[metric].rolling(window=window_size, center=True).mean()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Plot raw data
    plt.scatter(df['timestamp'], df[metric], alpha=0.6, s=30, label='Raw Data')
    
    # Plot moving average
    plt.plot(df['timestamp'], df['moving_avg'], color='red', linewidth=2, 
             label=f'Moving Average (window={window_size})')
    
    plt.title(f'{metric.replace("_", " ").title()} Trends Over Time', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_safety_analysis(
    db_path: str = "data/sessions.sqlite",
    threshold: float = 0.8
):
    """
    Create specialized plots for safety analysis.
    
    Args:
        db_path: Path to database
        threshold: Safety threshold for flagging
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT session_id, step, safety, timestamp FROM evaluation_results WHERE safety IS NOT NULL"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        print("No safety data found.")
        return
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Safety score distribution
    ax1.hist(df['safety'], bins=20, alpha=0.7, edgecolor='black', color='green')
    ax1.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Safety Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Safety Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Safety violations over time
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['is_violation'] = df['safety'] < threshold
    
    # Group by date and count violations
    daily_violations = df.groupby(df['timestamp'].dt.date)['is_violation'].sum()
    daily_total = df.groupby(df['timestamp'].dt.date).size()
    violation_rate = daily_violations / daily_total
    
    ax2.plot(violation_rate.index, violation_rate.values, marker='o', linewidth=2, color='red')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Violation Rate')
    ax2.set_title('Safety Violation Rate Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Safety by session
    session_safety = df.groupby('session_id')['safety'].mean().sort_values()
    colors = ['red' if x < threshold else 'green' for x in session_safety.values]
    
    ax3.barh(range(len(session_safety)), session_safety.values, color=colors, alpha=0.7)
    ax3.axvline(threshold, color='black', linestyle='--', linewidth=2)
    ax3.set_yticks(range(len(session_safety)))
    ax3.set_yticklabels(session_safety.index, fontsize=8)
    ax3.set_xlabel('Average Safety Score')
    ax3.set_title('Safety Score by Session')
    ax3.grid(True, alpha=0.3)
    
    # 4. Safety score vs other metrics correlation
    conn = sqlite3.connect(db_path)
    query = "SELECT safety, task_completion, helpfulness FROM evaluation_results WHERE safety IS NOT NULL"
    corr_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if 'task_completion' in corr_df.columns and 'helpfulness' in corr_df.columns:
        ax4.scatter(corr_df['task_completion'], corr_df['safety'], alpha=0.6, label='vs Task Completion')
        ax4.scatter(corr_df['helpfulness'], corr_df['safety'], alpha=0.6, label='vs Helpfulness')
        ax4.set_xlabel('Other Metric Score')
        ax4.set_ylabel('Safety Score')
        ax4.set_title('Safety vs Other Metrics')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Safety Correlation Analysis')
    
    plt.tight_layout()
    plt.show()