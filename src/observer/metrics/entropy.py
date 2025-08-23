import sqlite3
from collections import Counter
from typing import List, Union

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # Minimal numpy replacement for basic operations
    class np:
        @staticmethod
        def log2(x):
            import math
            return math.log2(x)

def calc_entropy(inputs: Union[str, List[str]]) -> float:
    """
    Calculate Shannon entropy for a collection of inputs.
    
    Args:
        inputs: String or list of strings to calculate entropy for
        
    Returns:
        Entropy in bits
    """
    if isinstance(inputs, str):
        inputs = list(inputs)
    
    counts = Counter(inputs)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def calc_character_entropy(response: str) -> float:
    """
    Calculate character-level entropy within a single response.
    Measures linguistic diversity and complexity.
    
    Args:
        response: Single model response text
        
    Returns:
        Character entropy in bits
    """
    return calc_entropy(list(response))

def calc_response_entropy(responses: List[str]) -> float:
    """
    Calculate entropy across multiple responses to measure consistency vs diversity.
    
    Args:
        responses: List of model responses to the same or similar prompts
        
    Returns:
        Response entropy in bits
    """
    return calc_entropy(responses)

def calc_input_entropy(inputs: List[str]) -> float:
    """
    Calculate entropy of input distribution to measure test coverage.
    
    Args:
        inputs: List of input prompts used in evaluation
        
    Returns:
        Input entropy in bits
    """
    return calc_entropy(inputs)

def plot_entropy_over_time(session_id, db_path="data/sessions.sqlite"):
    """Plot entropy over time for a session (requires matplotlib and pandas)."""
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Please install matplotlib and pandas.")
        return
        
    conn = sqlite3.connect(db_path)
    query = f"SELECT step, input FROM interactions WHERE session_id = ? ORDER BY step"
    df = pd.read_sql_query(query, conn, params=(session_id,))
    
    if df.empty:
        print("No data for session.")
        return

    entropies = []
    for i in range(1, len(df)+1):
        entropy = calc_entropy(df['input'].iloc[:i])
        entropies.append(entropy)

    plt.plot(range(1, len(entropies)+1), entropies, marker='o')
    plt.title("Input Entropy Over Time")
    plt.xlabel("Step")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.show()
