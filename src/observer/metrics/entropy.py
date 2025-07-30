import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

def calc_entropy(inputs):
    counts = Counter(inputs)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs if p > 0)

def plot_entropy_over_time(session_id, db_path="data/sessions.sqlite"):
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
