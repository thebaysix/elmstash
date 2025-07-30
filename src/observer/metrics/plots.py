import matplotlib.pyplot as plt
import pandas as pd
import sqlite3


def plot_entropy_over_time(db_path="data/sessions.sqlite", session_id=None):
    conn = sqlite3.connect(db_path)
    query = f"SELECT step, input FROM interactions"
    if session_id:
        query += f" WHERE session_id = '{session_id}'"
    df = pd.read_sql_query(query, conn)

    if df.empty:
        print("No data found.")
        return

    from collections import Counter
    import numpy as np

    def calc_entropy(inputs):
        counts = Counter(inputs)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    entropies = []
    for i in range(1, len(df)+1):
        entropy = calc_entropy(df["input"].iloc[:i])
        entropies.append(entropy)

    plt.plot(range(1, len(entropies)+1), entropies, marker='o')
    plt.title("Input Entropy Over Time")
    plt.xlabel("Step")
    plt.ylabel("Entropy (bits)")
    plt.grid(True)
    plt.show()