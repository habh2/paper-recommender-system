import os
import sqlite3
import json
import logging
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "preference_model.pkl")
MIN_CHOICES = 10


def load_choices(conn: sqlite3.Connection) -> list[dict]:
    return [
        {"chosen": r[0], "rejected": r[1]}
        for r in conn.execute("SELECT chosen_paper_id, rejected_paper_id FROM choices").fetchall()
    ]


def load_distributions(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    return {
        r[0]: np.array(json.loads(r[1]), dtype=np.float32)
        for r in conn.execute("SELECT paper_id, distribution FROM topic_distributions").fetchall()
    }


def build_difference_vectors(choices: list[dict], distributions: dict[str, np.ndarray]):
    X = []
    skipped = 0
    for c in choices:
        chosen = distributions.get(c["chosen"])
        rejected = distributions.get(c["rejected"])
        if chosen is None or rejected is None:
            skipped += 1
            continue
        X.append(chosen - rejected)
    if skipped:
        log.warning(f"Skipped {skipped} choices with missing distributions")
    return np.array(X, dtype=np.float32)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)

    choices = load_choices(conn)
    if len(choices) < MIN_CHOICES:
        log.error(f"Not enough choices — need at least {MIN_CHOICES}, have {len(choices)}")
        conn.close()
        raise SystemExit(1)

    log.info(f"Loaded {len(choices)} choices")

    distributions = load_distributions(conn)
    conn.close()
    log.info(f"Loaded {len(distributions)} topic distributions")

    X_pos = build_difference_vectors(choices, distributions)
    X = np.vstack([X_pos, -X_pos])
    y = np.array([1] * len(X_pos) + [0] * len(X_pos), dtype=int)
    log.info(f"Training on {len(X)} difference vectors of dimension {X.shape[1]}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved to {MODEL_PATH}")

    weights = model.coef_[0]
    top = np.argsort(weights)[::-1]
    log.info("Top 5 preferred topics: " + ", ".join(f"topic_{i} ({weights[i]:+.3f})" for i in top[:5]))
    log.info("Top 5 avoided topics:   " + ", ".join(f"topic_{i} ({weights[i]:+.3f})" for i in top[-5:]))
