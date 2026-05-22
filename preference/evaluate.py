"""
Offline evaluation via pairwise accuracy.

Holds out 20% of recorded choices, trains a fresh model on the remaining 80%,
then checks what fraction of held-out pairs the model ranks correctly
(chosen paper scores higher than rejected paper).

A random model scores 50%. A useful model should score 60%+.
"""
import os
import json
import sqlite3
import logging
import numpy as np
from sklearn.linear_model import LogisticRegression

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")


def _load_choices(conn: sqlite3.Connection) -> list[dict]:
    return [
        {"chosen": r[0], "rejected": r[1]}
        for r in conn.execute("SELECT chosen_paper_id, rejected_paper_id FROM choices").fetchall()
    ]


def _load_distributions(conn: sqlite3.Connection) -> dict[str, np.ndarray]:
    return {
        r[0]: np.array(json.loads(r[1]), dtype=np.float32)
        for r in conn.execute("SELECT paper_id, distribution FROM topic_distributions").fetchall()
    }


def _build_difference_vectors(choices: list[dict], distributions: dict[str, np.ndarray]) -> np.ndarray:
    vecs = []
    for c in choices:
        chosen = distributions.get(c["chosen"])
        rejected = distributions.get(c["rejected"])
        if chosen is not None and rejected is not None:
            vecs.append(chosen - rejected)
    return np.array(vecs, dtype=np.float32)


def pairwise_accuracy(
    test_choices: list[dict],
    distributions: dict[str, np.ndarray],
    model: LogisticRegression,
) -> tuple[float, int]:
    correct = 0
    evaluated = 0
    for c in test_choices:
        chosen_dist = distributions.get(c["chosen"])
        rejected_dist = distributions.get(c["rejected"])
        if chosen_dist is None or rejected_dist is None:
            continue
        score_chosen = model.predict_proba([chosen_dist])[0][1]
        score_rejected = model.predict_proba([rejected_dist])[0][1]
        if score_chosen > score_rejected:
            correct += 1
        evaluated += 1
    accuracy = correct / evaluated if evaluated > 0 else 0.0
    return accuracy, evaluated


def run_evaluation(conn: sqlite3.Connection, test_fraction: float = 0.2, seed: int = 42) -> tuple[float, int]:
    choices = _load_choices(conn)
    distributions = _load_distributions(conn)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(choices))
    split = int((1 - test_fraction) * len(choices))
    train_choices = [choices[i] for i in idx[:split]]
    test_choices = [choices[i] for i in idx[split:]]

    X_pos = _build_difference_vectors(train_choices, distributions)
    if len(X_pos) == 0:
        return 0.0, 0

    X = np.vstack([X_pos, -X_pos])
    y = np.array([1] * len(X_pos) + [0] * len(X_pos), dtype=int)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return pairwise_accuracy(test_choices, distributions, model)


if __name__ == "__main__":
    conn = sqlite3.connect(DB_PATH)
    acc, n = run_evaluation(conn)
    conn.close()

    if n == 0:
        log.error("No held-out pairs available — record more choices first")
        raise SystemExit(1)

    log.info(f"Pairwise accuracy: {acc:.1%}  ({n} held-out pairs, 80/20 split, seed=42)")
    if acc < 0.5:
        log.error("Below 50% threshold — model is worse than random, not restarting app")
        raise SystemExit(1)
