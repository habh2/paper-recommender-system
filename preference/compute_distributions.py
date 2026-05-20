import os
import sqlite3
import json
import logging
import time
import numpy as np
from bertopic import BERTopic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "topic_model")
BATCH_SIZE = 1000


def ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topic_distributions (
            paper_id     TEXT PRIMARY KEY,
            distribution TEXT NOT NULL
        )
    """)
    conn.commit()


def fetch_papers(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    rows = conn.execute("SELECT paper_id, abstract FROM papers_silver").fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def store_distributions(conn: sqlite3.Connection, paper_ids: list[str], distributions: np.ndarray):
    rows = [
        (pid, json.dumps(dist.tolist()))
        for pid, dist in zip(paper_ids, distributions)
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO topic_distributions (paper_id, distribution) VALUES (?, ?)",
        rows,
    )
    conn.commit()


if __name__ == "__main__":
    t0 = time.time()

    log.info(f"Loading BERTopic model from {MODEL_PATH}...")
    model = BERTopic.load(MODEL_PATH)

    log.info("Fetching abstracts from papers_silver...")
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)
    paper_ids, abstracts = fetch_papers(conn)
    log.info(f"Fetched {len(abstracts)} abstracts ({time.time() - t0:.1f}s)")

    log.info("Computing topic distributions...")
    distributions, _ = model.approximate_distribution(abstracts, batch_size=BATCH_SIZE)
    log.info(f"Distributions computed ({time.time() - t0:.1f}s)")

    log.info("Storing distributions in SQLite...")
    store_distributions(conn, paper_ids, distributions)
    conn.close()

    log.info(f"Done — {len(paper_ids)} distributions stored ({time.time() - t0:.1f}s)")
