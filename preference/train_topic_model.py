import os
import sqlite3
import time
import logging
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from qdrant_client import QdrantClient
from extract_embeddings import extract

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "topic_model")
NR_TOPICS = 80


def fetch_abstracts(conn: sqlite3.Connection, paper_ids: list[str], vectors: np.ndarray):
    id_to_abstract = dict(conn.execute(
        "SELECT paper_id, abstract FROM papers_silver"
    ).fetchall())
    filtered_ids, filtered_abstracts, filtered_vectors = [], [], []
    for pid, vec in zip(paper_ids, vectors):
        abstract = id_to_abstract.get(pid)
        if abstract:
            filtered_ids.append(pid)
            filtered_abstracts.append(abstract)
            filtered_vectors.append(vec)
    return filtered_ids, filtered_abstracts, np.array(filtered_vectors, dtype=np.float32)


def train(abstracts: list[str], vectors: np.ndarray) -> BERTopic:
    vectorizer = CountVectorizer(stop_words="english")
    model = BERTopic(nr_topics=NR_TOPICS, verbose=True, vectorizer_model=vectorizer)
    topics, _ = model.fit_transform(abstracts, embeddings=vectors)
    new_topics = model.reduce_outliers(abstracts, topics, strategy="embeddings", embeddings=vectors)
    model.update_topics(abstracts, topics=new_topics, vectorizer_model=vectorizer)
    return model


if __name__ == "__main__":
    t0 = time.time()

    log.info("Extracting embeddings from Qdrant...")
    client = QdrantClient(url=QDRANT_URL)
    paper_ids, vectors = extract(client)
    log.info(f"Extracted {len(paper_ids)} papers ({time.time() - t0:.1f}s)")

    log.info("Fetching abstracts from SQLite (papers_silver)...")
    conn = sqlite3.connect(DB_PATH)
    paper_ids, abstracts, vectors = fetch_abstracts(conn, paper_ids, vectors)
    conn.close()
    log.info(f"Filtered to {len(abstracts)} papers with abstracts ({time.time() - t0:.1f}s)")

    log.info(f"Training BERTopic (target {NR_TOPICS} topics) on {len(abstracts)} documents...")
    model = train(abstracts, vectors)
    log.info(f"Training done ({time.time() - t0:.1f}s)")

    topics = model.get_topic_info()
    log.info(f"Topics found: {len(topics) - 1}")
    print(topics[["Topic", "Count", "Name"]].to_string(index=False))

    model.save(MODEL_PATH, serialization="pickle", save_ctfidf=True)
    log.info(f"Model saved to {MODEL_PATH}")
