import json
import sqlite3
import pickle
import numpy as np
from bertopic import BERTopic
from qdrant_client import QdrantClient
from sklearn.linear_model import LogisticRegression

ANN_CANDIDATES = 200
MODEL_PATH_TOPIC = "topic_model"
MODEL_PATH_PREFERENCE = "preference_model.pkl"
COLLECTION = "papers"


def load_models(topic_model_path: str, preference_model_path: str):
    topic_model = BERTopic.load(topic_model_path)
    with open(preference_model_path, "rb") as f:
        preference_model = pickle.load(f)

    topic_labels = {
        row.Topic: row.Name
        for row in topic_model.get_topic_info().itertuples()
        if row.Topic != -1
    }
    return preference_model, topic_labels


def get_candidates(paper_id: str, client: QdrantClient) -> list[str]:
    results = client.query_points(
        collection_name=COLLECTION,
        query=client.retrieve(
            collection_name=COLLECTION,
            ids=[_paper_id_to_int(paper_id)],
            with_vectors=True,
        )[0].vector,
        limit=ANN_CANDIDATES,
        with_payload=True,
    ).points
    return [p.payload["paper_id"] for p in results if p.payload["paper_id"] != paper_id]


def _paper_id_to_int(paper_id: str) -> int:
    import hashlib
    return int(hashlib.md5(paper_id.encode()).hexdigest(), 16) % (2 ** 63)


def score_candidates(
    candidate_ids: list[str],
    conn: sqlite3.Connection,
    preference_model: LogisticRegression,
    topic_labels: dict[int, str],
    k: int,
) -> list[dict]:
    if not candidate_ids:
        return []

    placeholders = ",".join("?" * len(candidate_ids))
    rows = conn.execute(
        f"SELECT paper_id, distribution FROM topic_distributions WHERE paper_id IN ({placeholders})",
        candidate_ids,
    ).fetchall()

    distributions = {r[0]: np.array(json.loads(r[1]), dtype=np.float32) for r in rows}

    scored = []
    weights = preference_model.coef_[0]
    for pid, dist in distributions.items():
        score = float(preference_model.predict_proba([dist])[0][1])
        contributions = weights * dist
        top_topic_ids = np.argsort(contributions)[::-1][:3]
        top_topics = [
            {"label": topic_labels.get(i, f"topic_{i}"), "contribution": float(contributions[i])}
            for i in top_topic_ids
            if contributions[i] > 0
        ]
        scored.append({"paper_id": pid, "score": score, "top_topics": top_topics})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:k]


def enrich_with_metadata(results: list[dict], conn: sqlite3.Connection) -> list[dict]:
    if not results:
        return []

    ids = [r["paper_id"] for r in results]
    placeholders = ",".join("?" * len(ids))
    meta = {
        r[0]: {"title": r[1], "abstract": r[2]}
        for r in conn.execute(
            f"SELECT paper_id, title, abstract FROM papers_silver WHERE paper_id IN ({placeholders})",
            ids,
        ).fetchall()
    }
    return [
        {**r, **meta[r["paper_id"]]}
        for r in results
        if r["paper_id"] in meta
    ]
