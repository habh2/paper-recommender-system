import os
import sqlite3
import pytest
from qdrant_client import QdrantClient
from preference.rerank import load_models, get_candidates, score_candidates, enrich_with_metadata

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.db")
TOPIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "topic_model")
PREFERENCE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "preference_model.pkl")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def client():
    return QdrantClient(url=QDRANT_URL)


@pytest.fixture(scope="module")
def models():
    return load_models(TOPIC_MODEL_PATH, PREFERENCE_MODEL_PATH)


@pytest.fixture(scope="module")
def seed_paper(db):
    row = db.execute("SELECT chosen_paper_id FROM choices ORDER BY timestamp DESC LIMIT 1").fetchone()
    assert row, "No choices recorded — record at least one choice before running reranker tests"
    return row[0]


@pytest.fixture(scope="module")
def results(db, client, models, seed_paper):
    preference_model, topic_labels = models
    candidates = get_candidates(seed_paper, client)
    scored = score_candidates(candidates, db, preference_model, topic_labels, k=10)
    return enrich_with_metadata(scored, db)


def test_candidates_returned(client, seed_paper):
    candidates = get_candidates(seed_paper, client)
    assert len(candidates) > 0, "No ANN candidates returned"
    assert seed_paper not in candidates, "Seed paper should not appear in its own candidates"


def test_results_returned(results):
    assert len(results) > 0, "No results returned after reranking"


def test_results_sorted_by_score(results):
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted by score descending"


def test_results_have_required_fields(results):
    for r in results:
        assert "title" in r and r["title"], f"Missing title in result {r['paper_id']}"
        assert "abstract" in r and r["abstract"], f"Missing abstract in result {r['paper_id']}"
        assert "score" in r, f"Missing score in result {r['paper_id']}"
        assert "top_topics" in r, f"Missing top_topics in result {r['paper_id']}"


def test_results_from_silver(db, results):
    silver_ids = {
        r[0] for r in db.execute("SELECT paper_id FROM papers_silver").fetchall()
    }
    for r in results:
        assert r["paper_id"] in silver_ids, f"{r['paper_id']} is not in papers_silver"


def test_each_result_has_topic_explanation(results):
    for r in results:
        assert len(r["top_topics"]) > 0, f"Result {r['paper_id']} has no topic explanation"
