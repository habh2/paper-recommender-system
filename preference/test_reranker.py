import os
import json
import sqlite3
import numpy as np
import pytest
from unittest.mock import MagicMock
from preference.rerank import score_candidates, enrich_with_metadata

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.db")
TOPIC_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "topic_model")
PREFERENCE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "preference_model.pkl")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


# ---------------------------------------------------------------------------
# Unit tests — no external dependencies
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE topic_distributions (paper_id TEXT, distribution TEXT)")
    conn.executemany(
        "INSERT INTO topic_distributions VALUES (?, ?)",
        [
            ("paper-high", json.dumps([0.9, 0.1])),
            ("paper-low",  json.dumps([0.1, 0.9])),
            ("paper-mid",  json.dumps([0.5, 0.5])),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def _make_model():
    model = MagicMock()
    model.coef_ = [np.array([1.0, -1.0])]
    # higher score when dist[0] dominates (aligned with positive weight)
    def predict_proba(X):
        return [[1 - x[0][0], x[0][0]] for x in [X]]
    model.predict_proba = predict_proba
    return model


@pytest.mark.unit
def test_score_candidates_empty_input(in_memory_db):
    result = score_candidates([], in_memory_db, _make_model(), {}, k=10)
    assert result == []


@pytest.mark.unit
def test_score_candidates_ranks_by_preference(in_memory_db):
    model = _make_model()
    topic_labels = {0: "0_ml", 1: "1_cv"}
    result = score_candidates(["paper-high", "paper-low"], in_memory_db, model, topic_labels, k=10)
    assert len(result) == 2
    assert result[0]["paper_id"] == "paper-high", "Higher-aligned paper should rank first"
    assert result[0]["score"] > result[1]["score"]


@pytest.mark.unit
def test_score_candidates_respects_k(in_memory_db):
    result = score_candidates(["paper-high", "paper-low", "paper-mid"], in_memory_db, _make_model(), {}, k=2)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests — require live DB, Qdrant, and recorded choices
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def client():
    from qdrant_client import QdrantClient
    return QdrantClient(url=QDRANT_URL)


@pytest.fixture(scope="module")
def models():
    from preference.rerank import load_models
    return load_models(TOPIC_MODEL_PATH, PREFERENCE_MODEL_PATH)


@pytest.fixture(scope="module")
def seed_paper(db):
    row = db.execute("SELECT chosen_paper_id FROM choices ORDER BY timestamp DESC LIMIT 1").fetchone()
    assert row, "No choices recorded — record at least one choice before running reranker tests"
    return row[0]


@pytest.fixture(scope="module")
def results(db, client, models, seed_paper):
    from preference.rerank import get_candidates
    preference_model, topic_labels = models
    candidates = get_candidates(seed_paper, client)
    scored = score_candidates(candidates, db, preference_model, topic_labels, k=10)
    return enrich_with_metadata(scored, db)


@pytest.mark.integration
def test_candidates_returned(client, seed_paper):
    from preference.rerank import get_candidates
    candidates = get_candidates(seed_paper, client)
    assert len(candidates) > 0, "No ANN candidates returned"
    assert seed_paper not in candidates, "Seed paper should not appear in its own candidates"


@pytest.mark.integration
def test_results_returned(results):
    assert len(results) > 0, "No results returned after reranking"


@pytest.mark.integration
def test_results_sorted_by_score(results):
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results are not sorted by score descending"


@pytest.mark.integration
def test_results_have_required_fields(results):
    for r in results:
        assert "title" in r and r["title"], f"Missing title in result {r['paper_id']}"
        assert "abstract" in r and r["abstract"], f"Missing abstract in result {r['paper_id']}"
        assert "score" in r, f"Missing score in result {r['paper_id']}"
        assert "top_topics" in r, f"Missing top_topics in result {r['paper_id']}"


@pytest.mark.integration
def test_results_from_silver(db, results):
    silver_ids = {
        r[0] for r in db.execute("SELECT paper_id FROM papers_silver").fetchall()
    }
    for r in results:
        assert r["paper_id"] in silver_ids, f"{r['paper_id']} is not in papers_silver"


@pytest.mark.integration
def test_each_result_has_topic_explanation(results):
    for r in results:
        assert len(r["top_topics"]) > 0, f"Result {r['paper_id']} has no topic explanation"
