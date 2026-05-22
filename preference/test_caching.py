import sqlite3
import tempfile
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

pytestmark = pytest.mark.unit

PAPER_ID = "test-paper-1"
K = 5


@pytest.fixture(scope="module")
def client():
    pref_model = MagicMock()
    pref_model.coef_ = [np.array([0.1, -0.2, 0.3])]
    pref_model.predict_proba = lambda x: [[0.3, 0.7]] * len(x)

    topic_labels = {0: "0_ml", 1: "1_nlp", 2: "2_cv"}

    qdrant = MagicMock()
    qdrant.query_points.return_value.points = []

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE IF NOT EXISTS choices (id INTEGER PRIMARY KEY AUTOINCREMENT, chosen_paper_id TEXT, rejected_paper_id TEXT, timestamp TEXT)")
    conn.commit()
    conn.close()

    with (
        patch("preference.app.preference_model", pref_model),
        patch("preference.app.topic_labels", topic_labels),
        patch("preference.app.qdrant_client", qdrant),
        patch("preference.app.DB_PATH", db_path),
        patch("preference.app._recommend_cache", {}),
        patch("preference.app.get_candidates", return_value=["paper-a", "paper-b"]),
        patch("preference.app.score_candidates", return_value=[{"paper_id": "paper-a", "score": 0.9, "top_topics": []}]),
        patch("preference.app.enrich_with_metadata", return_value=[{"paper_id": "paper-a", "score": 0.9, "top_topics": [], "title": "Test", "abstract": "Abstract"}]),
    ):
        from preference.app import app
        yield TestClient(app, raise_server_exceptions=True)

    os.unlink(db_path)


def test_recommend_returns_results(client):
    response = client.get(f"/recommend?paper_id={PAPER_ID}&k={K}")
    assert response.status_code == 200
    body = response.json()
    assert body["seed_paper_id"] == PAPER_ID
    assert isinstance(body["recommendations"], list)


def test_cache_prevents_recomputation(client):
    score_mock = MagicMock(return_value=[{"paper_id": "paper-a", "score": 0.9, "top_topics": []}])
    enrich_mock = MagicMock(return_value=[{"paper_id": "paper-a", "score": 0.9, "top_topics": [], "title": "T", "abstract": "A"}])
    with (
        patch("preference.app._recommend_cache", {}),
        patch("preference.app.score_candidates", score_mock),
        patch("preference.app.enrich_with_metadata", enrich_mock),
    ):
        client.get("/recommend?paper_id=unique-cache-test&k=5")
        client.get("/recommend?paper_id=unique-cache-test&k=5")
    assert score_mock.call_count == 1, "score_candidates called twice — cache not working"


def test_different_keys_cached_independently(client):
    r1 = client.get(f"/recommend?paper_id={PAPER_ID}&k=5")
    r2 = client.get(f"/recommend?paper_id={PAPER_ID}&k=10")
    assert r1.status_code == 200
    assert r2.status_code == 200


def test_post_choice_persists(client):
    resp = client.post("/choose", json={"chosen_id": "paper-x", "rejected_id": "paper-y"})
    assert resp.status_code == 200
    assert resp.json()["total_choices"] >= 1


def test_health_ready(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


def test_health_no_model(client):
    with patch("preference.app.preference_model", None):
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_ready"] is False
