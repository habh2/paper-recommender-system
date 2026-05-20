import sqlite3
import json
import os
import pickle
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "preference_model.pkl")
MIN_CHOICES = 10


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def test_choices_persisted(db):
    count = db.execute("SELECT COUNT(*) FROM choices").fetchone()[0]
    assert count >= MIN_CHOICES, f"Expected at least {MIN_CHOICES} choices, got {count}"


def test_choices_have_valid_paper_ids(db):
    invalid = db.execute("""
        SELECT COUNT(*) FROM choices c
        WHERE NOT EXISTS (SELECT 1 FROM papers WHERE paper_id = c.chosen_paper_id)
           OR NOT EXISTS (SELECT 1 FROM papers WHERE paper_id = c.rejected_paper_id)
    """).fetchone()[0]
    assert invalid == 0, f"{invalid} choices reference paper IDs not in the papers table"


def test_all_silver_papers_have_distributions(db):
    silver = db.execute("SELECT COUNT(*) FROM papers_silver").fetchone()[0]
    distributed = db.execute("SELECT COUNT(*) FROM topic_distributions").fetchone()[0]
    assert distributed == silver, f"{silver - distributed} papers_silver rows missing topic distributions"


def test_distributions_have_correct_dimension(db):
    row = db.execute("SELECT distribution FROM topic_distributions LIMIT 1").fetchone()
    dist = json.loads(row[0])
    assert len(dist) > 0, "Distribution is empty"
    assert abs(sum(dist) - 1.0) < 1e-3, f"Distribution does not sum to 1 (got {sum(dist):.4f})"


def test_preference_model_is_logistic_regression(model):
    assert isinstance(model, LogisticRegression)


def test_preference_model_weights_match_topics(db, model):
    n_topics = db.execute(
        "SELECT LENGTH(distribution) - LENGTH(REPLACE(distribution, ',', '')) + 1 FROM topic_distributions LIMIT 1"
    ).fetchone()[0]
    assert model.coef_.shape[1] == n_topics, (
        f"Model has {model.coef_.shape[1]} weights but topic distributions have {n_topics} dimensions"
    )


def test_preference_score_varies_across_papers(db, model):
    rows = db.execute("SELECT distribution FROM topic_distributions ORDER BY RANDOM() LIMIT 50").fetchall()
    distributions = np.array([json.loads(r[0]) for r in rows], dtype=np.float32)
    scores = model.predict_proba(distributions)[:, 1]
    assert scores.max() - scores.min() > 0.01, "Preference scores are identical across papers — model has no signal"
