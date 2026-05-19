import sqlite3
import os
import pytest
from qdrant_client import QdrantClient

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "papers"
EMBEDDING_DIM = 768


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def qdrant():
    return QdrantClient(url=QDRANT_URL)


def test_all_papers_embedded(db):
    total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    embedded = db.execute("SELECT COUNT(*) FROM papers WHERE embedded = 1").fetchone()[0]
    assert embedded == total, f"{total - embedded} papers not yet embedded"


def test_collection_exists(qdrant):
    assert qdrant.collection_exists(COLLECTION), f"Collection '{COLLECTION}' not found in Qdrant"


def test_vector_count_matches_papers(db, qdrant):
    paper_count = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    vector_count = qdrant.get_collection(COLLECTION).points_count
    assert vector_count == paper_count, (
        f"Qdrant has {vector_count} vectors but DB has {paper_count} papers"
    )


def test_ann_search_returns_results(qdrant):
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=[0.0] * EMBEDDING_DIM,
        limit=10,
    ).points
    assert len(results) == 10, f"Expected 10 results, got {len(results)}"


def test_ann_search_returns_paper_ids(qdrant):
    results = qdrant.query_points(
        collection_name=COLLECTION,
        query=[0.0] * EMBEDDING_DIM,
        limit=1,
    ).points
    assert "paper_id" in results[0].payload, "Result payload missing paper_id"
