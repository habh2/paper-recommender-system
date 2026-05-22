import sqlite3
import os
import pytest

pytestmark = pytest.mark.integration

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "papers.db")


@pytest.fixture(scope="module")
def db():
    conn = sqlite3.connect(DB_PATH)
    yield conn
    conn.close()


def test_minimum_paper_count(db):
    count = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    assert count >= 30_000, f"Expected >= 30,000 papers, got {count:,}"


def test_lookup_by_id(db):
    sample_id = db.execute("SELECT paper_id FROM papers LIMIT 1").fetchone()
    assert sample_id is not None, "No papers in DB"
    row = db.execute(
        "SELECT paper_id, title FROM papers WHERE paper_id = ?", (sample_id[0],)
    ).fetchone()
    assert row is not None, f"Could not look up paper_id {sample_id[0]}"

def test_filter_by_year(db):
    count = db.execute(
        "SELECT COUNT(*) FROM papers WHERE year >= 2020"
    ).fetchone()[0]
    assert count > 0, "No papers found with year >= 2020"


def test_no_duplicate_paper_ids(db):
    total = db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    unique = db.execute("SELECT COUNT(DISTINCT paper_id) FROM papers").fetchone()[0]
    assert total == unique, f"{total - unique} duplicate paper_ids found"
