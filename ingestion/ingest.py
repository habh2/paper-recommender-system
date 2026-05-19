import requests
import sqlite3
import json
import time
import random
import os
import logging

BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = "paperId,title,abstract,year,citationCount,fieldsOfStudy,authors"
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
TARGET_PAPER_COUNT = 40000

logging.basicConfig(filename='app.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def create_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            paper_id        TEXT PRIMARY KEY,
            title           TEXT,
            abstract        TEXT,
            year            INTEGER,
            citation_count  INTEGER,
            fields_of_study TEXT,
            authors         TEXT,
            query           TEXT
        )
    """)
    conn.commit()
    return conn


def _backoff_sleep(attempt: int, base: float = 1.0, cap: float = 60.0) -> None:
    # jitter to avoid thundering herd when multiple retries fire simultaneously
    ceiling = min(cap, base * (2 ** attempt))
    time.sleep(random.uniform(0, ceiling))


def fetch_papers(query: str, token: str | None = None,
                 max_retries: int = 5) -> tuple[list, str | None]:
    params = {"query": query, "fields": FIELDS, "limit": 1000, "sort": "publicationDate:desc"}
    if token:
        params["token"] = token

    for attempt in range(max_retries):
        resp = requests.get(
            f"{BASE_URL}/paper/search/bulk",
            params=params,
            timeout=30,
        )

        if resp.status_code == 429:
            retry_after = min(60, 2 ** attempt) + random.uniform(0, 1)
            time.sleep(retry_after)
            continue

        if resp.status_code >= 500:
            if attempt == max_retries - 1:
                resp.raise_for_status()
            _backoff_sleep(attempt)
            continue

        resp.raise_for_status()
        body = resp.json()
        return body.get("data", []), body.get("token")

    return [], None


def upsert_papers(conn: sqlite3.Connection, papers: list, query: str) -> int:
    rows = [
        {
            "paper_id":        p["paperId"],
            "title":           p.get("title"),
            "abstract":        p.get("abstract"),
            "year":            p.get("year"),
            "citation_count":  p.get("citationCount"),
            "fields_of_study": json.dumps(p.get("fieldsOfStudy") or []),
            "authors":         json.dumps([a["name"] for a in (p.get("authors") or [])]),
            "query":           query,
        }
        for p in papers
    ]
    cursor = conn.executemany(
        """INSERT OR IGNORE INTO papers
           (paper_id, title, abstract, year, citation_count, fields_of_study, authors, query)
           VALUES (:paper_id, :title, :abstract, :year, :citation_count,
                   :fields_of_study, :authors, :query)""",
        rows,
    )
    conn.commit()
    return cursor.rowcount


def ingest(query: str) -> None:
    conn = create_db(DB_PATH)

    existing = conn.execute(
        "SELECT COUNT(*) FROM papers WHERE query = ?", (query,)
    ).fetchone()[0]

    if existing >= TARGET_PAPER_COUNT:
        print(f"Skipping — {existing:,} papers already in DB for [{query}]")
        conn.close()
        return

    token = None
    inserted = 0
    fetched = 0

    print(f"Ingesting [{query}] (have {existing:,}, want {TARGET_PAPER_COUNT:,})...")

    while fetched < TARGET_PAPER_COUNT:
        try:
            papers, token = fetch_papers(query, token=token)
        except requests.HTTPError as e:
            print(f"  Unrecoverable error: {e}")
            raise

        if not papers:
            logging.info("No papers were retrieved")
            break

        new_rows = upsert_papers(conn, papers, query)
        fetched += len(papers)
        inserted += new_rows

        print(f"  fetched={fetched}  new={inserted}", end="\r")
        time.sleep(0.1 + random.uniform(0, 0.05))

        if token is None:
            break

    print(f"\nDone — {fetched} fetched, {inserted} new rows")

    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    print(f"\nTotal papers in DB: {total}")
    conn.close()


QUERY = "sociology"

if __name__ == "__main__":
    ingest(QUERY)
