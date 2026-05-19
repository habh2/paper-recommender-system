import os
import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@asynccontextmanager
async def lifespan(_):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS choices (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                chosen_paper_id   TEXT NOT NULL,
                rejected_paper_id TEXT NOT NULL,
                timestamp         TEXT NOT NULL
            )
        """)
        conn.commit()
    yield


app = FastAPI(lifespan=lifespan)


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


class ChoiceRequest(BaseModel):
    chosen_id: str
    rejected_id: str


@app.get("/pair")
def get_pair():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT p.paper_id, p.title, p.abstract
            FROM papers_silver p
            JOIN topic_distributions td ON td.paper_id = p.paper_id
            ORDER BY RANDOM()
            LIMIT 2
        """).fetchall()

    if len(rows) < 2:
        raise HTTPException(status_code=500, detail="Not enough papers available")

    return {
        "paper_a": {"id": rows[0]["paper_id"], "title": rows[0]["title"], "abstract": rows[0]["abstract"]},
        "paper_b": {"id": rows[1]["paper_id"], "title": rows[1]["title"], "abstract": rows[1]["abstract"]},
    }


@app.post("/choose")
def post_choice(body: ChoiceRequest):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO choices (chosen_paper_id, rejected_paper_id, timestamp) VALUES (?, ?, ?)",
            (body.chosen_id, body.rejected_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        total = conn.execute("SELECT COUNT(*) FROM choices").fetchone()[0]

    log.info(f"Choice recorded — {total} total")
    return {"total_choices": total}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
