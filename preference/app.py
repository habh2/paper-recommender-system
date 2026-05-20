import os
import sqlite3
import logging
from datetime import datetime, timezone
from contextlib import contextmanager, asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from .rerank import load_models, get_candidates, score_candidates, enrich_with_metadata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
TOPIC_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "topic_model")
PREFERENCE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preference_model.pkl")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")

preference_model = None
topic_labels = None
qdrant_client = None


@asynccontextmanager
async def lifespan(_):
    global preference_model, topic_labels, qdrant_client
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
    log.info("Loading models...")
    preference_model, topic_labels = load_models(TOPIC_MODEL_PATH, PREFERENCE_MODEL_PATH)
    qdrant_client = QdrantClient(url=QDRANT_URL)
    log.info("Models loaded")
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


@app.get("/profile")
def get_profile(n: int = Query(default=5, ge=1, le=20)):
    weights = preference_model.coef_[0]
    ranked = sorted(enumerate(weights), key=lambda x: x[1], reverse=True)

    def fmt(idx, weight):
        label = topic_labels.get(idx, f"topic_{idx}")
        clean = label.replace(f"{idx}_", "").replace("_", " ")
        return {"label": clean, "weight": round(float(weight), 3)}

    return {
        "liked": [fmt(i, w) for i, w in ranked[:n]],
        "avoided": [fmt(i, w) for i, w in ranked[-n:][::-1]],
    }


@app.get("/recommend")
def get_recommendations(paper_id: str = Query(...), k: int = Query(default=10, ge=1, le=50)):
    candidates = get_candidates(paper_id, qdrant_client)
    if not candidates:
        raise HTTPException(status_code=404, detail="No candidates found for this paper")

    with get_db() as conn:
        results = score_candidates(candidates, conn, preference_model, topic_labels, k=k)
        results = enrich_with_metadata(results, conn)

    return {"seed_paper_id": paper_id, "recommendations": results}


@app.get("/last-chosen")
def get_last_chosen():
    with get_db() as conn:
        row = conn.execute(
            "SELECT chosen_paper_id, p.title, p.abstract FROM choices c JOIN papers p ON p.paper_id = c.chosen_paper_id ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No choices recorded yet")
    return {"paper_id": row["chosen_paper_id"], "title": row["title"], "abstract": row["abstract"]}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
