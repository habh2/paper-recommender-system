import sqlite3
import os
import hashlib
import time
import random
import logging
import torch
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

QDRANT_TIMEOUT = 60
MAX_RETRIES = 5

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "papers.db")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "papers"
MODEL_NAME = "allenai/specter2_base"
BATCH_SIZE = 32
EMBEDDING_DIM = 768


def paper_id_to_int(paper_id: str) -> int:
    return int(hashlib.md5(paper_id.encode()).hexdigest(), 16) % (2 ** 63)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    log.info(f"Model loaded on {device}")
    return tokenizer, model, device


def ensure_schema(conn: sqlite3.Connection):
    try:
        conn.execute("ALTER TABLE papers ADD COLUMN embedded INTEGER NOT NULL DEFAULT 0")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists


def ensure_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )


def fetch_unembedded(conn: sqlite3.Connection, batch_size: int) -> list[dict]:
    rows = conn.execute(
        "SELECT paper_id, title, abstract FROM papers_silver WHERE embedded = 0 LIMIT ?",
        (batch_size,)
    ).fetchall()
    return [{"id": r[0], "title": r[1] or "", "abstract": r[2] or ""} for r in rows]


def embed_batch(texts: list[str], tokenizer, model, device: str) -> list[list[float]]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model(**inputs)
    return output.last_hidden_state[:, 0, :].cpu().tolist()


def mark_embedded(conn: sqlite3.Connection, paper_ids: list[str]):
    conn.executemany(
        "UPDATE papers SET embedded = 1 WHERE paper_id = ?",
        [(pid,) for pid in paper_ids]
    )
    conn.commit()


def run():
    conn = sqlite3.connect(DB_PATH)
    ensure_schema(conn)

    client = QdrantClient(url=QDRANT_URL, timeout=QDRANT_TIMEOUT)
    ensure_collection(client)
    tokenizer, model, device = load_model()

    silver_count = conn.execute("SELECT COUNT(*) FROM papers_silver").fetchone()[0]
    already_embedded = conn.execute("SELECT COUNT(*) FROM papers_silver WHERE embedded = 1").fetchone()[0]
    to_embed = silver_count - already_embedded
    log.info(f"Silver table: {silver_count:,} papers | already embedded: {already_embedded:,} | to embed: {to_embed:,}")

    total = 0
    while True:
        papers = fetch_unembedded(conn, BATCH_SIZE)
        if not papers:
            break

        texts = [f"{p['title']} {p['abstract']}" for p in papers]
        vectors = embed_batch(texts, tokenizer, model, device)

        points = [
            PointStruct(
                id=paper_id_to_int(p["id"]),
                vector=vec,
                payload={"paper_id": p["id"]},
            )
            for p, vec in zip(papers, vectors)
        ]
        for attempt in range(MAX_RETRIES):
            try:
                client.upsert(collection_name=COLLECTION, points=points)
                break
            except ResponseHandlingException:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(random.uniform(0, min(60, 2 ** attempt)))

        mark_embedded(conn, [p["id"] for p in papers])

        total += len(papers)
        log.info(f"embedded={total}")

    log.info(f"Done — {total} papers embedded")
    conn.close()


if __name__ == "__main__":
    run()
