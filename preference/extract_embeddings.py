import os
import logging
import numpy as np
from qdrant_client import QdrantClient

log = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "papers"
SCROLL_BATCH = 1000


def extract(client: QdrantClient) -> tuple[list[str], np.ndarray]:
    paper_ids = []
    vectors = []

    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION,
            limit=SCROLL_BATCH,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        for point in results:
            paper_ids.append(point.payload["paper_id"])
            vectors.append(point.vector)
        log.info(f"  scrolled {len(paper_ids)} papers...")
        if offset is None:
            break

    return paper_ids, np.array(vectors, dtype=np.float32)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    client = QdrantClient(url=QDRANT_URL)
    paper_ids, vectors = extract(client)
    log.info(f"Extracted {len(paper_ids)} papers, vectors shape: {vectors.shape}")
