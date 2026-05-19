import os
import numpy as np
from qdrant_client import QdrantClient

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
        if offset is None:
            break

    return paper_ids, np.array(vectors, dtype=np.float32)


if __name__ == "__main__":
    client = QdrantClient(url=QDRANT_URL)
    paper_ids, vectors = extract(client)
    print(f"Extracted {len(paper_ids)} papers, vectors shape: {vectors.shape}")
