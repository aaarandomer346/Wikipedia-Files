import os
import json
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfig, PointStruct

drive_db_path = "qdrant_wiki_db"

client = QdrantClient(path=drive_db_path)
collection_name = "wikipedia_files"

client.create_collection(
    collection_name=collection_name,
    vectors_config=(
        size=384,
        Distance=Distance.COSINE,
        on_disk_payload=True
    ),
    hnsw_config=(
        m=16,
        ef_construct=100,
        on_disk=True
    )
)

BATCH_SIZE = 5000
id_count = 0

for i in range(1, 43):
    embedding_file = f"embeddings/embeddings_{i}.npy"
    text_file = f"chunks/output_{i}.jsonl"

    print(f"starting file: {i}")

    vectors = np.load(embedding_file, mmap_mode='r')
    points_batch = []

    with open(text_file, "r", encoding='utf-8') as infile:
        for vector, line in zip(vectors, infile):
            text_data = json.loads(line)
            chunk = text_data.get("chunk", "").strip()

            vector_list = vector.tolist()

            point = PointStruct(
                id=id_count,
                vector=vector_list,
                payload={"text": chunk}
            )
            points_batch.append(point)

            if len(points_batch) == BATCH_SIZE:
                client.upsert(
                    collection_name=collection_name,
                    points=points_batch,
                    wait=False
                )
                points_batch = []
        if points_batch:
            client-upsert(
                collection_name=collection_name,
                points=points_batch,
                wait=False
            )
        print(f"finished file: {i}, total saved: {id_count}")

print(f"yay done!!!")