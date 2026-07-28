import os
import json
import numpy as np
from itertools import islice
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, Batch, OptimizersConfigDiff




import psutil, threading

def monitor():
    count = 0
    while True:
        count += 1
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=60)
        print(f"Minute: {count} |  RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB | CPU: {cpu}%")

t = threading.Thread(target=monitor, daemon=True)
t.start()




drive_db_path = "qdrant_wiki_db"
client = QdrantClient(path=drive_db_path)
collection_name = "wikipedia_files"

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=384,
        distance=Distance.COSINE,
        on_disk=True
    ),
    hnsw_config=HnswConfigDiff(
        m=16,
        ef_construct=100,
        on_disk=True
    ),
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=0
    )
)

BATCH_SIZE = 5000
id_count = 0

for i in range(1, 43):
    embedding_file = f"embeddings/embeddings_{i}.npy"
    text_file = f"chunks/output_{i}.jsonl"

    print(f"starting file: {i}")

    vectors = np.load(embedding_file, mmap_mode='r')

    with open(text_file, "r", encoding='utf-8') as infile:
        stard_idx = 0

        while True:
            batch = list(islice(infile, BATCH_SIZE))
            if not batch:
                break
            curr_batch_size = len(batch)
            end_idx = start_idx + curr_batch_size

            vector_batch = vectors[start_idx:end_idx].tolist()

            payload = [
                {"text": json.loads(line).get("chunk", "").strip()}
                for line in batch
            ]

            ids = list(range(id_count, id_count + curr_batch_size))

            client.upsert(
                collection_name=collection_name,
                points=Batch(
                    ids=ids,
                    vectors=vector_batch,
                    payloads=payload
                ),
                wait=False
            )

            id_count += curr_batch_size
            start_idx = end_idx
        print(f"finished file: {id}, total saved: {id_count}")


client.update_collection(
    collection_name=collection_name,
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000
    )
)


print(f"yay done!!!")