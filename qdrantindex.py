import os
import orjson
import gc
import numpy as np
from itertools import islice
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, Batch, OptimizersConfigDiff, PointStruct




import psutil, threading

def monitor():
    count = 0
    while True:
        count += 0.25
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=15)
        print(f"Minute: {count} |  RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB | CPU: {cpu}%")

t = threading.Thread(target=monitor, daemon=True)
t.start()




drive_db_path = "qdrant_wiki_db"
client = QdrantClient(path=drive_db_path)
collection_name = "wikipedia_files"

if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    on_disk_payload=True,
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

BATCH_SIZE = 25000
id_count = 0

for i in range(1, 43):
    embedding_file = f"embeddings/embeddings_{i}.npy"
    text_file = f"chunks/output_{i}.jsonl"

    print(f"starting file: {i}")

    vectors = np.load(embedding_file, mmap_mode='r')


    def stream_points():
        with open(text_file, 'r', encoding='utf-8') as infile:
            for idx, line in enumerate(infile):
                yield PointStruct(
                    id = id_count+idx,
                    vector=vectors[idx].tolist(),
                    payload={"text": orjson.loads(line).get("chunk", "").strip()}
                )

    client.upload_points(
        collection_name=collection_name,
        points=stream_points(),
        batch_size=500,
        parallel=1,
        wait=True
    )

    points_in_file = vectors.shape[0]
    id_count += points_in_file

    del vectors
    gc.collect()
    
    print(f"finished file: {i}, total saved: {id_count}")


client.update_collection(
    collection_name=collection_name,
    optimizers_config=OptimizersConfigDiff(
        indexing_threshold=20000
    )
)


print(f"yay done!!!")