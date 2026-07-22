import json
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

base_dir = "/Volumes/Windows&Mac/Wikipedia Files/"

for i in range(1, 42):
    chunks = []
    with open(f"{base_dir}chunks/output_{i}.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            chunks.append(json.loads(line).get("chunk", "").strip())

    embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True).astype('float32')
    np.save(f"{base_dir}/embeddings/embeddings_{i}.npy", embeddings)
    print(f"done file num {i}")

    del embeddings, chunks