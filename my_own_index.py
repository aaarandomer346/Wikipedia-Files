# lowkey not an index and maybe inneficient highly
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import gc
from numba import jit

"""

Use Cosign similarity to check, idgaf about time, that i have lots. Iterate through all chunks (yes all) and compare using cosign similarity for all embeddings, store the top n (user input) as a list with the index and file number. Then just find the chunk and write to a .txt file.

import numpy as np

# Example: Two 384-dimensional embeddings (simulated here with random data)
embedding1 = np.random.rand(384)
embedding2 = np.random.rand(384)

# 1. Calculate cosine similarity
dot_product = np.dot(embedding1, embedding2)
magnitudes = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
cosine_similarity = dot_product / magnitudes

# 2. Convert to a 0% to 100% scale
# (Cosine similarity ranges from -1 to 1, so we map it to 0 to 100)
similarity_percentage = ((cosine_similarity + 1) / 2) * 100

print(f"Cosine Similarity: {cosine_similarity:.4f}")
print(f"Similarity Score: {similarity_percentage:.2f}%")

"""



import psutil, threading, time

def monitor():
    count = 0
    while True:
        time.sleep(60)
        count += 1
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=None)
        print(f"Minute: {count} |  RAM: {mem.used/1e9:.1f}/{mem.total/1e9:.1f} GB | CPU: {cpu}%")

t = threading.Thread(target=monitor, daemon=True)
t.start()




chunks_dir = "chunks/"
embeddings_dir = "embeddings/"

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

question_encoding = model.encode(input("What do you want to search? "), show_progress_bar=True)

print("searching......")



@jit(nopython=True)
def cosine_sim(uq, em):
    assert uq.shape[0] == em.shape[0]
    uqem = np.dot(em, uq)
    uqd = np.dot(uq, uq)
    emd = np.dot(em, em)
    return uqem / np.sqrt(uqd * emd)



top_10 = [[-100000, -1, -1]] # will have file number and line number

for i in range(1, 43):
    print(f"going through file: {i}")
    count = 0
    vectors = np.load(f"{embeddings_dir}embeddings_{i}.npy", mmap_mode='r')
    for idx, vec in enumerate(vectors):
        cosine = cosine_sim(question_encoding, vec)

        for idxx, val in enumerate(top_10):
            if val[0] < cosine:
                top_10.insert(idxx, [cosine, i, idx])
                break
        if len(top_10) > 10:
            top_10.pop()
    del vectors
    gc.collect()

print(top_10)