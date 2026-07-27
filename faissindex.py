import json
import faiss
import numpy as np
import os, gc
import random




# do 2 random numbers 1 till 42. then load those for training.
# for adding to the index just stream each file






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


quantizer = faiss.IndexFlatL2(384)
index = faiss.IndexIVFPQ(quantizer, 384, 32768, 16, 8)

all_training_files = []
all_files = []

base_dir = "/media/aaarandomer/Windows&Mac1/Wikipedia Files/embeddings/"


for i in range(5):
    while True:
        file = f"{base_dir}embeddings_{random.randint(1, 41)}.npy"
        if file not in all_training_files:
            break
    all_training_files.append(file)

for i in range(1, 43):
    all_files.append(f"{base_dir}embeddings_{i}.npy")

# Train the model
print("starting to train models")
data = [np.load(f, mmap_mode='r') for f in all_training_files]
data_for_training = np.concatenate(data, axis=0)

print("starting training")
index.train(data_for_training)
del data_for_training, data, all_training_files
gc.collect()

curr_file = 0

# Add to the index

print("actually work time now yay")

for current_file in all_files:
    curr_file += 1
    embeddings = np.load(current_file, mmap_mode='r')

    start = (curr_file - 1) * 1000000
    ids = np.arange(start, start + len(embeddings), dtype=np.int64)

    index.add_with_ids(embeddings, ids)

    print(f"done with file: {curr_file}")

faiss.write_index(index, "index.faiss")