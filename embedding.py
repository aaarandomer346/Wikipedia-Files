from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')
quantizer = faiss.IndexFlatL2(384)
index = faiss.IndexIVFPQ(quantizer, 384, 65536, 48, 8)

all_training_files = []
all_files = []

base_dir = "/media/aaarandomer/Windows&Mac/Wikipedia Files/chunks/"

for i in range(1, 7):
    all_training_files.append(f"{base_dir}output_{i}.jsonl")

for i in range(1, 43):
    all_files.append(f"{base_dir}output_{i}.jsonl")

# Train the model
training_embeddings = []

for training_file in all_training_files:
    chunks = []
    with open(training_file, "r", encoding="utf-8") as infiletraining:
        for line in infiletraining:
            page = json.loads(line)
            chunk = page.get("chunk", "").strip()
            chunks.append(chunk)
    training_embeddings.append(model.encode(chunks, batch_size=512, show_progress_bar=True))

training_matrix = np.vstack(training_embeddings).astype('float32')
index.train(training_matrix)
del training_matrix

curr_file = 0

# Add to the index
for current_file in all_files:
    with open(current_file, "r", encoding="utf-8") as infile:
        batch = []
        ids = []
        curr_line = 0
        curr_file += 1

        print(f"starting file: {curr_file}")
        for line in infile:
            curr_line += 1

            page = json.loads(line)
            chunk = page.get("chunk", "").strip()
            batch.append(chunk)

            ids.append(int(f"1{str(curr_file).zfill(2)}{str(curr_line).zfill(6)}")) # will output 1|xx|xxxxxx
        print("done appending chunks")

        embeddings = model.encode(batch, batch_size=512, show_progress_bar=True).astype('float32')
        index.add_with_ids(embeddings, np.array(ids, dtype=np.int64))

        print(f"done with file: {curr_file}")

    if curr_file % 5 == 0:
        faiss.write_index(index, f"index_checkpoint_{curr_file}.faiss")


faiss.write_index(index, "index.faiss")