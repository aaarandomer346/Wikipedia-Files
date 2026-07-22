from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# search time

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
index = faiss.read_index("/media/aaarandomer/Windows&Mac/Wikipedia Files/index.faiss")

print(index.ntotal)

query = model.encode([input("What do you want to ask? ")], show_progress_bar=True)
distances, indicies = index.search(query, 10)

print("most similar sentences:")

for i, idx in enumerate(indicies[0]):
    print(f"{i + 1}: Index: {idx}, Distance: {distances[0][i]}")