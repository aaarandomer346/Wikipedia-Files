from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import ollama


def write_to_file(infile, out, file_num, chunk_pos, chunk_num):
    line_count = 0
    for line in infile:
        line_count += 1
        if line_count == chunk_pos:
            out.write(f"\n\nChunk: ({json.loads(line).get("chunk", "").strip()})")
            return f"Chunk {chunk_num}: {json.loads(line).get("chunk", "").strip()}"

def ask_bot(role, query, dostream):
    response = ollama.chat(
        model='llama3.2:1b',
        messages=[
            {'role': 'system', 'content': role},
            {'role': 'user', 'content': query}
        ],
        options={
            'temperature': 0.1,
            'top_p': 0.1
        },
        stream=dostream
    )
    if dostream == True:
        for chunk in response:
            print(chunk.message.content, end='', flush=True)
    else:
        return response.message.content



to_wiki_role = 'Do not answer any questions asked, just convert it into a statement with similar language to wikipedia that can then be used in a AI similarity search index query.'

answer_question_role = 'You are an assistant that will use the information provided in the message in order to answer the question the user has asked, while accounting for the other parameters. Disregard any information not usefull to the question, only use information in the message to answer the question. If no information can answer the question respond with: "ERROR NO GOOD INFORMATION PROVIDED" and do not use any other information.'


# search time


model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
index = faiss.read_index("/media/aaarandomer/Windows&Mac1/Wikipedia Files/index.faiss")

print(index.ntotal)

index.nprobe = 32768

user_question = input("What question do you have? ")
other_parameters = input("Any other parameters, like specific word limit, type here: ")

bot_encoding = ask_bot(to_wiki_role, user_question, False)
print(bot_encoding)

query = model.encode([bot_encoding])
distances, indicies = index.search(query, 5)

print("most similar sentences:")

chunks_out = []

with open("output_text.txt", "w", encoding='utf-8') as out:
    for i, idx in enumerate(indicies[0]):
        file_num = idx // 1_000_000 + 1
        chunk_pos = idx % 1_000_000
        print(f"{i + 1}: File Number: {file_num}, Line Position: {chunk_pos}, Distance: {distances[0][i]}")
        with open(f"chunks/output_{file_num}.jsonl", "r", encoding='utf-8') as in_file:
            chunks_out.append(write_to_file(in_file, out, file_num, chunk_pos, i + 1))

final_query = f'User Query: {user_question} \nAny Other Parameters: {other_parameters} \n\nInformation: {chunks_out}'

print("asking bot")
ask_bot(answer_question_role, final_query, True)