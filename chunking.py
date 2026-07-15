# go through each jsonl file containing the articles
# go through each line, get the raw text
# make chunks based on 500-10 words (490 words + 10 from the previous chunk)
# save this to a new jsonl file with the text, file number, and title of the wikipage it is found in
# every 10,000 chunks create a new jsonl file (important for embedding later)

import json

def write_chunk_to_json(chunk, cc, article_name, outfile):
    # print("writing to outfile")
    with open(outfile, "a", encoding="utf-8") as out:
        entry = {
            "from_file": 1,
            "article_name": article_name,
            "chunk_count": cc,
            "chunk": chunk
        }
        out.write(json.dumps(entry) + "\n")

def add_to_chunk_or_word(char, word, chunk, cwi):
    if char != " ":
        word = "".join([word, char])
    else:
        chunk = " ".join([chunk, word])
        word = ""
        cwi += 1
    return word, chunk, cwi

def handle_adding(word, chunk, cwi, previous_text, char, cc, article_name, cwcl, filecount, total_chunk_count):
    word, chunk, cwi = add_to_chunk_or_word(char, word, chunk, cwi)
    # print(cwi, cwcl - 10)
    # print(cwi)
    if cwi >= cwcl - 26 and cwi < cwcl:
        previous_text = "".join([previous_text, char])

    elif cwi >= cwcl:
        cc += 1
        total_chunk_count += 1
        # print("writing to json")
        if total_chunk_count % 25000 == 0:
            filecount += 1
        outfile = f"/media/aaarandomer/Windows&Mac/Wikipedia Files/chunks/output_{filecount}.jsonl"
        write_chunk_to_json(chunk, cc, article_name, outfile)
        print(f"total chunk count: {total_chunk_count}")
        chunk = ""
        word = ""
        cwi = 0
    
    return word, chunk, cwi, previous_text, cc, filecount, total_chunk_count

def get_article_length(article):
    words = 0
    characters = 0
    for char in article:
        if char != " ":
            characters += 1
        else:
            words += 1
    return words

filecount = 1
total_chunk_count = 0

with open("/media/aaarandomer/Windows&Mac/Wikipedia Files/wiki dumps only articles/1_shortened.jsonl", "r", encoding="utf-8") as in_file:
    for line in in_file:
        
        word = ""
        chunk = ""
        prev_text = ""
        cwi = 0 # current word index
        cc = 0 # chunk count
        cwcl = 256 # chunk word count length
        
        added_prev = False

        page = json.loads(line)
        article = page.get("text", "").strip()
        article_name = page.get("title", "").strip()

        article_length = get_article_length(article)
        lcwc = article_length % cwcl # last chunk word count
        blc = (article_length - lcwc) / cwcl # before last chunk 

        print(article_length, lcwc, blc)

        for char in article:
            # print(cc)
            if cc != 0 and added_prev == False:
                chunk = prev_text
                added_prev = True

            if cc < blc:
                # print("Doing chunks that isnt last chunk")
                word, chunk, cwi, prev_text, cc, filecount, total_chunk_count = handle_adding(word, chunk, cwi, prev_text, char, cc, article_name, cwcl, filecount, total_chunk_count)
            elif cc == blc:
                # print("cc == blc")
                # print("doing last chunk")
                word, chunk, cwi, prev_text, cc, filecount, total_chunk_count = handle_adding(word, chunk, cwi, prev_text, char, cc, article_name, lcwc, filecount, total_chunk_count)
                # print(lcwc)
            else:
                print("done")
                break