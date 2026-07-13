import json

# Meta-namespaces that MediaWiki prepends to non-article titles
meta_namespaces = {
    "talk", "user", "user talk", "wikipedia", "wikipedia talk", 
    "file", "file talk", "mediawiki", "mediawiki talk", 
    "template", "template talk", "help", "help talk", 
    "category", "category talk", "portal", "portal talk", 
    "draft", "draft talk", "module", "module talk",
    "gadget", "gadget talk"
}

disambig_keywords = {"disambiguation", "disambig", "disam"}

total_saved_files = 0

print("Filtering using title parsing...")

def make_valid_jsonl(input_file, output_file, meta_namespaces, disambig_keywords, file_num):
    with open(input_file, "r", encoding="utf-8") as infile, \
        open(output_file, "a", encoding="utf-8") as outfile:
        
        processed_count = 0
        saved_count = 0
        
        for line in infile:
            processed_count += 1
            
            try:
                page = json.loads(line)
                title = page.get("title", "").strip()
                
                # If the title has a colon, check if it's a metadata prefix
                if ":" in title:
                    first_part = title.split(":")[0].strip().lower()
                    if first_part in meta_namespaces:
                        continue  # It's a meta page, skip it!

                # Keep the valid article
                outfile.write(json.dumps(page) + "\n")
                saved_count += 1
                
                if processed_count % 100 == 0:
                    print(f"File: {file_num}: Processed {processed_count} lines... Kept {saved_count} articles.")
                    
            except json.JSONDecodeError:
                continue

    print(f"\nDone! Filtered down to {saved_count} true articles.")
    return saved_count, processed_count

def save_page_count(count, skipped, filenum): 
    try:
        with open("/media/aaarandomer/Windows&Mac/Wikipedia Files/numbers.txt", 'a') as file:
            # Also add a trailing newline \n so entries don't smash into one giant line
            file.write(f"Filenum: {filenum}, Count of Files: {count}, Redirects Skipped: {skipped}\n")
    except Exception as e:
        print(f"Error! {e}")

# Define your specific file fragments as tuples: (input_fragment, file_number)
file_chunks = [
    (1, "1_shortened"),
    (2, "2_shortened"),
    (3, "3_shortened"),
    (4, "4_shortened"),
    (5, "5_shortened"),
    (6, "6_shortened"),
    (7, "7_shortened"),
    (8, "8_shortened"),
    (9, "9_shortened"),
    (10, "10_shortened"),
    (11, "11_shortened"),
    (12, "12_shortened"),
    (13, "13_shortened"),
    (14, "14_shortened"),
    (15, "15_shortened"),
    (16, "16_shortened"),
    (17, "17_shortened"),
    (18, "18_shortened"),
    (19, "19_shortened")
]
    
base_dir = "/media/aaarandomer/Windows&Mac/Wikipedia Files"

for filenameinput, filenameoutput in file_chunks:
    input_path = f"{base_dir}/wiki dumps processed into jsonls/{filenameinput}.jsonl"
    output_path = f"{base_dir}/wiki dumps only articles/{filenameoutput}.jsonl"
        
    saved_articles, total_processed = make_valid_jsonl(input_path, output_path, meta_namespaces, disambig_keywords, filenameinput)
    total_saved_files += saved_articles
    print(f"Total saved files: {total_saved_files}")
    skipped = total_processed - saved_articles
    save_page_count(saved_articles, skipped, filenameinput)