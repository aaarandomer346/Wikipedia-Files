import bz2
import json
import xml.etree.ElementTree as ET
import mwparserfromhell

def process_wikipedia_dump(input_file, output_file, filenum):
    print(f"Processing {input_file}... This may take a while.")
    
    count = 0
    skipped_redirects = 0
    
    # CHANGE THIS LINE:
    with bz2.open(input_file, 'rt', encoding='utf-8') as f_in, \
        open(output_file, 'w', encoding='utf-8', errors='surrogateescape') as f_out:
        
        prefix = ""
        context = ET.iterparse(f_in, events=('start', 'end'))
        context = iter(context)
        
        title = None
        text = None
        
        for event, elem in context:
            if not prefix and event == 'start':
                if '}' in elem.tag:
                    prefix = elem.tag.split('}')[0] + '}'
            
            if event == 'end':
                if elem.tag == f"{prefix}title":
                    title = elem.text
                elif elem.tag == f"{prefix}text":
                    text = elem.text
                elif elem.tag == f"{prefix}page":
                    
                    # 1. Base filter for meta/talk pages
                    if title and text and not title.startswith(('Wikipedia:', 'Template:', 'File:', 'Help:', 'Category:', 'Portal:', 'Talk:', 'Module:')):
                        
                        # 2. NEW FILTER: Skip if the text is just a Wikipedia redirect
                        # Strip whitespace and check if it starts with 'redirect' or '#redirect'
                        stripped_text = text.lstrip()
                        if stripped_text.lower().startswith(('redirect', '#redirect')):
                            skipped_redirects += 1
                        else:
                            # Clean the actual article markup
                            wikicode = mwparserfromhell.parse(text)
                            clean_text = wikicode.strip_code(normalize=True, collapse=False)

                            # NEW: Replace all line breaks (newlines and carriage returns) with a single space
                            # This prevents text from smashing together while removing all \n characters
                            no_newlines_text = " ".join(clean_text.splitlines())

                            # Prepare clean strings by ignoring lone surrogates completely
                            safe_title = title.encode('utf-8', errors='ignore').decode('utf-8')
                            safe_text = " ".join(no_newlines_text.split()).encode('utf-8', errors='ignore').decode('utf-8')

                            entry = {
                                "title": safe_title,
                                "text": safe_text
                            }
                            
                            # Standard json.dumps will now run perfectly with zero errors
                            f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                            count += 1
                        
                        if (count + skipped_redirects) % 20 == 0:
                            print(f"File {filenum}: Progress -> Saved: {count} articles | Skipped: {skipped_redirects} redirects")
                            
                    elem.clear()
                    title = None
                    text = None

    print(f"\nFinished! Successfully saved {count} articles.")
    print(f"Filtered out {skipped_redirects} redirect pages.")
    save_page_count(count, skipped_redirects, filenum)

# Fix the parameter typo here: filsenum -> filenum
def save_page_count(count, skipped_redirects, filenum): 
    try:
        with open("/media/aaarandomer/Windows&Mac/Wikipedia Files/numbers.txt", 'a') as file:
            # Also add a trailing newline \n so entries don't smash into one giant line
            file.write(f"Filenum: {filenum}, Count of Files: {count}, Redirects Skipped: {skipped_redirects}\n")
    except Exception as e:
        print(f"Error! {e}")

if __name__ == "__main__":
    # Define your specific file fragments as tuples: (input_fragment, file_number)
    file_chunks = [
        ("p10p1130124", 1)
    ]
    
    base_dir = "/media/aaarandomer/Windows&Mac/Wikipedia Files"
    
    for chunk, num in file_chunks:
        input_path = f"{base_dir}/enwiki-2026-06-01-{chunk}.xml.bz2"
        output_path = f"{base_dir}/wiki dumps processed into jsonls/{num}.jsonl"
        
        process_wikipedia_dump(input_path, output_path, num)