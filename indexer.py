import json
import os
import sys
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from posting import Posting
import re

unique_words = set()
# Skeleton code from lecture slides
def build_index(documents) -> dict[str, list[Posting]]:
    global unique_words
    # Create hashtable
    # Mapping = token: posting (document id)
    hashtable = {}
    #Set threshold for when to offload hashtable to json file
    doc_threshold = 19000
    counter = 0
    offload_count = 0
    # Use enumerate to map each doc to an id (n)
    
    for n, document in enumerate(documents):
        # T <- Parse documents
        # Remove duplicates from T
        tokens = set(parse(document))
        # Add each token to the hashtable
        for token in tokens:
            # Initialize new tokens
            unique_words.add(token)
            if token not in hashtable:
                hashtable[token] = []
            # Map each token to its posting (which contains this document's id)
            hashtable[token].append(Posting(n, 0))
        if counter >= doc_threshold:
            unload_to_disk(hashtable,offload_count)
            offload_count += 1
            del hashtable
            hashtable = {}
            counter = 0
        counter += 1
    if hashtable:
            unload_to_disk(hashtable,offload_count)
            offload_count += 1
            del hashtable
            hashtable = {}
    hashtable = merge_partial_indexes(offload_count)
    return hashtable


# Parse the json file and return the tokens from the file
def parse(document) -> list[str]:
    try:
        with open(document, 'r') as file:
            # Create json object (can access object like a dictionary)
            json_object = json.load(file)
            # print("Json File:", file.name)
            # print(" Url:", json_object["url"])
            # print(" Encoding:", json_object["encoding"])
            # Parse content using bs4 (passing in json_object["content"])
            soup = BeautifulSoup(json_object["content"], "lxml-xml")    # Do we need to use the encoding for this step?
            # Find all important text (bold text and header text)
            content = soup.find_all(["strong", "b", "h1", "h2", "h3"])
            # FIXME: Also find a way to get the rest of the text besides the important text
            tokens = _tokenized_and_stem(" ".join([text.get_text() for text in content]))   
            return tokens
    except FileNotFoundError:
        print("File " + document + " not found")
    except json.JSONDecodeError as json_err:
        print(json_err.msg)
    except Exception as err:
        print(err)

    # If there is an error, just return an empty list
    return []


def unload_to_disk(index,off_count):
    # Makes each posting serialized into dictionary so can be put in JSOn file
    serialized_index = {
        term: [posting.to_dict() for posting in postings] for term, postings in index.items()
    }

    filename = f"partial_index{off_count}.json"
    # Write the serialized index to disk
    with open(filename, 'w') as file:
        json.dump(serialized_index, file, indent=4)  # Printing with indent of 4 for readability
    return {}

# Merges all partial indexes into one final index file and dictionary
def merge_partial_indexes(off_count):
    
    final_index = {}
    
    #Will go through each partial index made and collect the information needed for the final index
    for i in range(off_count):
        filename = f"partial_index{i}.json"
        with open(filename, 'r') as file:
            partial_index = json.load(file)
            for token,docs in partial_index.items():
                if token in final_index:
                    final_index[token].extend(docs)
                else:
                    final_index[token] = docs
    

    with open('final_index','w') as file:
        json.dump(final_index, file, indent=4)
        
    #return final_index
# Code to tokenize a document when we have it in string form
def _tokenized_and_stem(text_string):
    # Apply stemming to each token
    stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', text_string.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens

if __name__ == "__main__":
    word_count = 0
    # Path to the folder containing the documents
    folder_path = sys.argv[1]
    # Transform the relative paths into absolute paths
    files = []
    for root, _, filenames in os.walk(folder_path):
        for file in filenames:
            file_path = os.path.join(root, file)
            files.append(file_path)

    table = build_index(files)
    # # Print each token in the table (sorted by alphabetical order)
    # for k, v in sorted(table.items(), key=lambda item: item[0]):
    #     print(k, ":", v)
    print("The number of indexed documents:", len(files))
    print("The number of unique words:", len(unique_words))