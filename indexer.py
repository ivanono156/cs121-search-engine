import json
import math
import os
import sys
import re
import urllib.parse
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from posting import Posting

# Global var to keep track of unique words
unique_words = set()
# Global dict to keep track of which document id maps to which document url
doc_ids_to_urls = {}
# Global list to keep track of document lengths, used to normalize term freqs
doc_lengths = []


def build_index(documents) -> dict[str, list[Posting]]:
    global unique_words
    # Create hashtable
    # Mapping = token: posting (document id)
    hashtable = {}
    # Set threshold for when to offload hashtable to json file
    doc_threshold = 18000
    counter = 0
    offload_count = 0
    # Use enumerate to map each doc to an id (n)

    for n, document in enumerate(documents):
        # T <- Parse documents
        # Remove duplicates from T
        tokens = parse(document, n)
        doc_length = sum(freq for freq in tokens.values()) if len(tokens) > 0 else 0
        doc_lengths.append(doc_length)
        # Add each token to the hashtable
        for token in tokens:
            # Initialize new tokens
            unique_words.add(token)
            term_freq = tokens[token]
            if token not in hashtable:
                hashtable[token] = []
            # calculate the log base 10 frequency weight of the term
            log_freq_weight = 1 + math.log10(term_freq) if term_freq > 0 else 0
            # print(f"term: {token}, tf: {term_freq} log freq weight: {log_freq_weight}")
            # Map each token to its posting (which contains this document's id)
            hashtable[token].append(Posting(n, log_freq_weight))
        if counter >= doc_threshold:
            unload_to_disk(hashtable, offload_count)
            offload_count += 1
            del hashtable
            hashtable = {}
            counter = 0
        counter += 1
    # Offloads rest of hashtable before returning
    if hashtable:
        unload_to_disk(hashtable, offload_count)
        offload_count += 1
        del hashtable
    # Merges all partial indexes into one
    hashtable = merge_partial_indexes(offload_count)

    # Store doc ids to url mappings
    store_table_as_json("document_ids_to_urls.json", doc_ids_to_urls, True)

    return hashtable


# Parse the json file and return the tokens from the file
def parse(document, doc_id) -> dict[str, int]:
    try:
        with open(document, 'r') as file:
            # Create json object (can access object like a dictionary)
            json_object = json.load(file)

            url = urllib.parse.urldefrag(json_object["url"]).url  # Remove fragments from urls
            if url in doc_ids_to_urls.values():
                # print("Duplicate url: " + url)
                return {}  # skip this url
            doc_ids_to_urls[doc_id] = url
            # Parse content using bs4 (passing in json_object["content"])
            soup = BeautifulSoup(json_object["content"], features="lxml-xml", from_encoding=json_object["encoding"])
            # Find all important text (bold text and header text)
            page_elements = soup.find_all(
                ["strong", "b", "h1", "h2", "h3", "h4", "h5", "h6", "em", "p", "ul", "ol", "li", "blockquote",
                 "a", "article", "section"])  # Still need to test for better/more valuable headers
            # TODO: Also find a way to get the rest of the text besides the important text
            tokens = _tokenized_and_stem(" ".join([element.get_text() for element in page_elements]))
            return tokens
    except FileNotFoundError:
        print("File " + document + " not found")
    except json.JSONDecodeError as json_err:
        print("JSON error" + json_err.msg)
    except Exception as err:
        print("Error: " + str(err))

    # If there is an error, just return an empty dict
    return dict()


def unload_to_disk(index, off_count):
    # Makes each posting serialized into dictionary so can be put in JSOn file
    serialized_index = {
        term: [posting.to_dict() for posting in postings] for term, postings in index.items()
    }

    filename = f"partial_index{off_count}.json"
    # Write the serialized index to disk
    with open(filename, 'w') as file:
        json.dump(serialized_index, file, indent=4, sort_keys=True)  # Printing with indent of 4 for readability
    return {}


# Merges all partial indexes into one final index file and dictionary
def merge_partial_indexes(off_count):
    # Initialize final mapping
    final_index = {}

    # Will go through each partial index made and collect the information needed for the final index
    for i in range(off_count):
        filename = f"partial_index{i}.json"
        with open(filename, 'r') as file:
            partial_index = json.load(file)
            for token, postings in partial_index.items():
                if token in final_index:
                    final_index[token].extend(postings)
                else:
                    final_index[token] = postings

    # Dict to keep track of terms and their positions in the index file (for use with seek later)
    token_offsets = {}
    total_docs = len(doc_lengths)  # used to calculate idf
    # This will be used to calculate each document's vector magnitude when performing the search queries
    doc_sum_of_squares = [0.0] * total_docs  # initialize sum of squares for each doc to zero

    with open('final_index.txt', 'wb') as file:
        # iterate through the dictionary in alphabetical order
        for token, postings in sorted(final_index.items(), key=lambda kv_pair: kv_pair[0]):
            doc_freq = len(postings)  # the number of documents that contain this term/token
            idf = math.log10(total_docs / doc_freq)  # calculate idf score for this token
            # combine the token's idf score with the existing tf score for each posting
            for posting in postings:
                tf = posting['tfidf_score']  # get the existing tf score
                tfidf_score = tf * idf  # calculate the raw tfidf score
                posting['tfidf_score'] = round(tfidf_score, 2)  # round tfidf score to 2 decimal places and store it
                # Compute the sum of squares for this document
                doc_id = int(posting['document_id'])
                doc_len = doc_lengths[doc_id]
                doc_sum_of_squares[doc_id] += math.pow(tf / doc_len * idf, 2)  # square tfidf score and add to sum
                # print(f"term: {token}, tf: {tf}, idf: {idf}, tfidf_score: {tfidf_score}")

            offset = file.tell()
            token_offsets[token] = offset
            # postings is a list of dicts
            postings_str = ";".join([f"({p['document_id']},{p['tfidf_score']})" for p in postings])
            # print("T,P: " + token + ":" + postings_str)
            s = token + ":" + postings_str + "\n"
            file.write(s.encode('utf-8'))  # write string as bytes to file

    store_table_as_json("document_lengths.json", {i: length for i, length in enumerate(doc_lengths)}, True)
    store_table_as_json("term_offsets.json", token_offsets, True)
    create_document_magnitudes(doc_sum_of_squares)

    # Returns final index, can comment out to save memory?
    return final_index


def create_document_magnitudes(doc_sum_of_squares):
    doc_magnitudes = {}
    for i, sum_of_square in enumerate(doc_sum_of_squares):
        doc_magnitudes[i] = math.sqrt(sum_of_square)
    store_table_as_json("document_magnitudes.json", doc_magnitudes, True)


# store diff tables (to json format most likely)
def store_table_as_json(file_name, table, sort=False):
    try:
        with open(file_name, 'w') as file:
            json.dump(table, file, indent=4, sort_keys=sort)
    except Exception as e:
        print("Could not write to file " + file_name + ": " + str(e))


# Code to tokenize a document when we have it in string form
def _tokenized_and_stem(text_string):
    # Apply stemming to each token
    stemmer = PorterStemmer()
    stemmed_tokens = {}
    tokens = re.findall(r'\b\w+\b', text_string.lower())
    for token in tokens:
        rev_token = stemmer.stem(token)
        if rev_token not in stemmed_tokens:
            stemmed_tokens[rev_token] = 1
        else:
            stemmed_tokens[rev_token] += 1
    return stemmed_tokens


def get_documents(directory):
    # Transform the relative paths into absolute paths
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(root, file)
            files.append(file_path)
    return files


if __name__ == "__main__":
    word_count = 0
    # Path to the folder containing the documents
    folder_path = sys.argv[1]
    docs = get_documents(folder_path)
    print("Building index...")
    index_table = build_index(docs)
    print("Building index complete")

    print("The number of indexed documents:", len(docs))
    print("The number of unique words:", len(unique_words))
