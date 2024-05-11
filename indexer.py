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
        # if len(hashtable) >= 250:
        #     unload_table(hashtable)
        #     del hashtable
        #     hashtable = {}
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
            tokens = _tokenized_and_stem(" ".join([text.get_text() for text in content]))    # Need to stem the words before this
            return tokens
    except FileNotFoundError:
        print("File " + document + " not found")
    except json.JSONDecodeError as json_err:
        print(json_err.msg)
    except Exception as err:
        print(err)

    # If there is an error, just return an empty list
    return []


# def unload_table(hashtable) -> None:
#     global word_count
#     try:
#         path = "C:\\Users\\Ivan Onofre\\University\\CS 121 - INF 141\\cs121-a3\\indexes.json"
#         if not path.endswith(".json"):
#             raise Exception("Must unload data onto a json file")
#         with open(path, "a") as json_file:
#             for term, postings in hashtable.items():
#                 json_object = json.loads('{"' +
#                                          term + '": {' +
#                                          ','.join([
#                                              '"' + str(posting.document_id) + '": ' +
#                                              str(posting.tfidf_score) for posting in postings]) +
#                                          '} }')
#                 json.dump(json_object, json_file)
#                 # FIXME: Need to find a way to merge files so that previously added terms are updated
#                 word_count += 1
#     except Exception as err:
#         print(err)


# Code to tokenize a document when we have it in string form
def _tokenized_and_stem(text_string):
    # Apply stemming to each token
    stemmer = PorterStemmer()
    tokens = re.findall(r'\b\w+\b', text_string.lower())
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return stemmed_tokens
    

def _computeWordWeights(tokens_list):
    # Initializing dictionary/map of the tokens and the number of occurrences
    word_freq = {}
    # Iterates through all the values in the token list and updates its value in the corresponding dictionary/mapping
    for token in tokens_list:
        # Checks the value of the token in the dictionary and sets it to the appropriate value either 1 or +1
        word_freq[token] = word_freq.get(token,0)  + 1


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