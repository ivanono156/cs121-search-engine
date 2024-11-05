import json
import math
import os
import sys
import re
import urllib.parse
from bs4 import BeautifulSoup
from nltk import PorterStemmer
from posting import Posting


def get_documents(directory: str) -> list[str]:
    # Transform the relative paths into absolute paths
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(root, file)
            files.append(file_path)
    return files


def stem_tokens(text_string):
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


def write_to_json_file(file_name, table, sort=False):
    try:
        with open(file_name, 'w') as file:
            json.dump(table, file, indent=4, sort_keys=sort)
    except Exception as e:
        print("Could not write to file " + file_name + ": " + str(e))


def create_document_magnitudes(doc_sum_of_squares):
    doc_magnitudes = {}
    for i, sum_of_square in enumerate(doc_sum_of_squares):
        doc_magnitudes[i] = math.sqrt(sum_of_square)
    write_to_json_file(Indexer.DOCUMENT_MAGNITUDES_FILE, doc_magnitudes, True)


def unload_to_disk(index, off_count):
    # Makes each posting serialized into dictionary so can be put in JSOn file
    serialized_index = {
        term: [posting.to_dict() for posting in postings] for term, postings in index.items()
    }

    filename = get_index_file_path(Indexer.INDEXES_DIRECTORY,
                                   Indexer.PARTIAL_INDEX_FILE.replace("#", str(off_count)))
    # Write the serialized index to disk
    with open(filename, 'w') as file:
        json.dump(serialized_index, file, indent=4, sort_keys=True)  # Printing with indent of 4 for readability


def get_index_file_path(directory, file_name):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory " + directory + " created")
    return os.path.join(directory, file_name)


class Indexer:
    # Set threshold for when to offload hashtable to json file
    OFFLOAD_THRESHOLD = 5000
    # Index files
    PARTIAL_INDEX_FILE = "partial_index#.json"
    FINAL_INDEX_FILE = "final_index.txt"
    INDEXES_DIRECTORY = "indexes/"
    # other json files
    TERM_OFFSETS_FILE = "term_offsets.json"
    DOCUMENT_MAGNITUDES_FILE = "document_magnitudes.json"
    DOCUMENT_IDS_TO_URLS_FILE = "document_ids_to_urls.json"
    DOCUMENT_LENGTHS_FILE = "document_lengths.json"
    LINKS_GRAPH_FILE = "link_graph.json"

    def __init__(self):
        self.unique_words = set()
        # Dict to keep track of which document id maps to which document url
        self.doc_ids_to_urls = {}
        # List to keep track of document lengths, used to normalize term freqs
        self.doc_lengths = []
        self.link_graph = {}

    def build_index(self, documents) -> dict[str, list[Posting]]:
        # Create hashtable
        # Mapping = token: posting (document id)
        hashtable = {}
        offload_count = 0
        # Use enumerate to map each doc to an id (n)
        for n, document in enumerate(documents):
            # T <- Parse documents
            # Remove duplicates from T
            tokens = self.parse(document, n)
            doc_length = sum(freq for freq in tokens.values()) if len(tokens) > 0 else 0
            self.doc_lengths.append(doc_length)
            # Add each token to the hashtable
            for token in tokens:
                # Initialize new tokens
                self.unique_words.add(token)
                term_freq = tokens[token]
                if token not in hashtable:
                    hashtable[token] = []
                # calculate the log base 10 frequency weight of the term
                log_freq_weight = 1 + math.log10(term_freq) if term_freq > 0 else 0
                # print(f"term: {token}, tf: {term_freq} log freq weight: {log_freq_weight}")
                # Map each token to its posting (which contains this document's id)
                hashtable[token].append(Posting(n, log_freq_weight))

            if (n > 0 and n % self.OFFLOAD_THRESHOLD == 0) or n == len(documents) - 1:
                unload_to_disk(hashtable, offload_count)
                offload_count += 1
                del hashtable
                hashtable = {}
                print(f"OFFLOAD #{offload_count}: {n} documents offloaded")
                if n == len(documents) - 1:
                    print(f"OFFLOADING COMPLETE: n == {len(documents) - 1} (len(documents) - 1)")

        # Store doc ids to url mappings
        write_to_json_file(self.DOCUMENT_IDS_TO_URLS_FILE, self.doc_ids_to_urls, True)
        # Save the link graph
        write_to_json_file(self.LINKS_GRAPH_FILE, self.link_graph, True)

        # Merges all partial indexes into one
        return self.merge_partial_indexes(offload_count)

    def parse(self, document, doc_id) -> dict[str, int]:
        try:
            with open(document, 'r') as file:
                json_object = json.load(file)
                url = urllib.parse.urldefrag(json_object["url"]).url
                if url in self.doc_ids_to_urls.values():
                    return {}  # skip this url
                self.doc_ids_to_urls[doc_id] = url

                soup = BeautifulSoup(json_object["content"], features="lxml-xml", from_encoding=json_object["encoding"])
                page_elements = soup.find_all(
                    ["strong", "b", "h1", "h2", "h3", "h4", "h5", "h6", "em", "p", "ul", "ol", "li", "blockquote",
                     "a", "article", "section"])
                tokens = stem_tokens(" ".join([element.get_text() for element in page_elements]))

                # Extract links
                links = []
                for link in soup.find_all('a', href=True):
                    link_url = urllib.parse.urldefrag(link['href']).url
                    for doc_id, doc_url in self.doc_ids_to_urls.items():
                        if doc_url == link_url:
                            links.append(doc_id)
                            break

                # Add links to link graph
                self.link_graph[doc_id] = links

                return tokens
        except FileNotFoundError:
            print("File " + document + " not found")
        except json.JSONDecodeError as json_err:
            print("JSON error: " + json_err.msg)
        except Exception as err:
            print("Error: " + str(err))
        return {}

    # Merges all partial indexes into one final index file and dictionary
    def merge_partial_indexes(self, off_count):
        # Initialize final mapping
        final_index = {}

        # Will go through each partial index made and collect the information needed for the final index
        for i in range(off_count):
            filename = get_index_file_path(Indexer.INDEXES_DIRECTORY,
                                           Indexer.PARTIAL_INDEX_FILE.replace("#", str(i)))
            with open(filename, 'r') as file:
                partial_index = json.load(file)
                for token, postings in partial_index.items():
                    if token in final_index:
                        final_index[token].extend(postings)
                    else:
                        final_index[token] = postings

        # Dict to keep track of terms and their positions in the index file (for use with seek later)
        token_offsets = {}
        total_docs = len(self.doc_lengths)  # used to calculate idf
        # This will be used to calculate each document's vector magnitude when performing the search queries
        doc_sum_of_squares = [0.0] * total_docs  # initialize sum of squares for each doc to zero

        final_index_file = get_index_file_path(self.INDEXES_DIRECTORY, self.FINAL_INDEX_FILE)
        with open(final_index_file, 'wb') as file:
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
                    doc_len = self.doc_lengths[doc_id]
                    doc_sum_of_squares[doc_id] += math.pow(tf / doc_len * idf, 2)  # square tfidf score and add to sum
                    # print(f"term: {token}, tf: {tf}, idf: {idf}, tfidf_score: {tfidf_score}")

                offset = file.tell()
                token_offsets[token] = offset
                # postings is a list of dicts
                postings_str = ";".join([f"({p['document_id']},{p['tfidf_score']})" for p in postings])
                # print("T,P: " + token + ":" + postings_str)
                s = token + ":" + postings_str + "\n"
                file.write(s.encode('utf-8'))  # write string as bytes to file

        write_to_json_file(self.DOCUMENT_LENGTHS_FILE, {i: length for i, length in enumerate(self.doc_lengths)}, True)
        write_to_json_file(self.TERM_OFFSETS_FILE, token_offsets, True)
        create_document_magnitudes(doc_sum_of_squares)

        return final_index


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = sys.argv[1]
    docs = get_documents(folder_path)
    print("Building index...")
    indexer = Indexer()
    index_table = indexer.build_index(docs)
    print("Building index complete")

    print("The number of indexed documents:", len(indexer.doc_ids_to_urls))
    print("The number of unique words:", len(indexer.unique_words))
