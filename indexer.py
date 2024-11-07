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
    """Transform the relative paths into absolute paths"""
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            file_path = os.path.join(root, file)
            files.append(file_path)
    return files


def stem_tokens(text_string: str) -> dict[str, int]:
    """Apply stemming to each token"""
    stemmer = PorterStemmer()
    stemmed_tokens = {}
    tokens = re.findall(r'\b\w+\b', text_string.lower())
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        if stemmed_token not in stemmed_tokens:
            stemmed_tokens[stemmed_token] = 1
        else:
            stemmed_tokens[stemmed_token] += 1
    return stemmed_tokens


def write_to_json_file(data: dict, file_name: str) -> None:
    try:
        with open(file_name, 'w') as file:
            json.dump(data, file, indent=4, sort_keys=True)
    except Exception as e:
        print("Could not write to json file " + file_name + ": " + str(e))


def ensure_directory_exists(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Directory " + directory + " created")


class Indexer:
    # Set threshold for when to offload hashtable to json file
    OFFLOAD_THRESHOLD = 1000
    # Index files
    INDEXES_DIRECTORY = "indexes"
    PARTIAL_INDEX_FILE = "partial_index#.json"
    FINAL_INDEX_FILE = "final_index.txt"
    # Helper files
    HELPERS_DIRECTORY = "helpers"
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
        ensure_directory_exists(self.INDEXES_DIRECTORY)
        ensure_directory_exists(self.HELPERS_DIRECTORY)

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
                self.unload_to_disk(hashtable, offload_count)
                offload_count += 1
                del hashtable
                hashtable = {}
                percentage = round(n / (len(documents) - 1) * 100, 2)
                print(f"OFFLOAD #{offload_count} ({percentage}% complete): {n}/{len(documents)-1} documents offloaded")

        write_to_json_file(self.doc_ids_to_urls,
                           os.path.join(self.HELPERS_DIRECTORY, self.DOCUMENT_IDS_TO_URLS_FILE))
        write_to_json_file(self.link_graph,
                           os.path.join(self.HELPERS_DIRECTORY, self.LINKS_GRAPH_FILE))

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

    def unload_to_disk(self, index, off_count):
        # Serialize each posting into a dictionary so that it can be put into a JSON file
        serialized_index = {
            term: [posting.to_dict() for posting in postings] for term, postings in index.items()
        }

        write_to_json_file(serialized_index,
                           os.path.join(self.INDEXES_DIRECTORY, self.PARTIAL_INDEX_FILE.replace("#", str(off_count))))

    def merge_partial_indexes(self, off_count):
        """Merges all partial indexes into one final index file and dictionary"""
        # Initialize final mapping
        final_index = {}

        # Will go through each partial index made and collect the information needed for the final index
        for i in range(off_count):
            filename = os.path.join(self.INDEXES_DIRECTORY, self.PARTIAL_INDEX_FILE.replace("#", str(i)))
            try:
                with open(filename, 'r') as file:
                    partial_index = json.load(file)
                    for token, postings in partial_index.items():
                        if token in final_index:
                            final_index[token].extend(postings)
                        else:
                            final_index[token] = postings
                os.remove(filename)
            except FileNotFoundError:
                print("Error occurred while merging partial indexes: File " + filename + " not found")
            except OSError:
                pass
            except Exception as e:
                print("Error occurred while merging partial indexes: " + str(e))

        # Dict to keep track of terms and their positions in the index file (for use with seek later)
        token_offsets = {}
        total_docs = len(self.doc_lengths)  # used to calculate idf
        # This will be used to calculate each document's vector magnitude when performing the search queries
        doc_sum_of_squares = [0.0] * total_docs  # initialize sum of squares for each doc to zero

        final_index_file = os.path.join(self.INDEXES_DIRECTORY, self.FINAL_INDEX_FILE)
        try:
            with open(final_index_file, 'wb') as file:
                # iterate through the dictionary in alphabetical order
                for token, postings in sorted(final_index.items(), key=lambda kv_pair: kv_pair[0]):
                    doc_freq = len(postings)  # the number of documents that contain this term/token
                    idf = math.log10(total_docs / doc_freq)  # calculate idf score for this token
                    # combine the token's idf score with the existing tf score for each posting
                    for posting in postings:
                        tf = posting['tfidf_score']  # get the existing tf score
                        tfidf_score = tf * idf  # calculate the raw tfidf score
                        posting['tfidf_score'] = round(tfidf_score, 2)  # round tfidf score to 2 decimal places
                        # Compute the sum of squares for this document
                        doc_id = int(posting['document_id'])
                        doc_len = self.doc_lengths[doc_id]
                        # square tfidf score and add to sum
                        doc_sum_of_squares[doc_id] += math.pow(tf / doc_len * idf, 2)
                        # print(f"term: {token}, tf: {tf}, idf: {idf}, tfidf_score: {tfidf_score}")

                    offset = file.tell()
                    token_offsets[token] = offset
                    # postings is a list of dicts
                    postings_str = ";".join([f"({p['document_id']},{p['tfidf_score']})" for p in postings])
                    # print("T,P: " + token + ":" + postings_str)
                    s = token + ":" + postings_str + "\n"
                    file.write(s.encode('utf-8'))  # write string as bytes to file
        except Exception as e:
            print(e)

        # Create the document magnitudes for each doc
        doc_magnitudes = {}
        for i, sum_of_square in enumerate(doc_sum_of_squares):
            doc_magnitudes[i] = math.sqrt(sum_of_square)

        write_to_json_file({i: length for i, length in enumerate(self.doc_lengths)},
                           os.path.join(self.HELPERS_DIRECTORY, self.DOCUMENT_LENGTHS_FILE))
        write_to_json_file(token_offsets,
                           os.path.join(self.HELPERS_DIRECTORY, self.TERM_OFFSETS_FILE))
        write_to_json_file(doc_magnitudes,
                           os.path.join(self.HELPERS_DIRECTORY, self.DOCUMENT_MAGNITUDES_FILE))

        return final_index


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = sys.argv[1]
    print("Getting documents...")
    docs = get_documents(folder_path)

    print("Building index...")
    indexer = Indexer()
    index_table = indexer.build_index(docs)
    print("Building index complete")

    print("The number of indexed documents:", len(indexer.doc_ids_to_urls))
    print("The number of unique words:", len(indexer.unique_words))
