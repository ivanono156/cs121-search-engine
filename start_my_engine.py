import json
import math
import queue
import re
import time
from nltk import PorterStemmer


def run():
    while True:
        search_query = input("Enter search query (enter 'q!' to quit): ").strip()   # Remove whitespace
        if search_query == "q!":
            print("Goodbye")
            break

        search_corpus(search_query)


def parse_query(search_query: str) -> list[str]:
    stemmer = PorterStemmer()
    # Remove punctuation, keep only words
    parsed = re.findall(r'\w+', search_query)
    # print("Parsed search query: " + str(parsed))
    stemmed = [stemmer.stem(token) for token in parsed]
    # print("Stemmed tokens: " + str(stemmed))
    return stemmed


def search_corpus(search_terms: str) -> None:
    start_time = time.time()

    parsed_query = parse_query(search_terms)
    # # Filter and score documents
    # results = document_at_a_time_retrieval(parsed_query, tf_scoring, filler_scoring, 5)
    results = cosine_scoring(parsed_query, 5)
    links = retrieve_links(results)

    end_time = time.time()
    print(f"Search query took {(end_time - start_time) * 1000} milliseconds")

    print(f"Top {len(links)} Results:")
    for link in links:
        print(link)


# search terms = stemmed terms from the search query
# returns a mapping of each search term to its inverted list
def get_inverted_lists(search_terms: list[str]) -> dict[str: list[(int, float)]]:
    inverted_lists = {}
    try:
        with (open('final_index.txt', 'rb') as index_file,
              open('term_offsets.json', 'r', encoding='utf8') as offsets_file):
            term_offsets = json.load(offsets_file)

            for term in search_terms:
                # Skip over terms that are not in the corpus
                if term not in term_offsets:
                    inverted_lists[term] = []
                    continue
                offset = term_offsets[term]
                index_file.seek(offset)
                line = index_file.readline().decode()   # decode bytes
                # print(line)
                postings = re.findall(r'(\d+,\d+\.\d+)', line)
                # print("postings for " + term + ": " + str(postings))
                inverted_lists[term] = []
                for posting in postings:
                    doc_id, term_freq = posting.split(",")
                    inverted_lists[term].append((int(doc_id), float(term_freq)))
                # print(f"{term}: {inverted_lists[term]}")
    except FileNotFoundError:
        print("Index file not found! Create the index file before searching")
    except json.JSONDecodeError:
        print("Error occurred while decoding json file")
    return inverted_lists


# Computes the cosine similarity between the given search query and the documents in the corpus
def cosine_scoring(query: list[str], k: int) -> list[int]:
    results = queue.PriorityQueue()
    inverted_lists = get_inverted_lists(query)

    doc_magnitudes = get_document_magnitudes()
    doc_lengths = get_document_lengths()
    total_docs = len(doc_magnitudes)
    query_tfidfs = compute_query_tfidfs(query, inverted_lists, total_docs)
    query_magnitude = math.sqrt(sum(math.pow(tfidf, 2) for tfidf in query_tfidfs.values()))
    # print("Query magnitude:", query_magnitude)

    scores = [0.0] * total_docs  # Initialize list of scores for all documents
    # Compute dot product of tfidf score of term in each document and the query
    for term in query:
        for doc_id, tfidf_score in inverted_lists[term]:
            # print(term, doc_id, tfidf_score)
            query_tfidf = query_tfidfs[term]
            doc_length = doc_lengths[str(doc_id)]
            # Normalize the term document tfidf score by dividing it by the document's length
            scores[doc_id] += (tfidf_score / doc_length) * query_tfidf

    # Normalize scores by dividing by the doc and query magnitudes to get the cosine similarity
    for i in range(total_docs):
        doc_magnitude = doc_magnitudes[str(i)]
        denominator = doc_magnitude * query_magnitude
        doc_score = scores[i] / denominator if denominator > 0 else 0
        if doc_score > 0:  # Only consider documents with a non-zero score
            # print(f"Score for doc #{i}: {doc_score}")
            results.put((-doc_score, i))  # Negative score for max-heap behavior

    # Retrieve the top k documents
    top_documents = []
    while not results.empty() and k > 0:
        doc_score, doc = results.get()
        # print(f"Score for doc #{doc}: {abs(doc_score)}")
        top_documents.append(doc)
        k -= 1
    return top_documents


def compute_query_tfidfs(query: list[str], inverted_lists: dict[str: list[str]], total_docs: int) -> dict[str, float]:
    # First, compute TF for each term
    term_freqs = {}
    for term in query:
        if term not in term_freqs:
            term_freqs[term] = 1
        else:
            term_freqs[term] += 1
    # Next, compute the tfidf score for each term in the query
    term_tfidfs = {}
    for term, tf in term_freqs.items():
        doc_freq = len(inverted_lists[term]) if term in inverted_lists else 0
        idf = math.log10(total_docs / doc_freq) if doc_freq > 0 else 0
        tfidf = tf * idf
        # print(f"tfidf for term {term}: {tfidf}")
        term_tfidfs[term] = tfidf
    return term_tfidfs


def document_at_a_time_retrieval(query, f, g, k):
    results = queue.PriorityQueue()
    inverted_lists = get_inverted_lists(query)

    # Initialize dictionary to keep track of how many terms a doc has in common with search
    doc_terms_count = {}

    # Each term in query inverted list is searched
    for term in query:
        if term not in inverted_lists:
            continue
        for posting in inverted_lists[term]:
            doc_id, tfidf_score = posting
            # print(f"Doc #{doc_id}: {tfidf_score}")
            if doc_id not in doc_terms_count:
                doc_terms_count[doc_id] = set()
            doc_terms_count[doc_id].add(term)

    # Creates filtered set of docs so only docs with all common terms are searched and scored later
    filtered_docs = set(doc_id for doc_id, terms in doc_terms_count.items() if len(terms) == len(query))

    # Scored the filtered set of documents
    for doc in filtered_docs:
        doc_score = 0
        for term in query:
            if term in inverted_lists:
                postings = inverted_lists[term]
                for posting in postings:
                    doc_id, tfidf_score = posting
                    if doc_id == doc:
                        doc_score += g(query, term) * f(tfidf_score)
        if doc_score > 0:  # Only consider documents with a non-zero score
            results.put((-doc_score, doc))  # Negative score for max-heap behavior
    top_documents = []
    while not results.empty() and k > 0:
        doc_score, doc = results.get()
        # print(doc, doc_score)
        top_documents.append(doc)
        k -= 1
    return top_documents


def tf_scoring(term_freq):
    return term_freq


def filler_scoring(query, term):
    return 1


def get_document_magnitudes() -> dict[str, float]:
    try:
        with open('document_magnitudes.json', 'r') as file:
            doc_magnitudes = json.load(file)
            return doc_magnitudes
    except FileNotFoundError:
        print("Document magnitudes file not found! Create index before searching")
    except Exception as e:
        print("Error occurred while reading document magnitudes: " + str(e))
    return {}


def get_document_lengths() -> dict[str, float]:
    try:
        with open('document_lengths.json', 'r') as file:
            doc_lengths = json.load(file)
            return doc_lengths
    except FileNotFoundError:
        print("Document lengths file not found! Create index before searching")
    except Exception as e:
        print("Error occurred while reading document lengths: " + str(e))
    return {}


def retrieve_links(doc_ids) -> list[str]:
    # Opens document where we store doc_id -> urls
    try:
        with open('document_ids_to_urls.json', 'r', encoding='utf8') as links_file:
            doc_links = json.load(links_file)
            # Gets the link that are linked to each doc id passed in
            links = [doc_links.get(str(doc_id), "Link not found") for doc_id in doc_ids]
            return links
    except FileNotFoundError:
        print("Links file not found! Create the links file before searching")
    except json.JSONDecodeError:
        print("Error occurred while decoding JSON file")
    return []


if __name__ == "__main__":
    run()
