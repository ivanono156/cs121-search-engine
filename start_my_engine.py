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


def parse_query(search_query) -> list[str]:
    stemmer = PorterStemmer()
    # Remove punctuation, keep only words
    parsed = re.findall(r'\w+', search_query)
    # print("Parsed search query: " + str(parsed))
    stemmed = [stemmer.stem(token) for token in parsed]
    # print("Stemmed tokens: " + str(stemmed))
    return stemmed


def search_corpus(search_terms):
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
def get_inverted_lists(search_terms) -> dict[str: list[str]]:
    inverted_lists = {}
    try:
        # TODO: Change final_index file to use bytes instead of text???
        with (open('final_index.txt', 'rb') as index_file,
              open('term_offsets.json', 'r', encoding='utf8') as offsets_file):
            term_offsets = json.load(offsets_file)

            for term in search_terms:
                # Skip over terms that are not in the corpus
                if term not in term_offsets:
                    inverted_lists[term] = []  # just for debugging, but we can remove this in the final submission
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


def cosine_scoring(query: list[str], k: int) -> list[int]:
    results = queue.PriorityQueue()
    inverted_lists = get_inverted_lists(query)

    # Compute cosine similarity scores
    doc_magnitudes = get_document_magnitudes()
    n = len(doc_magnitudes)
    query_magnitude = compute_query_magnitude(query, inverted_lists, n)
    print("Query magnitude:", query_magnitude)

    scores = [0] * n  # initialize list of scores

    for term in query:
        for doc_id, tfidf_score in inverted_lists[term]:
            # print(term, doc_id, tfidf_score)
            scores[doc_id] += tfidf_score

    for i in range(n):
        doc_magnitude = doc_magnitudes[str(i)]
        doc_score = scores[i] / doc_magnitude if doc_magnitude > 0 else 0  # normalize scores
        if doc_score > 0:  # Only consider documents with a non-zero score
            # print(f"Score for doc #{i}: {doc_score}")
            results.put((-doc_score, i))  # Negative score for max-heap behavior

    top_documents = []
    while not results.empty() and k > 0:
        doc_score, doc = results.get()
        print(f"Score for doc #{doc}: {doc_score}")
        top_documents.append(doc)
        k -= 1
    return top_documents


def compute_query_magnitude(query: list[str], inverted_lists: dict[str: list[str]], total_docs: int) -> float:
    # First, compute TF for each term
    term_freqs = {}
    for term in query:
        if term not in term_freqs:
            term_freqs[term] = 1
        else:
            term_freqs[term] += 1

    # Next, compute the tfidf for each term and compute the sum of the squares of each term weight
    sum_of_squares = 0.0
    for term in query:
        tf = term_freqs[term]
        doc_freq = len(inverted_lists[term]) if term in inverted_lists else 0
        idf = math.log10(total_docs / doc_freq) if doc_freq > 0 else 0
        tfidf = tf * idf
        print(f"tfidf for term {term}: {tfidf}")
        sum_of_squares += math.pow(tfidf, 2)   # square tfidf and add it to the sum
    print("Sum of squares:", sum_of_squares)

    return math.sqrt(sum_of_squares)   # return the query's magnitude


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
            # doc_id, term_freq = map(int, posting.split(','))
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
                    # doc_id, term_freq = map(int, posting.split(','))
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


def get_document_magnitudes():
    try:
        with open('document_magnitudes.json', 'r') as file:
            doc_magnitudes = json.load(file)
            return doc_magnitudes
    except FileNotFoundError:
        print("Document magnitudes file not found! Create index before searching")
    except Exception as e:
        print("Error occurred while reading document magnitudes: " + str(e))
    return {}


def retrieve_links(doc_ids):
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
