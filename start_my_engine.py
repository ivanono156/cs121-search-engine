import json
import queue
import re
import time
from nltk import PorterStemmer

def run():
    search_query = input("Enter search query: ")
    # start_time = time.time()
    parsed_query = parse_query(search_query)
    search_corpus(parsed_query)
    # end_time = time.time()    TODO: figure out where the best place to time this goes
    # print(f"Search query took {(end_time-start_time) * 1000} milliseconds")

def parse_query(search_query) -> list[str]:
    stemmer = PorterStemmer()
    # Remove punctuation, keep only words
    parsed = re.findall(r'\w+', search_query)
    print("Parsed search query: " + str(parsed))
    stemmed = [stemmer.stem(token) for token in parsed]
    print("Stemmed tokens: " + str(stemmed))
    return stemmed

def search_corpus(search_terms):
    inverted_lists = get_inverted_lists(search_terms)
    for term, inv_list in inverted_lists.items():
        print(f"Inverted list for {term}: {inverted_lists[term]}")
    #print(document_at_a_time_retrieval(search_terms, [1,2,3,4,5],1,2,5))

# search terms = stemmed terms from the search query
# returns a mapping of each search term to its inverted list
# If a
def get_inverted_lists(search_terms) -> dict[str: list[str]]:
    inverted_lists = {}
    try:
        # TODO: Change final_index file to use bytes instead of text
        with (open('final_index.txt', 'r', encoding='utf8') as index_file,
              open('term_offsets.json', 'r', encoding='utf8') as offsets_file):
            term_offsets = json.load(offsets_file)
            for term in search_terms:
                # Skip over terms that are not in the corpus
                if term not in term_offsets:
                    inverted_lists[term] = []   # just for debugging, but we can remove this in the final submission
                    continue
                offset = term_offsets[term]
                index_file.seek(offset)
                line = index_file.readline()
                postings = re.findall(r'(\d+,\d+)', line)
                # print("postings for " + term + ": " + str(postings))
                inverted_lists[term] = postings
    except FileNotFoundError:
        print("Index file not found! Create the index file before searching")
    except json.JSONDecodeError:
        print("Error occurred while decoding json file")
    return inverted_lists

# More skeleton code from lecture 19
def document_at_a_time_retrieval(query, index, f, g, k):
    results = queue.PriorityQueue()
    inverted_lists = get_inverted_lists(query)
    for doc in index:
        doc_score = 0
        # for li in inverted_lists:
        #     if li.get_current_document() == doc:
        #         doc_score = doc_score + g[i](query)*f[i](li)
        #     li.move_past_document(doc)

        # Add items to the priority queue as a tuple in the form: (priority_number, data)
        # Priority queue is order from least to greatest, so smaller priority_number = start of queue
        results.put((doc_score, doc))

    top_documents = []
    while not results.empty() and k > 0:
        top_documents.append(results.get())
        k -= 1
    return top_documents

# def document_at_a_time_retrieval(query, index, f, g, k):
#     inverted_lists = []
#     results = queue.PriorityQueue()
#     for term in query:
#         li = get_inverted_list(term, index)
#         inverted_lists.append(li)
#     for doc in index:
#         doc_score = 0
#         for li in inverted_lists:
#             if li.get_current_document() == doc:
#                 doc_score = doc_score + g[i](query)*f[i](li)
#             li.move_past_document(doc)
#         results.put((doc_score, doc))
#
#     res = []
#     while not results.empty() and k > 0:
#         res.append(results.get())
#         k -= 1
#     return res

if __name__ == "__main__":
    run()
