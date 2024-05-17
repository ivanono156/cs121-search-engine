import json
import queue
import re
import time
from nltk import PorterStemmer

def run():
    search_query = input("Enter search query: ")
    start_time = time.time()
    parsed_query = parse_query(search_query)
    search_corpus(parsed_query)
    end_time = time.time()
    
    print(f"Search query took {(end_time-start_time) * 1000} milliseconds")

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
    
 

    # Filter and score documents
    results = document_at_a_time_retrieval(search_terms, tf_scoring, filler_scoring, 5)
    links = retrieve_links(results)
    for link in links:
        print(link)
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

def document_at_a_time_retrieval(query, f, g, k):
    results = queue.PriorityQueue()
    inverted_lists = get_inverted_lists(query)
    
    # Initalize dictionary to keep track of how many terms a doc has in common with search
    doc_terms_count = {}
    
    # Each term in quert inverted list is searched
    for term in query:
        for posting in inverted_lists[term]:
            doc_id, term_freq = map(int, posting.split(','))
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
                    doc_id, term_freq = map(int, posting.split(','))
                    if doc_id == doc:
                        doc_score += g(query, term) * f(term_freq)
        if doc_score > 0:  # Only consider documents with a non-zero score
            results.put((-doc_score, doc))  # Negative score for max-heap behavior
    top_documents = []
    while not results.empty() and k > 0:
        top_documents.append(results.get()[1])
        k -= 1
    return top_documents
# # More skeleton code from lecture 19
# def document_at_a_time_retrieval(query, index, f, g, k):
#     results = queue.PriorityQueue()
#     inverted_lists = get_inverted_lists(query)
#     for doc in index:
#         doc_score = 0
#         # for li in inverted_lists:
#         #     if li.get_current_document() == doc:
#         #         doc_score = doc_score + g[i](query)*f[i](li)
#         #     li.move_past_document(doc)

#         # Add items to the priority queue as a tuple in the form: (priority_number, data)
#         # Priority queue is order from least to greatest, so smaller priority_number = start of queue
#         results.put((doc_score, doc))

#     top_documents = []
#     while not results.empty() and k > 0:
#         top_documents.append(results.get())
#         k -= 1
#     return top_documents


def tf_scoring(term_freq): # 
    return term_freq
def filler_scoring(query,term):
    return 1
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

def retrieve_links(doc_ids):
    # Opens document where we store doc_id -> urls
    try:
        with open('document_ids_to_urls.json', 'r', encoding='utf8') as  links_file:
            doc_links = json.load(links_file)
    except FileNotFoundError:
        print("Links file not found! Create the links file before searching")
        return []
    except json.JSONDecodeError:
        print("Error occurred while decoding JSON file")
        return []

    # Gets the link that are linked to each doc id passed in
    links = [doc_links.get(str(doc_id), "Link not found") for doc_id in doc_ids]
    return links
        
if __name__ == "__main__":
    run()
