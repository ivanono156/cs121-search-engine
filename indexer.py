from posting import Posting


# Skeleton code from lecture slides
def build_index(documents):
    # Create hashtable
    # Mapping = token: posting (document id)
    hashtable = {}
    # Use enumerate to map each doc to an id (n)
    for n, document in enumerate(documents):
        # T <- Parse documents
        # Remove duplicates from T
        tokens = set(tokenize(document))
        # Add each token to the hashtable
        for token in tokens:
            # Initialize new tokens
            if token not in hashtable:
                hashtable[token] = []
            # Map each token to its posting (which contains this document's id)
            hashtable[token].append(Posting(n, 0))
    return hashtable


# Code for tokenizing when we have file path instead
def tokenize(TextFilePath): 
    token_list = []
    # Opens file to be read byte by byte also catches error if file path is incorrect or wrong type
    try:
        with open(TextFilePath, encoding = 'utf-8') as file:
            byte = file.read(1) # Reads 1 byte, the first in this case
            current_token = ""
            while byte: # Go throughs byte by byte of the file until end is reached
                if(byte.isalnum() and byte.isascii()):
                    # Checks if the char is valid to be in token and if not, will restart token string to be added to list
                    current_token += byte
                else:
                    if len(current_token) != 0:
                        token_list.append(current_token.lower())
                    current_token = ""
                byte = file.read(1)
        if len(current_token) != 0:
            token_list.append(current_token.lower())        
    except FileNotFoundError:
        print("File path incorrect or incorrect type")
        pass
    
    return token_list


# Code to tokenize a document when we have it in string form
def _tokenized(text_string):
    token_list = []
    current_token = ""
    while (text_string):
        char = text_string[0] # Reads first char
        # Currently only breaking up tokens by if the current char is a letter used in any alphabet, not only english
        # Easily changed by altering the if statement below
        if char.isalpha():
            # Checks if the char is valid to be in token and if not, will restart token string to be added to set
            current_token += char
        else:
            if len(current_token) != 0:
                token_list.append(current_token.lower())
            current_token = ""
        text_string[1:]
        
    if len(current_token) != 0:
        token_list.append(current_token.lower())
        
    return token_list


def _computeWordWeights(tokens_list):
    # Initializing dictionary/map of the tokens and the number of occurrences
    word_freq = {}
    # Iterates through all the values in the token list and updates its value in the corresponding dictionary/mapping
    for token in tokens_list:
        # Checks the value of the token in the dictionary and sets it to the appropriate value either 1 or +1
        word_freq[token] = word_freq.get(token,0)  + 1
