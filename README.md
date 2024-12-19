# CS 121 Assignment 3: Search Engine

#### Ivan Onofre

* Parses over 55000 files and builds inverted index containing the contents of each file
* Retrieves results using multiple algorithms, including HITS and cosine similarity
* Indexer creates helper files to improve performance and speed of search engine at run time

#### Project Description
This project contains code for an indexer and a search engine, which searches the items 
parsed by the indexer. The indexer parses each file and builds helper files to keep track of
useful information, such as document lengths, document magnitudes, and term offsets for the 
inverted index, to help with the search engine. The search engine retrieves the information by
using cosine similarity and HITS to return results relevant to the search query. The search 
engine can be run from the command line or by using a local GUI interface.

## Requirements:

```
pip install -r requirements.txt
```

### To create the index:
- Run the indexer.py file from the command line.
- Enter the path to the folder that holds subfolders (the domains)with files containing the page information. 
- In this example, the folders in the crawled_files.zip folder can be used

Ex: 

```
python3 indexer.py "home/user/path-to-folder/DEV"
```

### To run the search engine:

#### Option 1: Using start_my_engine.py

- Run the start_my_engine.py file from the command line.
- Enter 1 for the search engine without UI, or enter 2 for the search engine with UI.

To run the search engine without UI:

```
python3 start_my_engine.py 1
```

To run the search engine with UI:

```
python3 start_my_engine.py 2
```

#### Option 2: From the command line
- Run the search_engine.py file from the command line. This will run the search engine without UI.

```
python3 search_engine.py
```

- From there you will be prompted to input a search query.

#### Option 3: With the Streamlit User Interface
- Run the search_engine_UI.py file from the command line using streamlit.

```
streamlit run search_engine_UI.py
```
