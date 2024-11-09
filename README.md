# CS 121 Assignment 3: Search Engine

## Requirements:

    pip install -r requirements.txt

### To create the index:
- Run the indexer.py file from the command line.
- Enter the path to the folder that holds subfolders (the domains)with files containing the page information. 

Ex: 

    python3 indexer.py "home/user/path-to-folder/DEV"

### To run the search engine:

#### Option 1: Using start_my_engine.py

- Run the start_my_engine.py file from the command line.
- Enter 1 for the search engine without UI, or enter 2 for the search engine with UI.

To run the search engine without UI:

    python3 start_my_engine.py 1

To run the search engine with UI:

    python3 start_my_engine.py 2

#### Option 2: From the command line
- Run the search_engine.py file from the command line. This will run the search engine without UI.


    python3 search_engine.py
- From there you will be prompted to input a search query.

#### Option 3: With the Streamlit User Interface
- Run the search_engine_UI.py file from the command line using streamlit.


    streamlit run search_engine_UI.py
