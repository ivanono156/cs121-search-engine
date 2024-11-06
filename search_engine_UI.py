from search_engine import SearchEngine
import streamlit as st


class SearchEngineUI:
    def __init__(self):
        self.search_engine = SearchEngine()

    def run(self):
        st.title("Search Engine")

        query = st.text_input("Enter your search query", placeholder="Search query")

        results = self.search_engine.search_corpus(query)
        if results:
            table = {
                "#": [i + 1 for i in range(len(results))],
                "results": results
            }

            st.dataframe(
                table,
                column_config={
                    "#": st.column_config.NumberColumn(label="Top"),
                    "results": st.column_config.LinkColumn(label="Results")
                },
                use_container_width=True,
            )
        elif query:
            st.write("No results found")


if __name__ == '__main__':
    engine_ui = SearchEngineUI()
    engine_ui.run()
