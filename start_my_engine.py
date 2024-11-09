import subprocess
import sys

if __name__ == "__main__":
    choice = sys.argv[1]
    if choice == "1":
        subprocess.run(["python3", "search_engine.py"])
    elif choice == "2":
        subprocess.run(["streamlit", "run", "search_engine_UI.py"])
    else:
        print("Invalid option")
