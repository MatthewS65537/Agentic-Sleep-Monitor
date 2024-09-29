import pickle
import sys
from pathlib import Path

def load_eval_data(file_name):
    try:
        file_path = Path(__file__).parent.parent / "results" / f"EVAL_{file_name}.pkl"
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File {file_name} not found.")
        return None

def display_report(eval_data, index):
    if eval_data is None or index >= len(eval_data["detailed"]):
        print("Invalid index or no data available.")
        return

    report = eval_data["detailed"][index]
    print(f"Report for index {index}:")
    print(f"Response: {report['response']}")
    print("Ratings:")
    for metric, score in report['ratings'].items():
        print(f"  {metric.capitalize()}: {score}")

def main():
    file_name = input("Enter the pickle file name: ")
    try:
        index = int(input("Enter the index: "))
    except ValueError:
        print("Error: Index must be an integer.")
        sys.exit(1)

    eval_data = load_eval_data(file_name)
    if eval_data:
        display_report(eval_data, index)

if __name__ == "__main__":
    main()
