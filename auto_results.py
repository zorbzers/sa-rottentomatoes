"""
A script to run NB_sentiment_analyser.py with all 4 different configurations.
"""
import subprocess

"""
List of runs to execute.
    (1) 3 classes, using all words as features.
    (2) 5 classes, using all words as features.
    (3) 3 classes, using features given by the selected method in feature_selection.py.
    (4) 5 classes, using features given by the selected method in feature_selection.py.
"""
runs = [
    ["moviereviews\\train.tsv", "moviereviews\\dev.tsv", "moviereviews\\test.tsv", "-classes", "3", "-features", "features", "-output_files", "-confusion_matrix"],
    ["moviereviews\\train.tsv", "moviereviews\\dev.tsv", "moviereviews\\test.tsv", "-classes", "5", "-features", "features", "-output_files", "-confusion_matrix"],
    ["moviereviews\\train.tsv", "moviereviews\\dev.tsv", "moviereviews\\test.tsv", "-classes", "3", "-features", "all_words", "-output_files", "-confusion_matrix"],
    ["moviereviews\\train.tsv", "moviereviews\\dev.tsv", "moviereviews\\test.tsv", "-classes", "5", "-features", "all_words", "-output_files", "-confusion_matrix"]
]

# Loop through the runs and execute the program
for i, args in enumerate(runs, start=1):
    print(f"Running program for configuration {i}")
    subprocess.run(["python", "NB_sentiment_analyser.py"] + args)

print("All runs completed.")