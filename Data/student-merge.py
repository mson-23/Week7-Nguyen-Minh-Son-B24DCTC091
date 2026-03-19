import os
import pandas as pd

# Optional: Change working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Read the datasets
d1 = pd.read_csv("student-mat.csv", sep=";")
d2 = pd.read_csv("student-por.csv", sep=";")

# Merge the datasets based on the specified columns
merge_columns = [
    "school", "sex", "age", "address", "famsize", "Pstatus",
    "Medu", "Fedu", "Mjob", "Fjob", "reason", "nursery", "internet"
]
d3 = pd.merge(d1, d2, on=merge_columns)

# Print the number of rows in the merged dataset
print(len(d1))
print(len(d2))
print(len(d3))
print(d3.columns)

d3.to_csv("student-merge.csv")