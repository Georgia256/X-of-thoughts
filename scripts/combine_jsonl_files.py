import os
import json
import re


directory = "/Users/olga/Desktop/Pattern Recognition/Project/XoT-Pattern_Recognition_Project/outputs/gsm/plan"
output_file = "/Users/olga/Desktop/Pattern Recognition/Project/XoT-Pattern_Recognition_Project/outputs/gsm/plan/phi_plan_0_end.jsonl"


# Function to extract number from filename
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


# Collect all relevant files
jsonl_files = [
    f for f in os.listdir(directory) if f.startswith("batch") and f.endswith(".jsonl")
]

# Sort files based on the extracted number
jsonl_files.sort(key=extract_number)

# Write sorted files to output
with open(output_file, "w") as outfile:
    for filename in jsonl_files:
        with open(os.path.join(directory, filename), "r") as infile:
            for line in infile:
                outfile.write(line)
