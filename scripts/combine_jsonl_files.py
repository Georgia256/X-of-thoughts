import os
import json
import re
import sys


# if len(sys.argv) != 3:
#     print("Usage: script.py <input_folder> <output_file>")
#     sys.exit(1)

input_directory = "/Users/olga/Desktop/Pattern Recognition/Project/X-of-thoughts/outputs/gsm/tot"
output_file = "/Users/olga/Desktop/Pattern Recognition/Project/X-of-thoughts/outputs/gsm/tot/phi_tot_0_end.jsonl"


# Function to extract number from filename
def extract_number(filename):
    match = re.search(r"\d+", filename)
    return int(match.group()) if match else 0


# Collect all relevant files
jsonl_files = [
    f
    for f in os.listdir(input_directory)
    #if "rating" in f and f.endswith(".jsonl")
    if f.startswith("batch") and f.endswith(".jsonl")
]

# Sort files based on the extracted number
jsonl_files.sort(key=extract_number)

# Write sorted files to output
with open(output_file, "w") as outfile:
    for filename in jsonl_files:
        with open(os.path.join(input_directory, filename), "r") as infile:
            for line in infile:
                outfile.write(line)
