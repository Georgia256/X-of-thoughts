import os

def rename_files(directory):
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if "v2_" in filename:
            # Generate new file name by replacing 'v2_' with ''
            new_filename = filename.replace("v2_", "")
            # Generate full file paths
            old_file_path = os.path.join(directory, filename)
            new_file_path = os.path.join(directory, new_filename)
            # Rename the file
            os.rename(old_file_path, new_file_path)
            print(f'Renamed "{filename}" to "{new_filename}"')

# Specify the directory path
directory_path = '/Users/olga/Desktop/Pattern Recognition/Project/XoT-Pattern_Recognition_Project/outputs/gsm/metacognitive_eval_cot'
rename_files(directory_path)
