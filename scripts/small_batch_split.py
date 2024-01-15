import json


def split_jsonl_file(file_path, batch_size=10, method="cot"):
    """
    Split a JSONL file into multiple files, each containing a specified number of lines.

    :param file_path: Path to the JSONL file.
    :param batch_size: Number of lines in each batch file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    prefix = "/Users/olga/Desktop/Pattern Recognition/Project/XoT-Pattern_Recognition_Project/outputs/gsm/cot/minibatches/"
    total_batches = len(lines) // batch_size + (1 if len(lines) % batch_size else 0)
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        batch_lines = lines[start_idx:end_idx]
        batch_file_name = (
            f"{prefix}minibatch_{i+1}_{method}_{start_idx}_{end_idx}.jsonl"
        )

        with open(batch_file_name, 'w') as batch_file:
            for i, line in enumerate(batch_lines):
                if i == len(batch_lines) - 1:
                    # Remove newline character from the last line
                    line = line.rstrip('\n')
                batch_file.write(line)



# Replace 'your_file.jsonl' with the path to your JSONL file
split_jsonl_file(
    "/Users/olga/Desktop/Pattern Recognition/Project/XoT-Pattern_Recognition_Project/outputs/gsm/cot/phi_cot_0_end.jsonl"
)
