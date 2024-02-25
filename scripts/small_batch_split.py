import json


def split_jsonl_file(file_path, batch_size=10, offset=1120, method="tot"):
    """
    Split a JSONL file into multiple files, each containing a specified number of lines.

    :param file_path: Path to the JSONL file.
    :param batch_size: Number of lines in each batch file.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    prefix = "/Users/olga/Desktop/Pattern Recognition/Project/X-of-thoughts/outputs/gsm/tot/minibatches/"
    total_batches = len(lines) // batch_size + (1 if len(lines) % batch_size else 0)
    for i in range(total_batches):
        start_idx = i * batch_size  #+ offset
        end_idx = (i + 1) * batch_size #+ offset
        batch_lines = lines[start_idx:end_idx]
        name_idx = (i + 1) + offset/batch_size
        name_idx = int(name_idx)
        start_idx_name = start_idx + offset
        end_idx_name = end_idx + offset
        batch_file_name = (
            f"{prefix}minibatch_{name_idx}_{method}_{start_idx_name}_{end_idx_name}.jsonl"
        )

        with open(batch_file_name, 'w') as batch_file:
            for i, line in enumerate(batch_lines):
                if i == len(batch_lines) - 1:
                    # Remove newline character from the last line
                    line = line.rstrip('\n')
                batch_file.write(line)



# Replace 'your_file.jsonl' with the path to your JSONL file
split_jsonl_file(
    "/Users/olga/Desktop/Pattern Recognition/Project/X-of-thoughts/outputs/gsm/tot/phi2_tot_1120_1210.jsonl"
)
