# Let's open the provided JSONL file and check for lines where the key 'metacognitive_eval/cot/ans' is missing.
# We will collect the IDs of such lines.
import json 
file_path = '/Users/olga/Desktop/Pattern Recognition/Project/X-of-thoughts/outputs/gsm/metacognitive_eval_tot/deepseek_metacognitive_eval_tot_0_end.jsonl'
missing_keys_ids = []

with open(file_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        # Check if the key 'metacognitive_eval/cot/ans' is missing
        if 'metacognitive_eval/tot/ans' not in data:
            # If missing, collect the 'id' of the line
            missing_keys_ids.append(data.get('id', 'No ID Found'))

print((missing_keys_ids))
