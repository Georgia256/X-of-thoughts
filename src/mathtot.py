import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import jsonlines

# Define your function for loading the dataset
def myload_dataset(data_path):
    instances = []
    with open(data_path, "r+", encoding="utf8") as f:
        for inst in jsonlines.Reader(f):
            instances.append(inst)

    print(f"Load {len(instances)} data from {data_path}.")
    return instances

# Define your function for generating completions using phi-2 model
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def phi2_completion(prompt, temperature, k=1, stop=None):
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2", torch_dtype="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/phi-2", trust_remote_code=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    n_examples = len(prompt.split("<END>")) - 1

    max_len = math.ceil(input_ids.shape[1] * (1 + 1 / (n_examples - 1)))

    # Generate completion
    outputs = model.generate(
        input_ids=input_ids,
        max_length=max_len,
        temperature=temperature,
        num_return_sequences=k,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        **({"do_sample": True, "top_k": 50, "top_p": 0.95} if stop is None else {"early_stopping": True, "max_length": stop})
    )

    # Decode the generated sequences
    completions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return completions

# Modify your code to handle the phi-2 completions
def openai_phi2_handler(prompt, temperature, k=1, stop=None):
    completions = phi2_completion(prompt, temperature, k, stop)
    return completions

def generate_text_phi(prompt, k):
    response = openai_phi2_handler(prompt, 0.9, k)
    thoughts = [response]  # Assuming response is already a list of completions
    return thoughts

# Define the ranking function
def ranking(prompt, question, past):
    comparison_prompt = f"""
    To achieve the following goal: '{question}', and based on the current steps taken towards solving the problem {past}
    pessimistically value the below mentioned step and choose one of the following options that will be the best option towards the goal.
    Return the exact same chosen option, don't change or format it.
    The options to choose from \n
    {prompt}\n

    NOTE:
    1) Evaluate all the options and choose the option which is the best direction for the next step to move based on the past solution we have found till now. Don't choose the output that jumps to the result directly.
    2) MAKE SURE YOU DON'T CHOOSE THE OPTION THAT HAS A SIMILAR MEANING (STEP) TO WHAT IS ALREADY THERE IN THE PAST SOLUTION ARRAY.

    DO NOT RETURN ANYTHING ELSE JUST THE OPTION THAT IS THE BEST NEXT STEP, NO EXPLANATION FOR THE CHOICE
    """
    a = generate_text_phi(comparison_prompt, 1)
    return a

# Define your main function
def main():
    # Parameters
    max_steps = 3
    k = 1
    status = ["None"]
    question = "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"

    for i in range(max_steps):
        print("*****************NEW STEP*****************")
        print(f"The status array is {status}")
        initial_prompt = f"""
        Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next step to solve the problem involving a single arithmetic option. If there are multiple options for how to proceed, you should generate up to 3 options.

        The format of the problem is as below, follow this format only
        Input: XXXX
        Steps taken so far: {status}
        Output: ZZZZ

        NOTE: The options should not be sequential or connected with each other, each option should be in a way that it can be evaluated independently. Don't jump to the result directly.
        IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
        """
        out = ranking(initial_prompt, question, status)

        # Update status with the chosen option
        if "None" in status:
            status = [out]
        else:
            status.append(out)

        print(f"The option chosen as the best choice is {out}")
        print("\n\n\n")

if __name__ == "__main__":
    main()
