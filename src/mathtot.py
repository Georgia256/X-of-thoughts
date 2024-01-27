import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff'''
#import datasets

import re
import time
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import jsonlines
import json
from utils import *

import datasets
from datasets import load_dataset

KEY_GROUP = {"a": ["YOUR_API_KEY"]}

api_model="phi-2"

from IPython.core.inputtransformer2 import ESC_HELP
#from openai.error import Error  # Add this line to import the Error class

data_path = "data/gsm8k/test.jsonl"

def myload_dataset(data_path):
    instances = []
    with open(data_path, "r+", encoding="utf8") as f:
        for inst in jsonlines.Reader(f):
            instances.append(inst)

    print(f"Load {len(instances)} data from {data_path}.")
    return instances

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

# Modified function to handle phi-2 completions
def openai_phi2_handler(prompt, temperature, k=1, stop=None):
    completions = phi2_completion(prompt, temperature, k, stop)
    return completions

def openai_choice2text_handler(completion):
    text = completion.strip()
    return text

def generate_text_phi(prompt, k):
    response = openai_phi2_handler(prompt, 0.9, k)
    thoughts = [openai_choice2text_handler(completion) for completion in response]
    return thoughts



def ranking(prompt,question,past):
  # ranks = []
  # for i in range(len(prompt)):
  comparison_prompt = f"""
  To achieve the following goal: '{question}', and based on the current steps taken towards solving the problem {past}
  pessimistically value the below mentioned step and choose one of the follwing options that will be the best option towards the goal.
  Return the exact same chosen option, dont change or format it.
  The options to choose from \n
  {prompt}\n

  NOTE:
  1) Evaluate all the options and choose the option which is the best direction for the next step to move based on the past solution we have found till now. Dont choose the output that jumps to the result directly.
  2)MAKE SURE YOU DONT CHOOSE THE OPTION THAT HAS A SIMILAR MEANING (STEP) TO WHAT IS ALREADY THERE IN THE PAST SOLUTION ARRAY.

  DO NOT RETURN ANYTHING ELSE JUST THE OPTION THAT IS THE BEST NEXT STEP, NO EXPLANATION FOR THE CHOICE
  """
  a = generate_text_phi(comparison_prompt,1)
  return a

def parse_output_options(output):
  # output = output.split("Output")[1:]
  # output = " ".join(output).strip()
  output = output.split("\n")
  return output

"""# Single phi Instance with multiple thoughts"""

initial_promp_temp = f"""
Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next step to solve the problem involving a single arithmetic option. If there are multiple options for how to proceed, you should generate up to 3 options.

The format of the problem is as below, follow this format only
Input: XXXX
Steps taken so far: YYYY
Output: ZZZZ

NOTE: The options should not be sequential or connected with each other, each option should be in a way that it can be evaluated independently. Dont jump to the result directly.
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
Solved Example:

Example 1
Input: "Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?"

Steps take so far: [Calculate the price of cheddar cheese which is $10 (given)]


Output: Possible independent steps:
1) Calculate the price of cold cuts which is 2*10 = $20.
2)Calculate the price of cream cheese which is 10/2 = $5 per pound.

Example 2
Input: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

Steps taken so far: [None]

Output: Possible next steps:
1) Convert the minutes of babysitting to hours.
2) Convert the wage per hour to wage per minute.

Example 3
Input: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

Steps taken so far: [Number of letter written to 1 friend in a week = 2 as he writes twice a week]

Output: Possible next steps:
1) Number of letter written to 2 friends in a week = 2*2 = 4 letters a week.
2) Calculate the number of pages written to 1 friend in a week = 2*3 = 6 pages.


Now give the possible steps for the below question
Input: "Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"

Steps taken so far:

"""

output_string = " \n Output: Possible independent steps:"


question = """Albert is wondering how much pizza he can eat in one day. He buys 2 large pizzas and 2 small pizzas. A large pizza has 16 slices and a small pizza has 8 slices. If he eats it all, how many pieces does he eat that day?"""

#Parameters

max_steps = 3
k=1
status = ["None"]

for i in range(max_steps):
  print("*****************NEW STEP*****************")
  print(f"The status array is {status}")
  initial_promp = initial_promp_temp + str(status) + output_string
  out = generate_text_phi(initial_promp,k)[0]
  # print(f"The output from the GPT is {out}")
  outputs = parse_output_options(out)
  print(f"The parsed output is {outputs}")
  option = ranking(outputs,question,status)

# Call reason_tot to get the reason for the chosen option
  #reason = reason_tot(initial_prompt, option)

# Display the reason
  #print(f"The reason for the chosen option is:\n{reason}")


  if("None") in status:
    status = [option]
  else:
    status.append(option)
  print(f"The option chosen as the best choice is {option}")
  print("\n\n\n")

