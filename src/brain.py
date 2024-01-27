import re
import os
import openai
import json
import time
import backoff
from time import sleep
import torch
import math
import jsonlines

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from tenacity import retry, stop_after_attempt, wait_random_exponential


from prompts.plan import PLAN_SYSTEM, PLAN
from prompts.cot_complex import COT_SYSTEM, COT
from prompts.pot import POT_SYSTEM, POT
from prompts.eot import EOT_SYSTEM, EOT
from prompts.peano import PEANO_SYSTEM, PEANO
from prompts.check import ASSERT_SYSTEM, ASSERT_PROMPT
from prompts.self_refine import REFINE_SYSTEM, REFINE
from prompts.metacognitive_eval_deepseek import META_EVAL_SYSTEM, META_EVAL

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from utils import *

KEY_GROUP = {"a": ["YOUR_API_KEY"]}

api_model="phi-2"

from IPython.core.inputtransformer2 import ESC_HELP
#from openai.error import Error  # Add this line to import the Error class

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


def load_dataset(data_path):
    instances = []
    with open(data_path, "r+", encoding="utf8") as f:
        for inst in jsonlines.Reader(f):
            instances.append(inst)

    print(f"Load {len(instances)} data from {data_path}.")
    return instances

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

    

class Brain:
    def __init__(self, args):
        self.args = args
        self.data = load_dataset(args.data_path)
        self.result_path = os.path.join(
            args.output_dir, f"{args.tag}_{args.range_start}_{args.range_end}.jsonl"
        )

        if os.path.exists(self.result_path) and not args.overwrite:
            raise ValueError(
                f"Result file {self.result_path} already exists. Please set --overwrite to True."
            )
        elif os.path.exists(self.result_path) and args.overwrite:
            print(f"Result file {self.result_path} already exists. Will overwrite.")
            os.remove(self.result_path)

        print(f"Will save result at {self.result_path}")
        print(
            f"Will save metric at {os.path.join(args.output_dir, f'{args.tag}_metric.json')}"
        )

        self.debug = args.debug
        if self.debug:
            print(f"===== Debug mode is on. =====")

        self.metrics = {
            "cot/correct": 0.0,
            "pot/correct": 0.0,
            "eot/correct": 0.0,
            "peano/correct": 0.0,
            "refine/correct": 0.0,
        }

        self.KEYS = KEY_GROUP[args.key_group]
        print(f"Using KEY GROUP <{args.key_group}>, len: {len(self.KEYS)}")
        self.ORG_ID = args.org_id if len(args.org_id) > 0 else None

    def set_instance(self, instance_id):
        self.id = instance_id
        self.instance = self.data[instance_id]
        self.cache = {
            "id": instance_id,
            "inst/question": self.instance["input"],
            "inst/gold_answer": float(self.instance["target"]),
        }
        self.api_key = self.KEYS[instance_id % len(self.KEYS)]

    def set_instance_check(self, instance_id):
        self.id = instance_id
        self.instance = self.data[instance_id - int(self.args.range_start)]
        self.cache = {k: v for k, v in self.instance.items()}
        self.api_key = self.KEYS[instance_id % len(self.KEYS)]

    def plan(self):
        """
        Plan the reasoning method.
        """
        question = self.cache["inst/question"]
        chat_input = self.build_chat_input(PLAN_SYSTEM, PLAN.format(question=question))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache["plan"] = response

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")

    def reason_cot(self):
        """
        Reasoning with Chain-of-Thought, prompts from COT
        """
        question = self.cache["inst/question"]
        chat_input = self.build_chat_input(COT_SYSTEM, COT.format(question=question))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache["reason/cot"] = response

        # execute
        answer_format_flag = "The answer is" in response
        pred_str = (
            response.split("The answer is")[-1].strip(".").replace(",", "").strip()
        )

        try:
            all_digit = re.findall(r"[-+]?\d*\.?\d+|\d+", pred_str)
            if answer_format_flag:
                pred = all_digit[0]
            else:
                pred = all_digit[-1]
            pred = floatify_ans(pred)
            if pred is not None and "%" in pred_str:
                pred = pred / 100
        except Exception as e:
            print(e)
            print(pred_str)
            pred = None
        score = 0.0
        if pred is not None:
            score = 1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
        self.cache["reason/cot/ans"] = pred
        self.cache["reason/cot/score"] = score
        self.metrics["cot/correct"] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"score: {score}")

    def reason_pot(self):
        """
        Reason with Program-of-Thought, prompts from PAL
        """
        question = self.cache["inst/question"]
        chat_input = self.build_chat_input(POT_SYSTEM, POT.format(question=question))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache["reason/pot"] = response

        # execute
        score = 0.0
        try:
            pred = safe_execute(response)
            pred = floatify_ans(pred)
            if pred is not None:
                score = (
                    1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
                )
        except Exception as e:
            pred = None
            score = 0.0
        self.cache["reason/pot/ans"] = pred
        self.cache["reason/pot/score"] = score
        self.metrics["pot/correct"] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"score: {score}")

    def reason_eot(self):
        """
        Reason with Equation-of-Thought
        """
        question = self.cache["inst/question"]
        chat_input = self.build_chat_input(EOT_SYSTEM, EOT.format(question=question))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache["reason/eot"] = response

        # execute
        pattern = r"\s*\(\d+\)$"
        equations = []
        for line in response.split("\n"):
            if "=" in line:
                line = "=".join(line.split("=")[:2])
                equations.append(line.strip())
            else:
                continue

        try:
            solutions = safe_solve_equation_system(equations)
            pred = floatify_ans(solutions["ans"])
            score = 1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
        except Exception as e:
            if self.debug:
                print(e)
            pred = None
            score = 0.0

        self.cache["reason/eot/equations"] = equations
        self.cache["reason/eot/ans"] = pred
        self.cache["reason/eot/score"] = score
        self.metrics["eot/correct"] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"equations: {equations}")
            print(f"score: {score}")

    def reason_peano(self):
        """
        Reason with Peano, from https://arxiv.org/pdf/2304.09102.pdf
        :return:
        """
        question = self.cache["inst/question"]
        chat_input = self.build_chat_input(
            PEANO_SYSTEM, PEANO.format(question=question)
        )
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache["reason/peano"] = response

        # execute
        eq_list = re.findall(r"\[\[.*?\]\]", response)

        def reformat_incre_equations(x):
            result = ""
            if len(x) >= 1:
                for eq in x:
                    if len(result) == 0:
                        result += eq[2:-2]
                    else:
                        result += ", " + eq[2:-2]
            return result

        def reformat_equations_from_peano(eq_list):
            result = ""
            for eq in eq_list.split(","):
                if "eq" in eq:
                    if len(result) == 0:
                        result += eq[eq.index("eq") + 2 :]
                    else:
                        result += ", " + eq[eq.index("eq") + 2 :]
                elif "answer" in eq:
                    if len(result) == 0:
                        result += eq[eq.index("answer") + 6 :].strip() + " = ?"
                    else:
                        result += ", " + eq[eq.index("answer") + 6 :].strip() + " = ?"
            return result

        if len(eq_list) > 0:
            eq_list = reformat_equations_from_peano(reformat_incre_equations(eq_list))

        answer = get_final_using_sympy(eq_list)
        try:
            score = (
                1.0
                if abs(float(answer) - float(self.cache["inst/gold_answer"])) < 1e-3
                else 0.0
            )
        except Exception as e:
            print(e)
            score = 0.0

        self.cache["reason/peano/ans"] = answer
        self.cache["reason/peano/score"] = score
        self.metrics["peano/correct"] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"ans: {answer}")
            print(f"gold: {self.cache['inst/gold_answer']}")
            print(f"score: {score}")

    def think_check(self, method="pot"):
        """
        Think moduel: Check
        For PoT, check with ASSERT
        """
        question = self.cache["inst/question"]
        pred = self.cache[f"reason/{method}/ans"]

        if method == "pot":
            code = self.cache[f"reason/{method}"]
            variables = get_stepwise_exec_results(code, method)
        elif method == "eot":
            code = self.cache[f"reason/{method}/equations"]
            try:
                var_dict = safe_solve_equation_system(code)
                all_var_name = sort_words_by_first_appearance(
                    list(var_dict.keys()), " ".join(code)
                )
                all_var_name = [
                    name for name in all_var_name if name != "x" and name != "ans"
                ]
                variables = [f"{name} = {var_dict[name]}" for name in all_var_name]

            except Exception as e:
                variables = code
        else:
            raise NotImplementedError(f"Method {method} not supported")

        chat_input = self.build_chat_input(
            ASSERT_SYSTEM,
            ASSERT_PROMPT.format(question=question, variables="\n".join(variables)),
        )
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache[f"think/check/{method}/variables"] = variables
        self.cache[f"think/check/{method}"] = response

        # execute
        if method == "pot":
            code = variables + [f"ans = {pred}"] + [response]
        elif method == "eot":
            code = [f"x = {pred}"] + variables + [f"ans = x"] + [response]
        code_str = "\n".join(code)

        check_flag = False if timeout_exec(code_str) is None else True

        self.cache[f"think/check/{method}/flag"] = int(check_flag)

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"code: {code_str}")
            print(f"response: {response}")
            print(f"score: {self.cache[f'reason/{method}/score']}")
            print(f"flag: {check_flag}")

    ####

    def think_meta_eval(self, method="cot"):
        if method not in ["cot", "pot", "eot"]:
            raise ValueError(f"Invalid method '{method}' passed to think_meta_eval")

        question = self.cache["inst/question"]
        pred_init = self.cache.get(f"reason/{method}/ans")
        if pred_init is None:
            raise ValueError(f"Prediction for method '{method}' not found in cache")
        answer = self.cache[f"reason/{method}"]
        chat_input = self.build_chat_input(
            META_EVAL_SYSTEM,
            META_EVAL.format(question=question, answer=answer),
        )
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache[f"metacognitive_eval/{method}"] = response

        if "the solution is correct" in response.lower():
            pred = pred_init
            self.cache[f"metacognitive_eval/{method}/ans"] = pred
        elif "the solution is incorrect" in response.lower():
            answer_format_flag = "Corrected solution:" in response
            pred_str = (
                response.split("Corrected solution:")[-1]
                .strip(".")
                .replace(",", "")
                .strip()
            )
            try:
                all_digit = re.findall(r"[-+]?\d*\.?\d+|\d+", pred_str)
                if answer_format_flag:
                    pred = all_digit[0]
                else:
                    pred = all_digit[-1]
                pred = floatify_ans(pred)
                if pred is not None and "%" in pred_str:
                    pred = pred / 100
            except Exception as e:
                print(e)
                print(pred_str)
                pred = None
            self.cache[f"metacognitive_eval/{method}/ans"] = pred
        else:
            pred = None
            print(f"Warning: Unknown response from metacognitive eval: {response}")
        score = 0.0
        if pred is not None:
            score = 1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
        self.cache[f"metacognitive_eval/{method}/score"] = score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"score: {score}")
            print(f"score: {self.cache.get(f'reason/{method}/score', 'N/A')}")

    def think_meta_eval_deepseek(self, method="cot"):
        if method not in ["cot", "pot", "eot"]:
            raise ValueError(f"Invalid method '{method}' passed to think_meta_eval")

        question = self.cache["inst/question"]
        pred_init = self.cache.get(f"reason/{method}/ans")
        # if pred_init is None:
        #     raise ValueError(f"Prediction for method '{method}' not found in cache")
        answer = self.cache[f"reason/{method}"]
        chat_input = self.build_chat_input(
            META_EVAL_SYSTEM,
            META_EVAL.format(question=question, answer=answer),
        )
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache[f"metacognitive_eval/{method}"] = response
        match = re.search(r"Conclusion:(.*?)Correct solution", response, re.DOTALL)
        if match is None:
            matched_str = "incorrect"
        else:
            matched_str = match.group(1).strip()
        if "incorrect" in matched_str.lower():
            if "Final numeric result:" in response:
            #answer_format_flag = "Final numeric result:" in response
                pred_str = (
                    response.split("Final numeric result:")[-1]
                    .strip(".")
                    .replace(",", "")
                    .strip()
                )
            else:
                pred_str = (
                    response.split("Correct solution:")[-1]
                    .strip(".")
                    .replace(",", "")
                    .strip()
                )
            try:
                all_digit = re.findall(r"[-+]?\d*\.?\d+|\d+", pred_str)
                # if answer_format_flag:
                #     pred = all_digit[0]
                # else:
                pred = all_digit[-1]
                pred = floatify_ans(pred)
                if pred is not None and "%" in pred_str:
                    pred = pred / 100
            except Exception as e:
                print(e)
                print(pred_str)
                pred = None
            self.cache[f"metacognitive_eval/{method}/ans"] = pred
        elif "correct" in matched_str.lower():
            pred = pred_init
            self.cache[f"metacognitive_eval/{method}/ans"] = pred
        else:
            pred = None
            print(f"Warning: Unknown response from metacognitive eval: {response}")
        score = 0.0
        if pred is not None:
            score = 1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
        self.cache[f"metacognitive_eval/{method}/score"] = score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"score: {score}")
            print(f"score: {self.cache.get(f'reason/{method}/score', 'N/A')}")

    ####
    def think_refine(self):
        """
        Self-refine on PAL
        """
        code = self.cache["reason/pot"]
        chat_input = self.build_chat_input(REFINE_SYSTEM, REFINE.format(code=code))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache[f"think/refine"] = response

        # execute
        score = 0.0
        try:
            code = extract_code(response)
            code = [line for line in code.split("\n") if "print" not in line]
            pred = safe_execute_func("\n".join(code))
            pred = floatify_ans(pred)
            if pred is not None:
                score = (
                    1.0 if abs(pred - self.cache["inst/gold_answer"]) < 1e-3 else 0.0
                )
        except Exception as e:
            pred = None
            score = 0.0
        self.cache["think/refine/ans"] = pred
        self.cache["think/refine/score"] = score
        self.metrics["refine/correct"] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input_to_string(chat_input)}")
            print(f"response: {response}")
            print(f"ori_score: {self.cache['reason/pot/score']}")
            print(f"refine_score: {score}")

    """
    def think_refine_eot(self):
        #Self-refine on PAL
        question = self.cache["inst/question"]
        code = self.cache['reason/pot']
        chat_input = self.build_chat_input(REFINE_EOT_SYSTEM, REFINE_EOT.format(question=question, code=code))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache[f'reason/eot'] = response

        # execute
        equations = extract_equations(response)
        try:
            solutions = safe_solve_equation_system(equations)
            pred = floatify_ans(solutions['ans'])
            score = 1.0 if abs(pred - self.cache['inst/gold_answer']) < 1e-3 else 0.0
        except Exception as e:
            if self.debug:
                print(e)

            pred = None
            score = 0.0

        self.cache['reason/eot/equations'] = equations
        self.cache['reason/eot/ans'] = pred
        self.cache['reason/eot/score'] = score
        self.metrics['eot/correct'] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"equations: {equations}")
            print(f"score: {score}")

    
    def reflection_reason_php(self, method='cot'):
        # Iterative-refinement module for cot, prompts from php
        question = self.cache["inst/question"]
        hint_ans = self.cache[f'reason/{method}/ans']
        self.cache[f'inst/hint_ans'] = self.cache[f'reason/{method}/ans']
        self.cache[f'inst/hint_score'] = self.cache[f'reason/{method}/score']

        chat_input = self.build_chat_input(PHP_SYSTEM, PHP.format(question=question,
                                                                  answer=hint_ans))
        response = get_chat_response(self.args, chat_input, self.api_key, self.ORG_ID)
        self.cache['reason/cot'] = response

        # execute
        answer_format_flag = 'The answer is' in response
        pred_str = response.split('The answer is')[-1].strip('.').replace(',', '').strip()
        # print(f"pred: {pred_str}")
        try:
            all_digit = re.findall(r"[-+]?\d*\.?\d+|\d+", pred_str)
            if answer_format_flag:
                pred = all_digit[0]
            else:
                pred = all_digit[-1]
            pred = floatify_ans(pred)
            if pred is not None and '%' in pred_str:
                pred = pred / 100
        except Exception as e:
            print(e)
            print(pred_str)
            pred = None
        score = 0.0
        if pred is not None:
            score = 1.0 if abs(pred - self.cache['inst/gold_answer']) < 1e-3 else 0.0
        self.cache['reason/cot/ans'] = pred
        self.cache['reason/cot/score'] = score
        self.metrics['cot/correct'] += score

        if self.debug:
            print(f"=== inst i: {self.id} ===")
            print(f"chat_input: {chat_input}")
            print(f"response: {response}")
            print(f"score: {score}")
    """

    def save_cache(self):
        with open(self.result_path, "a") as out_f:
            output_json = json.dumps(self.cache)
            out_f.write(output_json + "\n")

    def print_cache(self):
        for k, v in self.cache.items():
            print(f"{k}: {v}")

    def get_metrics(self):
        self.metrics["cot/acc"] = self.metrics["cot/correct"] / len(self.data)
        self.metrics["pot/acc"] = self.metrics["pot/correct"] / len(self.data)
        self.metrics["eot/acc"] = self.metrics["eot/correct"] / len(self.data)
        self.metrics["peano/acc"] = self.metrics["peano/correct"] / len(self.data)
        self.metrics["refine/acc"] = self.metrics["refine/correct"] / len(self.data)

        self.metrics["total"] = len(self.data)
        return self.metrics
    
    def reason_tot(self):

        def cluster(sentences):
            sentences = [s.lower() for s in sentences]

            model = SentenceTransformer('all-mpnet-base-v2')

            embeddings = model.encode(sentences, convert_to_tensor=True)

            num_clusters = 2
            kmeans = KMeans(n_clusters=num_clusters, random_state=0)
            cluster_indices = kmeans.fit_predict(np.array(embeddings).tolist())

            clusters = {}
            for i, cluster_id in enumerate(cluster_indices):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(sentences[i].lower())

            # for cluster_id, sentences_in_cluster in clusters.items():
            #     print(f"Cluster {cluster_id + 1}:\n")
            #     for sentence in sentences_in_cluster:
            #         print(sentence)
            #     print("\n")
            return clusters

        #Parameters
        questions_big = []
        status_big = []


        max_steps = 5
        k=10
        pred = []
        true = []
        num_questions_to_solve = 2
        correct = 0
        wrong = 0
        total = 0
        store_question = []
        store_gpt0 = []
        store_gpt1 = []
        store_gpt2 = []
        store_true = []
        store_chosen = []
        store_answer = []
        store_chosen_cache = []


        dataset = load_dataset("data/gsm8k/test.jsonl")



        status = ["None"]

        

        for i in range(max_steps):
            store_question.append(self.cache["inst/question"])
            store_true.append(self.cache["inst/gold_answer"])
            layer_options = []
            print("*****************NEW STEP*****************")
            print(f"The status array is {status} \n\n")
            initial_promp = initial_prompt_temp + str(question) + str("\n Steps taken so far:") + str(status) + output_string
            out = generate_text_phi(initial_promp,k)

            for j in range(k):
                print(f"######## This is the thought from instance number {j} ##########")
                outputs = parse_output_options(out[j])
                print(f"The parsed output is {outputs}")
                a = [one_option for one_option in outputs]
                layer_options.extend(a)

            store_gpt0.append(layer_options[0])
            store_gpt1.append(layer_options[1])
            store_gpt2.append(layer_options[2])

            chosen_option = ranking(layer_options,question,status)
            layer_entropy = cluster(layer_options)
            layer_entropy = list(layer_entropy.values())


            for clus in range(len(layer_entropy)):
                print(f"Chosen option is {chosen_option[0].lower()} and the layer_entropy is {layer_entropy[clus]}")
            if(eval(chosen_option[0]).lower() in layer_entropy[clus]):
                entropy = (len(layer_entropy[clus])/10.0)
                print(f"THE ENTROPY IS {entropy}")

            store_chosen.append(chosen_option)
            store_chosen_cache.append(list(set(re.findall(r'\d+(?:\.\d+)?', chosen_option[0]))))

            if("None") in status:
                status = [chosen_option]
            else:
                status.append(chosen_option)
            print(f"The option chosen as the best choice is {chosen_option}")
            print("\n\n\n")


        question_summary = generate_text_phi(summary_question_prompt + str(question),1)
        predict_prompt_full = predict_prompt + str(question_summary) + str("Based on the current status - ") + str(status) + str("\n Just give the final answer in number nothing else no text, no calculations")

        answer = generate_text_phi(predict_prompt_full ,1)

        pred.append([answer[0]]*max_steps)
        true.append([true_answer]*max_steps)


        try:
            if(float(answer[0])==true_answer):
                correct +=1

                store_answer.append(["C"]*max_steps)

            else:
                wrong+=1
                store_answer.append(["W"]*max_steps)
            total+=1
        except:
            store_answer.append(["Error"]*max_steps)
            continue

        questions_big.append(question)
        status_big.append(status)
        print(f"Current status is -----------------> correct = {correct} and wrong = {wrong}")


    @staticmethod
    def build_chat_input(instruction, user_input):
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": user_input},
        ]