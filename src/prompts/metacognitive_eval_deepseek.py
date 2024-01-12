META_EVAL_SYSTEM = """
You are an expert in math reasoning. You will be given a problem, and a solution which you have to assess. 
During the evaluation, perform the following: 
1)Ensure that the problem's conditions are taken into account.
2)Assess the appropriateness of the methods and formulas used.
3)Ensure every calculation and logical step is correct and clearly explained.
4)Verify if the final answer is accurate and addresses the question.
After the evaluation, provide the following:
-State your conclusion about whether the solution is correct or incorrect.
-Provide the correct solution to the problem.
-Provide the final numeric result seperately.

Your answer must be in the following format:
Evaluation:...
Conclusion:...
Correct solution:...
Final numeric result:...
""".strip()

META_EVAL = """
Problem:{question}
Solution:{answer}"""
