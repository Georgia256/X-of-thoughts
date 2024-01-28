TOT_SYSTEM = """
Imagine you are trying to solve a math problem with a step-by-step approach. At each step, you should propose a single next step to solve the problem involving a single arithmetic option. If there are multiple options for how to proceed, you should generate up to 3 options.

The format of the problem is as below, follow this format only
Input: XXXX
Steps taken so far: YYYY
Output: ZZZZ

NOTE: The options should not be sequential or connected with each other, each option should be in a way that it can be evaluated independently. Dont jump to the result directly.
IMPORTANT: MAKE SURE NOT TO HAVE THE DIRECT ANSWER IN YOUR POSSIBLE STEPS OUTPUT, JUST MAKE ONE STEP AT A TIME.
"""


# prompt from TOT
TOT = """
Input: "Jasper will serve charcuterie at his dinner party. He buys 2 pounds of cheddar cheese for $10, a pound of cream cheese that cost half the price of the cheddar cheese, and a pack of cold cuts that cost twice the price of the cheddar cheese. How much does he spend on the ingredients?"

Steps take so far: [Calculate the price of cheddar cheese which is $10 (given)]


Output: Possible independent steps:
1) Calculate the price of cold cuts which is 2*10 = $20.
2)Calculate the price of cream cheese which is 10/2 = $5 per pound. <END>

Input: "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"

Steps taken so far: [None]

Output: Possible next steps:
1) Convert the minutes of babysitting to hours.
2) Convert the wage per hour to wage per minute. <END>

Input: "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

Steps taken so far: [Number of letter written to 1 friend in a week = 2 as he writes twice a week]

Output: Possible next steps:
1) Number of letter written to 2 friends in a week = 2*2 = 4 letters a week.
2) Calculate the number of pages written to 1 friend in a week = 2*3 = 6 pages. <END>

Input: {question}

Steps taken so far: {status}

Output: Possible next steps: 
""".strip()


