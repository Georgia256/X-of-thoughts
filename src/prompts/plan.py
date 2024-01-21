# PLAN_SYSTEM = """
# You are an expert in math reasoning. Please choose the best method to solve the math word problems. Read the description of the methods and the examples. You only need to suggest the best method. Do not solve the problem.
# """.strip()
'''
PLAN_SYSTEM = """
You are an expert in math reasoning. Please choose the best method to solve the math word problems between the following methods: 
- Python Program: This method generates a Python program that can solve the given question. It takes in the question and possible context and produces a program. Normally, we consider using this method when the questions and contexts involve forward reasoning, such as arithmetic operations over multiple numbers, or when the questions involve complex logical operations, such as "if-else" statements.
- System of linear equations: This method builds a math model and generates a system of linear equations that contains the answer as an unknown variable. Normally, we consider using this method when the questions and contexts involve an unknown variable that must be used to build an equation, especially when the question can be better modeled with abstract mathematical declarations, or when the unknown variable appears at the beginning of the questions and needs backward reasoning to solve it.
- Chain of Thought: This method involves breaking down the question into a series of logical steps or considerations, almost like a narrative or a conversation with oneself. It's typically used when the problem requires multi-step reasoning, where each step builds upon the previous one. The method is particularly useful for complex problems that don't lend themselves to straightforward computational or algorithmic solutions. Instead of directly arriving at the answer, the chain of thought method involves articulating each step of the thought process, providing clarity and justification for each decision or inference made along the way. This approach is ideal for problems where the reasoning path is as important as the final answer, such as in puzzles, riddles, or scenarios with multiple variables and possible interpretations.
""".strip()
'''
PLAN_SYSTEM ="""
You are an expert in math reasoning. please rate the probability of successfully solving this problem using each of the following methods:
- Python Program: This method generates a Python program that can solve the given question. It takes in the question and possible context and produces a program. Normally, we consider using this method when the questions and contexts involve forward reasoning, such as arithmetic operations over multiple numbers, or when the questions involve complex logical operations, such as "if-else" statements.
- System of linear equations: This method builds a math model and generates a system of linear equations that contains the answer as an unknown variable. Normally, we consider using this method when the questions and contexts involve an unknown variable that must be used to build an equation, especially when the question can be better modeled with abstract mathematical declarations, or when the unknown variable appears at the beginning of the questions and needs backward reasoning to solve it.
- Chain of Thought: This method involves breaking down the question into a series of logical steps or considerations, almost like a narrative or a conversation with oneself. It's typically used when the problem requires multi-step reasoning, where each step builds upon the previous one. The method is particularly useful for complex problems that don't lend themselves to straightforward computational or algorithmic solutions. Instead of directly arriving at the answer, the chain of thought method involves articulating each step of the thought process, providing clarity and justification for each decision or inference made along the way. This approach is ideal for problems where the reasoning path is as important as the final answer, such as in puzzles, riddles, or scenarios with multiple variables and possible interpretations.
The format of your answer should be the following:
Python_Program_rating:{rating}
System_of_linear_equations_rating:{rating}
Chain_of_Thought_rating:{rating}
""".strip()
#3-shot
PLAN="""
Question: Arnel had ten boxes of pencils with the same number of pencils in each box.  He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?
Python_Program_rating:3
System_of_linear_equations_rating:5
Chain_of_Thought_rating:4 <END>

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Python_Program_rating:2
System_of_linear_equations_rating:2
Chain_of_Thought_rating:5 <END>

Question: Larry spends half an hour twice a day walking and playing with his dog. He also spends a fifth of an hour every day feeding his dog. How many minutes does Larry spend on his dog each day?
Python_Program_rating:5
System_of_linear_equations_rating:2
Chain_of_Thought_rating:3 <END>

Question: {question}
Python_Program_rating:
System_of_linear_equations_rating:
Chain_of_Thought_rating:
""".strip()

################
'''
# 8-shot
PLAN = """
Question: Arnel had ten boxes of pencils with the same number of pencils in each box.  He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?
Method: System of linear equations <END>

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Method:  Chain of Thought <END>

Question: Larry spends half an hour twice a day walking and playing with his dog. He also spends a fifth of an hour every day feeding his dog. How many minutes does Larry spend on his dog each day?
Method: Python Program <END>

Question: Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Method:  Chain of Thought <END>

Question: Angela is a bike messenger in New York. She needs to deliver 8 times as many packages as meals. If she needs to deliver 27 meals and packages combined, how many meals does she deliver?
Method: System of linear equations <END>

Question: Last year Dallas was 3 times the age of his sister Darcy. Darcy is twice as old as Dexter who is 8 right now. How old is Dallas now?
Method: System of linear equations <END>

Question: A small poultry farm has 300 chickens, 200 turkeys and 80 guinea fowls. A strange, incurable disease hit the farm and every day the farmer lost 20 chickens, 8 turkeys and 5 guinea fowls. After a week, how many birds will be left in the poultry?
Method: Python Program <END>

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Method: Python Program <END>

Question: Alyssa, Keely, and Kendall ordered 100 chicken nuggets from a fast-food restaurant. Keely and Kendall each ate twice as many as Alyssa. How many did Alyssa eat?
Method: System of linear equations <END>

Question: Conor can chop 12 eggplants, 9 carrots, and 8 potatoes in a day. If he works 4 times a week, how many vegetables can he chop?
Method: Python Program <END>

Question: A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?
Method:  Chain of Thought <END>

Question: {question}
Method:
""".strip()
'''
'''
PLAN_SYSTEM = """
You are an expert in math reasoning. Please choose the best method to solve the math word problems between the following methods: 
- Chain of Thought: This method is ideal for multi-step reasoning problems, where each step builds on the previous one. It's like a conversation with oneself, breaking down the question into a series of logical steps or considerations. This method shines in scenarios where the reasoning path is critical, such as complex problems with multiple variables, puzzles, or problems requiring a narrative explanation of the reasoning process.
If the problem is not straightforward, it is better to use this method.
- Program of Thought: This method generates a Python program that can solve the given question. It is used for questions involving straightforward arithmetic operations or when implementing complex logical operations, like "if-else" statements.
- Equations of Thought: This method involves formulating a math model as a system of linear equations to solve for unknown variables. It's best for questions where variables are integral to the problem and can be defined in equations, particularly for backward reasoning processes.
""".strip()

# 8-shot
PLAN = """
Question: Arnel had ten boxes of pencils with the same number of pencils in each box.  He kept ten pencils and shared the remaining pencils equally with his five friends. If his friends got eight pencils each, how many pencils are in each box?
Method: Equations of Thought <END>

Question: A gardener has a new flower bed that can fit 24 flowers. He plants tulips and roses. He plants twice as many tulips as roses. How many of each type does he plant?
Method: Chain of Thought <END>

Question: Larry spends half an hour twice a day walking and playing with his dog. He also spends a fifth of an hour every day feeding his dog. How many minutes does Larry spend on his dog each day?
Method: Program of Thought <END>

Question: A library initially had 1,500 books. Last year, they added thrice as many books as they sold. If the total number of books is now 2,100, how many books were added and sold?
Method: Chain of Thought <END>

Question: Angela is a bike messenger in New York. She needs to deliver 8 times as many packages as meals. If she needs to deliver 27 meals and packages combined, how many meals does she deliver?
Method: Equations of Thought <END>

Question: Last year Dallas was 3 times the age of his sister Darcy. Darcy is twice as old as Dexter who is 8 right now. How old is Dallas now?
Method: Equations of Thought <END>

Question: A small poultry farm has 300 chickens, 200 turkeys and 80 guinea fowls. A strange, incurable disease hit the farm and every day the farmer lost 20 chickens, 8 turkeys and 5 guinea fowls. After a week, how many birds will be left in the poultry?
Method: Program of Thought <END>

Question: In a game, a player earns 5 points for each level completed and loses 2 points for each level failed. If the player completed 15 levels and failed 4, what is their total score?
Method: Chain of Thought <END>

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Method: Program of Thought <END>

Question: Alyssa, Keely, and Kendall ordered 100 chicken nuggets from a fast-food restaurant. Keely and Kendall each ate twice as many as Alyssa. How many did Alyssa eat?
Method: Equations of Thought <END>

Question: Conor can chop 12 eggplants, 9 carrots, and 8 potatoes in a day. If he works 4 times a week, how many vegetables can he chop?
Method: Program of Thought <END>

Question: If each page of a book contains about 300 words, approximately how many words are in a book with 450 pages?
Method: Chain of Thought <END>

Question: {question}
Method:
""".strip()
'''
