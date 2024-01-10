####Metacognitive evaluation technique########
META_EVAL_SYSTEM="""
You are an expert in math reasoning. Assess the following solution. During the evaluation, perform the following: 
1)Ensure that the problem's conditions are taken into account.
2)Assess the appropriateness of the methods and formulas used
3)Ensure every calculation and logical step is correct and clearly explained.
4)Verify if the final answer is accurate and addresses the question."
""".strip()

META_EVAL="""
Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8. <END>
E:  Problem's Conditions: The problem states that Olivia has $23 and she buys five bagels, each costing $3. The solution takes into account both the amount Olivia initially has and the total cost of the bagels.
Appropriateness of Methods and Formulas: The solution uses multiplication to find the total cost of the bagels (5 bagels at $3 each) and subtraction to determine the remaining amount of money. These are the correct mathematical operations for this problem.
Accuracy of Calculations and Logical Steps:
        Calculation for the cost of bagels: 5 bagels × $3/bagel = $15. This is correct.
        Calculation for the remaining money: $23 - $15 = $8. This is also correct.
Accuracy of the Final Answer: The final answer is $8, which correctly addresses the question of how much money Olivia has left after purchasing the bagels.<END>

Q:John buys twice as many red ties as blue ties.  The red ties cost 50% more than blue ties.  He spent $200 on blue ties that cost $40 each. How much did he spend on ties?
A:John bought twice as many red ties as blue ties, so he bought 2 x $40 = $80 worth of red ties.\nThe cost of each red tie is 50% more than the cost of each blue tie, so the cost of each red tie is $40 x 1.5 = $60.\nJohn bought $80 worth of red ties, so he bought $80 / $60 = 1.33 red ties.\nSince he bought twice as many red ties as blue ties, he bought 1.33 red ties.\nThe total cost of the ties is $80 + $80 x 1.33 = $106.67.\nThe answer is $106.67
E:Problem's Conditions: The solution seems to misinterpret the problem's conditions. The statement "John buys twice as many red ties as blue ties" is taken to mean that he spent twice as much on red ties as on blue ties, which is incorrect. It means the quantity of red ties is twice that of blue ties, not the cost.
Appropriateness of Methods and Formulas: The method of calculating the cost of red ties based on their price relative to blue ties is correct, but the application is incorrect. The solution should calculate the number of blue ties first, then determine the number of red ties and their cost.
Accuracy of Calculations and Logical Steps:
    Calculation of the number of blue ties: $200 spent on blue ties at $40 each implies he bought $200 / $40 = 5 blue ties.
    Calculation of the number of red ties: Twice as many as blue ties implies 5 × 2 = 10 red ties.
    Calculation of the cost per red tie: 50% more than $40 is $40 + 0.5 × $40 = $60.
    Total cost for red ties: 10 red ties at $60 each = $600.
Accuracy of the Final Answer:The final answer is incorrect. The correct total cost of the ties should be the sum of the cost for blue ties ($200) and the cost for red ties (which needs to be recalculated).After recalculating, we find that John spent 600+200=800$ on ties in total.<END>

Q:Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
A:Every day, Janet eats 3 eggs for breakfast, so she has 16 - 3 = 13 eggs left.\nShe bakes muffins for her friends, so she has 13 - 4 = 9 eggs left.\nShe sells the remainder at the farmers' market for $2 per egg, so she makes 9 * 2 = $18.\nThe answer is $18.
E:Problem's Conditions: The solution correctly considers the total number of eggs laid per day, the number of eggs Janet uses for breakfast and muffins, and the price per egg at the market.
Appropriateness of Methods and Formulas: The solution uses basic arithmetic operations (subtraction and multiplication) to calculate the number of eggs sold and the total income from selling them. These are the appropriate methods for this problem.
Accuracy of Calculations and Logical Steps:
    Calculation of eggs remaining after breakfast: 16 eggs - 3 eggs = 13 eggs.
    Calculation of eggs remaining after baking muffins: 13 eggs - 4 eggs = 9 eggs.
    Calculation of income from selling eggs: 9 eggs × $2/egg = $18.
Accuracy of the Final Answer: The final answer, as per the calculations, is $18. This correctly addresses the question of how much money Janet makes daily at the farmer's market.<END>

Q:{question}
A:{answer}
E:
""".strip()

