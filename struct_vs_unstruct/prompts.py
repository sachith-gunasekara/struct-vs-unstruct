### SELECT PROMPT ###

SELECT_PROMPT = """Select several reasoning modules that are crucial to utilize in order to solve the given task.

All reasoning module descriptions:
{reasoning_modules}

Task: 
{task_description}

Select several modules are crucial for solving the task above. 
Your response should only be the list of select modules, no explanations or solutions are required."""


### ADAPT PROMPT ###

ADAPT_PROMPT = """Rephrase and specify each reasoning module so that it better helps solving the task.

SELECTED module descriptions:
{selected_modules}

Task:
{task_description}

Adapt each reasoning module description to better solve the task.
Your response should only be the list of adapted modules, no explanations or solutions are required."""

### DERIVING REASONING MODULES PROMPT ###

DERIVING_REASONING_MODULES_PROMPT = """Generate a few atomic seed reasoning modules that can be used to solve the given task.

Example Reasoning Modules:
- What are the key assumptions underlying this problem?
- What are the potential obstacles or challenges that might arise in solving this problem?
- Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?
- etc...

Your response should be as follows (Do not include a prepended intro):
[List of unnumbered reasoning modules]

Task:
{task_description}"""


### REASONING PLAN PROMPT ###

NL_REASONING_PLAN_PROMPT = """Operationalize the reasoning modules into a step-by-step reasoning plan in plain English to solve the given task.
Make sure the plan is concrete, intuitive, and unambigous.
The reasoning plan should help an AI agent follow it and be able to derive a solution to the given task.

Here's an example:
Example task:
If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:
Find position after instruction 1.
FInd position after instruction 2.
Find position after instruction n.
Is final position the same as starting position?

Reasoning Module description:
{adapted_modules}

Task:
{task_description}

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN that can be followed to arrive at the correct answer the given task."""


### FOLLOW PLAN PROMPT ###

FOLLOW_REASONING_PLAN_PROMPT = """Follow the reasoning plan step-by-step to arrive at the correct answer
Your response should only contain the reasoning process for the given task.
The final answer should be complete and not only be the letter or Answer corresponding to the task (Example answer:  K. Ananda).

Reasoning Plan:
{reasoning_plan}

Task:
{task_description}"""


### STRUCTURE RESPONSE ###

STRUCTURE_RESPONSE_PROMPT = """From the given reasoning process, extract the reasoning trajectory and answer and provide it in a JSON format as given below.

Here are some additional details:
- The reasoning trajectory should be the complete string other than the final answer in the reasoning process. Do not condense or omit content. Remember to maintain the original structure including white spaces, newlines etc.
- The answer should be the answer mentioned in the reasoning process to the given task.
- Your response should only contain the formatted JSON response, no explaination required

Task:
{task_description}

Reasoning Process:
{reasoning}

JSON Structure:
{{
    "trajectory": [The reasoning trajectory],
    "answer_pred": [Answer]
}}"""