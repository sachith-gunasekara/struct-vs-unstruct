# Directly adapted and modified from https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover

import os
import getpass
from typing import Optional, TypedDict

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
import pandas as pd
from pyprojroot import here

from struct_vs_unstruct.helpers.llm import model_kwargs, MODEL_ID

load_dotenv()

if not os.getenv("NVIDIA_API_KEY"):
    # Note: the API key should start with "nvapi-"
    os.environ["NVIDIA_API_KEY"] = getpass.getpass("Enter your NVIDIA API key: ")


select_prompt = hub.pull("hwchase17/self-discovery-select")
adapt_prompt = hub.pull("hwchase17/self-discovery-adapt")
structured_prompt = hub.pull("hwchase17/self-discovery-structure")
reasoning_prompt = hub.pull("hwchase17/self-discovery-reasoning")

NL_REASONING_PLAN_PROMPT = """Operationalize the reasoning modules into a step-by-step reasoning plan in plain English to solve the given task.
Make sure the plan is concrete, intuitive, and unambigous.
The reasoning plan should help any person follow it and be able to derive a solution to the given task.

Here's an example:

Example task:
If you follow these instructions, do you return to the starting point? Always face forward. Take 1 step backward. Take 9 steps left. Take 2 steps backward. Take 6 steps forward. Take 4 steps forward. Take 4 steps backward. Take 3 steps right.

Example reasoning structure:
Find position after instruction 1.
FInd position after instruction 2.
Find position after instruction n.
Is final position the same as starting position?

Adapted module description:
{adapted_modules}

Task: {task_description}

Implement a reasoning plan for solvers to follow step-by-step and arrive at the correct answer.

Note: do NOT actually arrive at a conclusion in this pass. Your job is to generate a PLAN so that in the future you can follow it to arrive at the correct conclusion for tasks like this"""

nl_reasoning_plan_prompt = PromptTemplate.from_template(NL_REASONING_PLAN_PROMPT)

FOLLOW_REASONING_PLAN_PROMPT = """Follow the reasoning plan step-by-step to arrive at the correct answer
Your response should only contain the reasoning process for the given task.

Reasoning Plan:
{reasoning_plan}

Task: {task_description}"""

follow_reasoning_plan_prompt = PromptTemplate.from_template(FOLLOW_REASONING_PLAN_PROMPT)


# Function to log token usage to a CSV file using pandas.concat()
def log_token_usage(result, file_name=here('struct_vs_unstruct/logs/token_usage_log.csv')):
    # Check if the file already exists
    if os.path.exists(file_name):
        # Load the existing file
        df = pd.read_csv(file_name)
    else:
        # Create a new DataFrame with the appropriate columns
        df = pd.DataFrame(columns=["prompt_tokens", "completion_tokens", "total_tokens"])

    token_data = result.response_metadata.get("token_usage", {})

    # Create a DataFrame for the new token usage data
    new_row_df = pd.DataFrame([{
        "prompt_tokens": token_data.get('prompt_tokens', 0),
        "completion_tokens": token_data.get('completion_tokens', 0),
        "total_tokens": token_data.get('total_tokens', 0)
    }])

    # Use pd.concat to append the new row to the existing DataFrame
    df = pd.concat([df, new_row_df], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_name, index=False)


class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    reasoning_plan: Optional[str]
    answer: Optional[str]


model = ChatNVIDIA(
    model=MODEL_ID,
    **model_kwargs
)

def output_parser(result):
    parser = StrOutputParser()
    return parser.invoke(result)

def select(inputs):
    select_chain = select_prompt | model
    
    result = select_chain.invoke(inputs)

    log_token_usage(result)

    return {"selected_modules": output_parser(result)}


def adapt(inputs):
    adapt_chain = adapt_prompt | model

    result = adapt_chain.invoke(inputs)

    log_token_usage(result)

    return {"adapted_modules": output_parser(result)} # -> reasoning 


def structure(inputs):
    structure_chain = structured_prompt | model

    result = structure_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning_structure": output_parser(result)}


def reason(inputs):
    reasoning_chain = reasoning_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"answer": output_parser(result)}


def nl_reasoning_plan(inputs):
    reasoning_chain = nl_reasoning_plan_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning_plan": output_parser(result)}


def follow_reasoning_plan(inputs):
    reasoning_chain = follow_reasoning_plan_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"answer": output_parser(result)}


def add_nodes(modified: bool = False):
    graph = StateGraph(SelfDiscoverState)

    graph.add_node(select)
    graph.add_node(adapt)

    if not modified:
        graph.add_node(structure)
        graph.add_node(reason)
    else:
        graph.add_node(nl_reasoning_plan)
        graph.add_node(follow_reasoning_plan)

    return graph


def create_self_discover_graph(modified: bool = False):
    graph = add_nodes(modified)

    graph.add_edge(START, "select")
    graph.add_edge("select", "adapt")

    if not modified:
        graph.add_edge("adapt", "structure")
        graph.add_edge("structure", "reason")
        graph.add_edge("reason", END)
    else:
        graph.add_edge("adapt", "nl_reasoning_plan")
        graph.add_edge("nl_reasoning_plan", "follow_reasoning_plan")
        graph.add_edge("follow_reasoning_plan", END)

    app = graph.compile()

    return app


def self_discover(task_description: str, modified: bool = False):
    reasoning_modules = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        # "3. How could I measure progress on this problem?",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "6. What are the potential risks and drawbacks of each solution?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
        "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
        "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
        "37. Ignoring the current best solution, create an entirely new solution to the problem."
        # "38. Let’s think step by step."
        "39. Let’s make a step by step plan and implement it with good notation and explanation.",
    ]
    reasoning_modules_str = "\n".join(reasoning_modules)

    app = create_self_discover_graph(modified)

    return app.invoke(
        {
            "task_description": task_description,
            "reasoning_modules": reasoning_modules_str,
        }
    )