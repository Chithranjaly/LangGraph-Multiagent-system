from dotenv import load_dotenv
import os
import ast
import json
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
from langchain_xai.chat_models import ChatXAI
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ====================== SHARED STATE ======================
class AgentState(TypedDict):
    task: str
    plan: List[str]           # List of subtasks from planner
    current_step: int
    results: List[str]        # Results from executed steps
    messages: Annotated[List, "add_messages"]
    critique: str
    retry_count: int

# ====================== LLM SETUP (OpenAI) ======================
from langchain_openai import ChatOpenAI

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")

llm = ChatOpenAI(
    model="gpt-4o-mini",      
    temperature=0.7,
)


# ====================== PLANNER NODE ======================
def planner(state: AgentState):
    prompt = ChatPromptTemplate.from_template(
    "You are an expert Task Planner Agent.\n"
    "Break down the following high-level task into 4-6 clear, sequential, actionable subtasks.\n"
    "Return ONLY a Python list of strings.\n"
    "Example format: [\"Step 1\", \"Step 2\", \"Step 3\"]\n\n"
    "Task: {task}"
)
    
    response = llm.invoke(prompt.format_messages(task=state["task"]))
    # Extract numbered lines

    try:
        plan = ast.literal_eval(response.content)
    except Exception:
        raise ValueError("Planner failed to produce valid plan")
    if not plan:
        raise ValueError("Planner returned empty plan")
        
    print("\n=== PLANNER OUTPUT ===\n", response.content)
    return {"plan": plan, "current_step": 0, "results": [],"retry_count": 0, "messages": [AIMessage(content=response.content)]}

# ====================== EXECUTOR NODE ======================
def executor(state: AgentState):
    if state["current_step"] >= len(state.get("plan", [])):
        return {"messages": [AIMessage(content="All tasks completed.")]}

    current_task = state["plan"][state["current_step"]]
    context = "\n".join(state["results"]) if state["results"] else "No previous results yet."

    prompt = ChatPromptTemplate.from_template(
        "You are an Executor Agent. Carefully perform the following subtask.\n"
        "Use the previous results as context when needed.\n"
        "Provide a clear, useful output (code snippet, explanation, summary, etc.).\n\n"
        "Subtask: {current_task}\n\n"
        "Previous context:\n{context}"
    )
    
    response = llm.invoke(prompt.format_messages(current_task=current_task, context=context))
    
    new_result = f"Step {state['current_step'] + 1}: {response.content.strip()}"
    new_results = state["results"] + [new_result]
    
    print(f"\n=== EXECUTOR - Step {state['current_step'] + 1} ===\n", response.content)
    
    return {
        "results": new_results,
        "messages": [AIMessage(content=response.content)]
    }

# ====================== CONTROLLING THE LOOP ======================
MAX_STEPS = 10

def should_continue(state: AgentState):
    if state["current_step"] >= len(state["plan"]):
        return END
    if state["current_step"] >= MAX_STEPS:
        return END
    return "executor"

# ====================== ADDING A CRITIC ======================
def critic(state: AgentState):
    last_result = state["results"][-1]

    prompt = f"""
    Evaluate this result. 
    Answer strictly with:
    PASS - if correct
    FAIL - if incorrect, with reason.

    Result:
    {last_result}
    """

    response = llm.invoke(prompt)

    return {
        "messages": [AIMessage(content=response.content)],
        "critique": response.content
    }

def route_after_critic(state: AgentState):
    if "FAIL" in state.get("critique", ""):
        state["retry_count"] += 1

        if state["retry_count"] >= 2:
            # skip bad step after retries
            state["retry_count"] = 0
            state["current_step"] += 1
            return "executor"

        return "executor"

    # PASS case
    state["retry_count"] = 0
    state["current_step"] += 1

    if state["current_step"] >= len(state["plan"]):
        return END

    if state["current_step"] >= MAX_STEPS:
        return END

    return "executor"




# ====================== BUILD THE GRAPH ======================
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("executor", executor)
workflow.add_node("critic",critic)


workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "critic")
workflow.add_conditional_edges(
    "critic",
    route_after_critic
)


# For simplicity, we stop manually when steps are done (you can improve this later)
app = workflow.compile()

# ====================== RUN THE AGENT ======================
if __name__ == "__main__":
    user_task = "Develop a basic REST API endpoint for adding and listing expenses in a personal finance app using Django DRF with JWT authentication."

    initial_state = {
        "task": user_task,
        "plan": [],
        "current_step": 0,
        "results": [],
        "messages": []
    }

    print("Starting Multi-Agent Workflow...\n")
    result = app.invoke(initial_state)

    def serialize(obj):
        if hasattr(obj, "content"):
            return obj.content
        return str(obj)

    with open("run_output.json", "w") as f:
        json.dump(result, f, indent=2, default=serialize)

    print("\n" + "="*60)
    print("✅ FINAL RESULTS")
    print("="*60)
    for i, res in enumerate(result.get("results", [])):
        print(f"\n{i+1}. {res}")