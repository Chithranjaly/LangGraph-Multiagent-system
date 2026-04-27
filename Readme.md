# Multi-Agent Task Execution System (LangGraph)

## Overview
This project implements a controlled multi-agent workflow using LangGraph and LLMs to execute complex tasks in a structured and reliable manner.

Instead of relying on a single LLM response, the system follows an iterative loop:
**Planning → Execution → Evaluation → Retry/Advance**

This approach improves robustness, traceability, and control over task completion.

---

## Architecture

The system is composed of three core agents:

### Planner
- Decomposes a high-level task into sequential subtasks
- Produces a structured list of actionable steps

### Executor
- Executes each subtask step-by-step
- Uses previous results as context for continuity

### Critic
- Evaluates the output of each step
- Returns:
  - `PASS` → proceed to next step
  - `FAIL` → retry (with limit)



## Key Features

- Iterative multi-agent loop (Plan → Execute → Evaluate)
- Controlled termination (prevents infinite loops)
- Retry mechanism with limits
- Self-evaluation using a critic agent
- Structured state management via LangGraph
- JSON output logging for reproducibility
- Secure API key handling using environment variables

---

## Tech Stack

- Python
- LangGraph
- LangChain
- OpenAI API

---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/repo-name.git
cd repo-name
```
## Install dependencies

pip install -r requirements.txt

## How to Run

python multi_agent_demo.py



