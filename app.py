from fastapi import FastAPI
from multi_agent_demo import app as graph_app

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Multi-Agent System is running"}

@app.post("/run")
def run_agent(task: str):
    initial_state = {
        "task": task,
        "plan": [],
        "current_step": 0,
        "results": [],
        "messages": []
    }

    result = graph_app.invoke(initial_state)

    # serialize result
    def serialize(obj):
        if hasattr(obj, "content"):
            return obj.content
        return str(obj)

    return result