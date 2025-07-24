import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict

class GraphState(TypedDict):
    input: str
    response: str

device = torch.device("cpu")

# Load model and tokenizer
model_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Define the callable for LangGraph
def huggingface_generate(state: dict):
    prompt = state["input"]
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": response}

graph = StateGraph(GraphState)
graph.add_node("start", RunnableLambda(lambda x: {"input": x["input"]}))
graph.add_node("llm", RunnableLambda(huggingface_generate))

graph.set_entry_point("start")
graph.add_edge("start", "llm")
graph.add_edge("llm", END)

compiled = graph.compile()

# Test it
result = compiled.invoke({"input": "Explain quantum physics in simple terms."})
print(result["response"])