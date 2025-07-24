import asyncio
from abc import ABC, abstractmethod
from typing import List
from hugging_face import HuggingFaceLLM


class NodeExecutionStrategy(ABC):
    @abstractmethod
    async def execute(self, node: str, inputs: List[str], duration: float) -> str:
        pass
      

class DurationSleepStrategy(NodeExecutionStrategy):
    async def execute(self, node: str, inputs: List[str], duration: float) -> str:
        sleep_time = 0.1 * duration
        await asyncio.sleep(sleep_time)
        return f"{node}_result({' + '.join(inputs) if inputs else f'init_{node}'})"



class LLMStrategy(NodeExecutionStrategy):
    def __init__(self, llm):
        self.llm = llm  # e.g. HuggingFaceLLM or MockLLM

    async def execute(self, node: str, inputs: List[str], duration: float) -> str:
        prompt = f"Node {node} - process: " + (" + ".join(inputs) if inputs else f"init_{node}")
        return await asyncio.to_thread(self.llm._call, prompt)
