import asyncio
import time
import logging
from collections import defaultdict
from typing import Dict
from node_strategy import NodeExecutionStrategy
from weighted_dag import WeightedDAG

logger = logging.getLogger(__name__)

class WorkflowExecutor:
    """
    Executes a weighted DAG asynchronously. Each node is executed only after all its dependencies complete.
    Results and timing information are stored for performance analysis.
    """

    def __init__(self, dag: WeightedDAG, strategy: NodeExecutionStrategy):
        self.dag = dag
        self.strategy = strategy

        # Final outputs and duration tracking
        self.results: Dict[str, str] = {}
        self.timings: Dict[str, float] = {}
        self.active_tasks = 0
        self.max_parallelism = 0

        # Input aggregation from parents
        self.child_inputs = defaultdict(list)

        # In-degree tracker for each node to determine when it's ready to run
        self.in_degrees = {node: self.dag.graph.in_degree(node) for node in self.dag.graph.nodes}

        # Lock to synchronize access to shared structures
        self.lock = asyncio.Lock()

        # Wall clock tracking
        self.wall_start_time = None
        self.wall_end_time = None

    async def _run_node_task(self, node: str):
        """
        Executes a node:
        - Logs debug info
        - Executes the node task according to the provided strategy
        - Updates timing and triggers successors
        """
        # Update active node performance metrics
        async with self.lock:
            self.active_tasks += 1
            self.max_parallelism = max(self.max_parallelism, self.active_tasks)
            
        start_time = time.perf_counter()
        logger.info(f"🔵 Starting node: {node}")

        # Calculate total incoming duration
        incoming_durations = [
            self.dag.graph.edges[(src, node)]['duration']
            for src in self.dag.graph.predecessors(node)
        ]

        total_duration = sum(incoming_durations)
        
        try:
            result = await self.strategy.execute(node, [], total_duration)
        except Exception as e:
            result = f"ERROR: {e}"
            logger.error(f"❌ Strategy failed for node {node}: {e}")

        # Record result + timing
        end_time = time.perf_counter()
        self.results[node] = result
        self.timings[node] = end_time - start_time
        logger.info(f"✅ Finished node: {node} in {self.timings[node]:.2f} sec\n")

        # Update successors
        async with self.lock:
            self.active_tasks -= 1
            
            for succ in self.dag.graph.successors(node):
                self.in_degrees[succ] -= 1
                self.child_inputs[succ].append(result)
                
                if self.in_degrees[succ] == 0:
                    logger.info(f"🟢 Scheduling successor: {succ}")
                    asyncio.create_task(self._run_node_task(succ))

    async def _wait_until_all_done(self):
        """
        Polls until all DAG nodes have been executed.
        """
        while len(self.results) < len(self.dag.graph.nodes):
            await asyncio.sleep(0.05)


    async def run(self):
        """
        Orchestrates the asynchronous DAG workflow execution.
        """
        self.wall_start_time = time.perf_counter()
        await self._run_node_task(self.dag.source)
        await self._wait_until_all_done()
        self.wall_end_time = time.perf_counter()

    def get_performance_metrics(self) -> Dict:
        """
        Computes performance statistics including:
        - Wall time
        - Sequential time
        - Speedup and time saved
        - Amdahl's alpha estimate
        """
        total_sequential_time = sum(self.timings.values())
        total_wall_time = self.wall_end_time - self.wall_start_time
        time_saved = total_sequential_time - total_wall_time
        speedup = total_sequential_time / total_wall_time if total_wall_time > 0 else float('inf')

        alpha = min(1.0, max(0.0, 1 - (time_saved / total_sequential_time))) if total_sequential_time else 1.0

        return {
            "wall_time": total_wall_time,
            "sequential_time": total_sequential_time,
            "time_saved": time_saved,
            "speedup": speedup,
            "alpha": alpha,
            "critical_path": max(self.timings.values()) if self.timings else 0,
            "node_count": len(self.timings),
            "max_parallelism": self.max_parallelism
        }
