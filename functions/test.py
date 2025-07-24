import pytest
import statistics
from lang_executor import LangGraphDag
from weighted_dag import WeightedDAG
from workflow_executor import WorkflowExecutor
from node_strategy import DurationSleepStrategy, LLMStrategy

import logging

NUM_TRIALS = 5
NUM_NODES = 15
EDGE_PROBABILITY = 0.20

logging.basicConfig(
    filename="workflow_executor.log",
    filemode="w",
    format="%(asctime)s - %(message)s",
)

@pytest.mark.asyncio
@pytest.mark.parametrize("trial", range(NUM_TRIALS))
async def test_dag_executor_performance(trial):
    dag = WeightedDAG(num_nodes=NUM_NODES, edge_probability=EDGE_PROBABILITY)
    executor = WorkflowExecutor(dag, strategy=DurationSleepStrategy())

    await executor.run()
    metrics = executor.get_performance_metrics()

    print(f"Trial {trial+1}: {metrics}")


def test_benchmark_aggregate():
    """
    Run multiple executions in one test and analyze aggregated performance.
    """
    import asyncio

    wall_times = []
    sequential_times = []
    speedups = []
    alphas = []

    for _ in range(NUM_TRIALS):
        dag = WeightedDAG(num_nodes=NUM_NODES, edge_probability=EDGE_PROBABILITY)
        dag.print_summary()
        executor = WorkflowExecutor(dag, strategy=DurationSleepStrategy())
        graph = LangGraphDag(dag.lang_schema)
        graph.save_png()
        asyncio.run(executor.run())
        metrics = executor.get_performance_metrics()

        wall_times.append(metrics["wall_time"])
        sequential_times.append(metrics["sequential_time"])
        speedups.append(metrics["speedup"])
        alphas.append(metrics["alpha"])

    print("\n==== DAG Performance Summary ====")
    print(f"Trials: {NUM_TRIALS}")
    print(f"Avg wall time        : {statistics.mean(wall_times):.4f} sec")
    print(f"Avg sequential time  : {statistics.mean(sequential_times):.4f} sec")
    print(f"Avg speedup          : {statistics.mean(speedups):.2f}x")
    print(f"Avg Amdahl alpha     : {statistics.mean(alphas):.2f}")
