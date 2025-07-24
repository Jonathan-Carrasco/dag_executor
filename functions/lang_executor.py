from typing import Dict
from typing_extensions import TypedDict
from langgraph.graph import START, StateGraph

class State(TypedDict):
    data: int  # Required dummy state field for LangGraph compatibility

class LangGraphDag:
    """
    Converts a DAG schema into a LangGraph-compatible StateGraph.
    This mainly exists now only for printing dags to png.
    """

    def __init__(self, graph_data: Dict) -> None:
        """
        Initialize LangGraphDag from a precomputed DAG schema.
        
        Args:
            graph_data: A dictionary with keys `nodes`, `edges`, `durations`, and `entry_point`,
                        typically produced by WeightedDAG.to_langgraph_schema().
        """
        self.graph_data = graph_data  # Expected to be in LangGraph schema format
        self.builder = StateGraph(State)
        self._build()

    def _build(self) -> None:
        """
        Build the LangGraph StateGraph based on the input DAG schema.
        Steps:
            - Add a dummy update function to each DAG node.
            - Wire the LangGraph edges to match the DAG topology.
            - Define the entry point and compile the graph.
        """
        # Add LangGraph nodes
        for node in self.graph_data["nodes"]:
            self.builder.add_node(node, lambda x : {"data": x})

        # Set entry point
        self.builder.add_edge(START, self.graph_data["source"])

        # Add directed edges from DAG
        for from_node, to_node in self.graph_data["edges"]:
            self.builder.add_edge(from_node, to_node)

        # Compile LangGraph
        self.app = self.builder.compile()
        self.graph = self.app.get_graph()

    def save_png(self, filename: str = "dag_graph.png") -> None:
        """
        Save a PNG visualization of the compiled graph.
        
        Args:
            filename: Path to the output PNG file.
        """
        png_data = self.graph.draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")