from collections import defaultdict
import json
import random
from typing import Dict
import networkx as nx


class WeightedDAG:
    """
    A class to generate a connected Directed Acyclic Graph (DAG) using NetworkX.
    - Each edge has a random duration (weight).
    - The graph is guaranteed to be a single connected component from the source node.
    """

    def __init__(self, num_nodes: int = 15, edge_probability: float = 0.3):
        self.num_nodes: int = num_nodes
        self.edge_probability: float = edge_probability
        self.graph: nx.DiGraph = self._generate_connected_dag()
        self.source: str = self._get_source_node()
        self.lang_schema: Dict = self._to_langgraph_schema()
        self.schema: Dict = self._to_schema()


    def _generate_connected_dag(self) -> nx.DiGraph:
        """
        Generates a connected DAG with a topological order.
        Ensures that all nodes are reachable from at least one root (source).
        """
        G = nx.DiGraph()
        
        # Step 1: Create a topological ordering of nodes and add them to graph
        ordering = list(range(self.num_nodes))
        random.shuffle(ordering)
        G.add_nodes_from(ordering)

        # Step 2: Add random edges respecting the topological ordering
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if random.random() < self.edge_probability:
                    G.add_edge(ordering[i], ordering[j], duration=random.randint(1, 10))

        # Step 3: Ensure all nodes are reachable (connected DAG)
        reachable = set(nx.descendants(G, ordering[0])) | {ordering[0]}
        unreachable = set(G.nodes()) - reachable

        for node in unreachable:
          inserted = False
          for target in reachable:
              G.add_edge(node, target, duration=random.randint(1, 10))
              if nx.is_directed_acyclic_graph(G):
                  inserted = True
                  break
              G.remove_edge(node, target)

              G.add_edge(target, node, duration=random.randint(1, 10))
              if nx.is_directed_acyclic_graph(G):
                  inserted = True
                  break
              G.remove_edge(target, node)

          if inserted:
              reachable.add(node)

        return nx.relabel_nodes(G, lambda n: f"node_{n}")


    def _get_source_node(self) -> str:
      """
      Identifies the entry point (source) of the DAG.
      If multiple sources exist, a virtual root node is added.
      """
      sources = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]

      if len(sources) == 1:
        return sources[0]
      
      # Add a virtual source node that connects to all existing sources
      for source in sources:
        self.graph.add_edge("virtual", source, duration=random.randint(1, 10))
      
      return "virtual"


    def _to_langgraph_schema(self) -> Dict:
        """
        Converts the graph into a format suitable for LangGraph use.
        Includes:
        - List of nodes
        - List of edges
        - Durations for each edge
        - Entry point
        """
        return {
            "nodes": list(self.graph.nodes),
            "edges": [(u, v) for u, v in self.graph.edges],
            "durations": {(u, v): self.graph.edges[u, v]['duration'] for u, v in self.graph.edges},
            "source": self.source
        }
    
    
    def _to_schema(self) -> Dict:
      """
      Converts the graph into an adjacency list format with durations.
      Example:
      {
          "node_1": [("node_3", 5), ("node_7", 2)],
          ...
      }
      """
      schema = defaultdict(list)
      for u, v in self.graph.edges:
          schema[u].append((v, self.graph.edges[u, v]['duration']))
      return dict(schema)


    def save_to_file(self, filename: str = "dag.json", to_lang: bool = False) -> None:
        """
        Saves the graph schema to a JSON file.
        If `to_lang` is True, saves in LangGraph format.
        """
        with open(filename, "w") as f:
            json.dump(self.lang_schema if to_lang else self.schema, f, indent=2)

    
    def print_summary(self) -> None:
        """
        Prints a summary of the graph structure in adjacency list format.
        """
        for u, edges in self.schema.items():
            print(f"{u}: {list(edges)}")