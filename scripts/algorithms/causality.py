import os
from typing import Tuple, List
import re
import networkx as nx
import pandas as pd
import random
import numpy as np
from causalnex.inference import InferenceEngine
from causalnex.network import BayesianNetwork
from causalnex.structure import StructureModel
from causalnex.structure.pytorch import from_pandas
from matplotlib import pyplot as plt
import json
from path_repo import GLOBAL_PATH_REPO

FONT_SIZE_NODE_GRAPH = 7
ARROWS_SIZE_NODE_GRAPH = 30
NODE_SIZE_GRAPH = 1000


class CausalDiscovery:
    def __init__(self, df, env_name, bins_discretization=10):
        past_df = df

        random.seed(42)
        np.random.seed(42)

    def add_data(self):
        self.df = df
        self.env_name = env_name
        self.bins = bins_discretization
        self.dir_save = f'{GLOBAL_PATH_REPO}/causal_knowledge/{self.env_name}'
        os.makedirs(self.dir_save, exist_ok=True)
        self.df.to_pickle(f'{self.dir_save}/df_original.pkl')
        self.df = self.discretize_continuous_features(self.df, self.bins)
        self.df.to_pickle(f'{self.dir_save}/df_discretized{self.bins}bins.pkl')
        self.features_names = self.df.columns.to_list()

        for col in self.df.columns:
            self.df[str(col)] = self.df[str(col)].astype(str).str.replace(',', '').astype(float)


    @staticmethod
    def discretize_continuous_features(df, bins=10, unique_threshold=20):
        """
        Discretize continuous features in the dataframe using qcut.

        Parameters:
        - df: pandas DataFrame
        - bins: int, number of bins to discretize into
        - unique_threshold: int, threshold for unique values to consider a feature continuous

        Returns:
        - df: pandas DataFrame with discrete continuous features
        """
        df_copy = df.copy()  # Make a copy to avoid modifying the original dataframe

        for column in df_copy.columns:
            # Check if the feature is numeric and has more unique values than the threshold
            if pd.api.types.is_numeric_dtype(df_copy[column]) and df_copy[column].nunique() > unique_threshold:
                df_copy[column] = pd.cut(df_copy[column], bins=bins, labels=False)

        return df_copy

    def training(self):

        print(f'{self.env_name} - structuring model through NOTEARS... {len(self.df)} timesteps')
        self.notears_graph = from_pandas(self.df, max_iter=500, use_gpu=True)
        self.notears_graph.remove_edges_below_threshold(0.2)
        self._plot_and_save_graph(self.notears_graph, False)

        if nx.number_weakly_connected_components(self.notears_graph) == 1 and nx.is_directed_acyclic_graph(
                self.notears_graph):

            print('do-calculus-1...')
            # assessment of the no-tears graph
            causal_relationships, _, _ = self._causality_assessment()

            sm = StructureModel()
            sm.add_edges_from([(node1, node2) for node1, node2 in causal_relationships])
            self.causal_graph = sm

            self._plot_and_save_graph(self.causal_graph, True)

            if_causal_graph_DAG = nx.is_directed_acyclic_graph(self.causal_graph)
            if not if_causal_graph_DAG:
                print('**** Causal graph is not a DAG ****')

        else:
            self.causal_graph = None
            print(f'Number of graphs: {nx.number_weakly_connected_components(self.notears_graph)},'
                  f' DAG: {nx.is_directed_acyclic_graph(self.notears_graph)}')

    def _causality_assessment(self) -> Tuple[List[Tuple], List, List]:
        """ Given an edge, do-calculus on each direction once each other (not simultaneously) """

        bn = BayesianNetwork(self.notears_graph)
        bn = bn.fit_node_states_and_cpds(self.df)

        bad_nodes = [node for node in bn.nodes if not re.match("^[0-9a-zA-Z_]+$", node)]
        if bad_nodes:
            print('Bad nodes: ', bad_nodes)

        ie = InferenceEngine(bn)

        # Initial assumption: all nodes are independent until proven dependent
        independent_vars = set(self.notears_graph.nodes)
        dependent_vars = set()
        causal_relationships = []

        # Precompute unique values for all nodes
        unique_values = {node: self.df[node].unique() for node in self.notears_graph.nodes}

        # Initial query to get the baseline distributions
        before = ie.query()

        # Iterate over each node in the graph
        for node in self.notears_graph.nodes:
            connected_nodes = list(self.notears_graph.neighbors(node))
            change_detected = False

            # Perform interventions on the node and observe changes in all connected nodes
            for value in unique_values[node]:
                try:
                    ie.do_intervention(node, int(value))
                    after = ie.query()

                    # Check each connected node for changes in their distribution
                    for conn_node in connected_nodes:
                        best_key_before, max_value_before = max(before[conn_node].items(), key=lambda x: x[1])
                        best_key_after, max_value_after = max(after[conn_node].items(), key=lambda x: x[1])
                        uniform_probability_value = round(1 / len(after[conn_node]), 8)

                        if max_value_after > uniform_probability_value and best_key_after != best_key_before:
                            dependent_vars.add(conn_node)  # Mark as dependent
                            if conn_node in independent_vars:
                                independent_vars.remove(conn_node)  # Remove from independents
                                print(f"Link removed: {node} -> {conn_node}")
                            change_detected = True
                            causal_relationships.append((node, conn_node))  # Ensure this is a 2-tuple

                    # Also check the intervened node itself
                    best_key_before, max_value_before = max(before[node].items(), key=lambda x: x[1])
                    best_key_after, max_value_after = max(after[node].items(), key=lambda x: x[1])
                    uniform_probability_value = round(1 / len(after[node]), 8)

                    if max_value_after > uniform_probability_value and best_key_after != best_key_before:
                        change_detected = True

                except Exception as e:
                    # Log the error
                    print(f"Error during intervention on {node} with value {value}: {str(e)}")
                    pass

            if change_detected:
                dependent_vars.add(node)  # Mark as dependent
                if node in independent_vars:
                    independent_vars.remove(node)  # Remove from independents
                    print(f"Link removed: {node}")

        return causal_relationships, list(independent_vars), list(dependent_vars)

    def return_causal_graph(self):
        if self.causal_graph is not None:
            structure_to_return = [(x[0], x[1]) for x in self.causal_graph.edges]
            return structure_to_return
        else:
            return None

    def _plot_and_save_graph(self, sm: StructureModel, if_causal: bool):

        import warnings
        warnings.filterwarnings("ignore")

        fig = plt.figure(dpi=1000)
        if if_causal:
            plt.title(f'Causal graph - {len(sm)} nodes - {len(sm.edges)} edges', fontsize=16)
        else:
            plt.title(f'NOTEARS graph - {len(sm)} nodes - {len(sm.edges)} edges', fontsize=16)

        nx.draw(sm, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
                arrowsize=ARROWS_SIZE_NODE_GRAPH if if_causal else 0,
                arrows=if_causal,
                edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold', node_color='skyblue',
                pos=nx.circular_layout(sm))
        plt.show()

        structure_to_save = [(x[0], x[1]) for x in sm.edges]

        if if_causal:
            plt.savefig(f'{self.dir_save}/causal_graph.png')

            with open(f'{self.dir_save}/causal_structure.json', 'w') as json_file:
                json.dump(structure_to_save, json_file)
        else:
            plt.savefig(f'{self.dir_save}/notears_graph.png')

            with open(f'{self.dir_save}/notears_structure.json', 'w') as json_file:
                json.dump(structure_to_save, json_file)

        plt.close(fig)


def define_causal_graph(list_for_causal_graph: list):
    def plot_causal_graph(G):
        fig = plt.figure(dpi=800)
        plt.title(f'Causal graph - {len(sm)} nodes - {len(sm.edges)} edges', fontsize=16)

        nx.draw(G, with_labels=True, font_size=FONT_SIZE_NODE_GRAPH,
                arrowsize=ARROWS_SIZE_NODE_GRAPH,
                arrows=True,
                edge_color='orange', node_size=NODE_SIZE_GRAPH, font_weight='bold',
                pos=nx.circular_layout(sm))
        plt.show()

    # Create a StructureModel
    sm = StructureModel()

    # Add edges to the StructureModel
    for relationship in list_for_causal_graph:
        cause, effect = relationship
        sm.add_edge(cause, effect)

    # plot_causal_graph(sm)

    return sm


if __name__ == "__main__":
    x = pd.read_pickle('C:\\Users\giova\Documents\Research\cdrl_framework\story_env_0.pkl')
    x.rename(columns={'reward_agent_0': 'agent_0_reward'}, inplace=True)
    x.rename(columns={'action_agent_0': 'agent_0_action'}, inplace=True)


    # x.drop('agent_0_sensor5', axis=1, inplace=True)

    def split_dataframe_by_agents(df):
        """
        Split dataframe into separate dataframes for each agent based on column names.

        Parameters:
        - df: pandas DataFrame

        Returns:
        - dict: Dictionary of dataframes split by agents
        """
        agent_dfs = {}
        agents = set(col.split('_')[1] for col in df.columns if col.startswith('agent'))
        for agent in agents:
            agent_columns = [col for col in df.columns if
                             f"_{agent}_" in col or col.startswith(f"reward_{agent}") or col.startswith(
                                 f"action_{agent}")]
            agent_dfs[agent] = df[agent_columns].copy()

        return agent_dfs


    agents = split_dataframe_by_agents(x)
    df = agents['0']
    df = df.loc[:, df.std() != 0]

    cd = CausalDiscovery(df, 'navigation')
    cd.training()
    causal_graph = cd.return_causal_graph()
    print(causal_graph)
