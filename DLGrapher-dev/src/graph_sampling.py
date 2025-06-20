# Source:
# https://github.com/Ashish7129/Graph_Sampling/blob/master/Graph_Sampling/SRW_RWF_ISRW.py
import random
import numpy as np
import networkx as nx


class SRW_RWF_ISRW:

    def __init__(self):
        self.growth_size = 2
        self.T = 100    # number of iterations
        # with a probability (1-fly_back_prob) select a neighbor node
        # with a probability fly_back_prob go back to the initial vertex
        self.fly_back_prob = 0.15

    def random_walk_sampling_simple(self, complete_graph, nodes_to_sample):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                edges_before_t_iter = sampled_graph.number_of_edges()
        return sampled_graph

    def random_walk_sampling_with_fly_back(self, complete_graph, nodes_to_sample, fly_back_prob):
        complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
        # giving unique id to every node same as built-in function id
        for n, data in complete_graph.nodes(data=True):
            complete_graph.nodes[n]['id'] = n

        nr_nodes = len(complete_graph.nodes())
        upper_bound_nr_nodes_to_sample = nodes_to_sample

        index_of_first_random_node = random.randint(0, nr_nodes - 1)
        sampled_graph = nx.Graph()

        sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

        iteration = 1
        edges_before_t_iter = 0
        curr_node = index_of_first_random_node
        while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges) - 1)
            chosen_node = edges[index_of_edge]
            sampled_graph.add_node(chosen_node)
            sampled_graph.add_edge(curr_node, chosen_node)
            choice = np.random.choice(['prev', 'neigh'], 1, p=[fly_back_prob, 1 - fly_back_prob])
            if choice == 'neigh':
                curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((sampled_graph.number_of_edges() - edges_before_t_iter) < self.growth_size):
                    curr_node = random.randint(0, nr_nodes - 1)
                    print("Choosing another random node to continue random walk ")
                edges_before_t_iter = sampled_graph.number_of_edges()

        return sampled_graph

    def random_walk_induced_graph_sampling(self, complete_graph, nodes_to_sample):
        upper_bound_nr_nodes_to_sample = nodes_to_sample
        complete_graph_node_list = list(complete_graph)
        curr_node = random.choice(complete_graph_node_list)

        sampled_nodes = set((curr_node,))

        iteration = 1
        nodes_before_t_iter = 0
        while len(sampled_nodes) != upper_bound_nr_nodes_to_sample:
            edges = [n for n in complete_graph.neighbors(curr_node)]
            chosen_node = random.choice(edges)
            sampled_nodes.add(chosen_node)
            curr_node = chosen_node
            iteration = iteration + 1

            if iteration % self.T == 0:
                if ((len(sampled_nodes) - nodes_before_t_iter) < self.growth_size):
                    curr_node = random.choice(complete_graph_node_list)
                nodes_before_t_iter = len(sampled_nodes)

        sampled_graph = complete_graph.subgraph(sampled_nodes)

        return sampled_graph
