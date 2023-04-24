import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import networkx as nx
import numpy as np


class Single_Neuron_Node:
    def __init__(self, type, inputs, outputs):
        self.type = type
        self.inputs = inputs
        self.outputs = outputs
        self.activated = False

    def forward(self, x):
        pass

class Random_RNN(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    input_weights: Tensor
    associative_weights: Tensor


    def get_connections_num(self):
        """
        Calculate the number of 'out' connections
         for INPUT and ASSOCIATIVE neurons.
        :return: Dictionary {input, associative} number
         of 'out' connections for INPUT and ASSOCIATIVE neurons.
        """
        associative_start_index = self.in_features
        output_start_index = self.in_features + self.associative

        connections = {'input': 0, 'associative': 0}

        for i in range(self.in_features):
                    connections['input'] += len(self.graph.edges(i))

        for i in range(associative_start_index, output_start_index):
                    connections['associative'] += len(self.graph.edges(i))

        return connections


    def arrange_weights(self):
        """
        Simply arranges all model weights on the
         graph edges. It references the weights and put
         them as an attributes of edges.
        :return: None
        """
        input_w_counter = 0
        ass_w_counter = 0

        for node in self.graph.nodes(data=True):
            for edge in self.graph.edges(node[0], data=True):
                if node[1]["type"] == 'input':
                    edge[2]["weight"] = self.input_weights[input_w_counter]
                    input_w_counter += 1

                elif node[1]["type"] == 'associative':
                    edge[2]["weight"] = self.associative_weights[ass_w_counter]
                    ass_w_counter += 1


    def random_graph_init(self):
        """
        Randomly initialize graph structure of the network based on the number of neurons and connection percentage.
        :return: None
        """

        # --------------------- NEURONS INITIALIZATION ---------------------

        associative_start_index = self.in_features
        output_start_index = self.in_features + self.associative

        neurons = []

        def test_forward(x):
            return x

        input_neuron_data = {"type": "input",
                             "activated": False,
                             "forward": test_forward,
                             'color': 'lightblue',
                             'layer': 0}
        output_neuron_data = {"type": "output",
                              "activated": False,
                              "forward": test_forward,
                              'color': 'lightgreen',
                              'layer': 2}
        associative_neuron_data = {"type": "associative",
                                   "activated": False,
                                   "forward": test_forward,
                                   'color': 'gold',
                                   'layer': 1}

        # Generate input, output and associative (between input and output) population of neurons
        for i in range(self.in_features):
            neurons.append((i, input_neuron_data))
        for i in range(self.associative):
            neurons.append((associative_start_index + i, associative_neuron_data))
        for i in range(self.out_features):
            neurons.append((output_start_index + i, output_neuron_data))

        # Initialize all nodes of the graph
        self.graph.add_nodes_from(neurons)


        # --------------------- EDGES INITIALIZATION ---------------------
        edges = []
        in_connected_neurons = []

        # --- Generate edges for INPUT neurons
        for input_index in range(0, associative_start_index):
            req_out_connection = False

            for ass_index in range(associative_start_index, output_start_index):

                # Create a connection with probability
                if np.random.random() < self.connect_percentage:
                    edges.append((input_index, ass_index, {"weight": None}))
                    in_connected_neurons.append(ass_index)
                    req_out_connection = True

            # If no connection created, randomly chose one associative neuron and create a connection
            if not req_out_connection:
                ass_index = np.random.randint(associative_start_index, output_start_index)
                edges.append((input_index, ass_index, {"weight": None}))


        # --- Generate edges for ASSOCIATIVE neurons
        for ass_index in range(associative_start_index, output_start_index):
            req_out_connection = False

            # Enumerate over associative (for recurrent connections) and output neurons
            for connection_index in range(associative_start_index, self.total_neurons):

                # Create a connection with probability
                if np.random.random() < self.connect_percentage:
                    edges.append((ass_index, connection_index, {"weight": None}))

                    # Recurrent relation is not considered as 'in' connection
                    if ass_index != connection_index:
                        in_connected_neurons.append(connection_index)
                        req_out_connection = True

            # If no 'out' connection created, randomly chose one associative or output neuron and create a connection
            if not req_out_connection:

                # Iterate until get not recurrent connection
                while True:
                    connection_index = np.random.randint(associative_start_index, self.total_neurons)
                    if ass_index != connection_index:
                        edges.append((ass_index, connection_index, {"weight": None}))
                        break

            # If no 'in' connection, randomly chose one input or associative neuron and create a connection
            if ass_index not in in_connected_neurons:

                # Iterate until get not recurrent connection
                while True:
                    connection_index = np.random.randint(0, output_start_index)
                    if ass_index != connection_index:
                        edges.append((connection_index, ass_index, {"weight": None}))
                        break


        # --- Generate edges for OUTPUT neurons
        for output_index in range(output_start_index, self.total_neurons):

            if output_index not in in_connected_neurons:
                connection_index = np.random.randint(associative_start_index, output_start_index)
                edges.append((connection_index, output_index, {"weight": None}))


        # Finally, load edges into the graph
        self.graph.add_edges_from(edges)



    def __init__(self, in_features, out_features, neurons, connect_percentage, device=None, dtype=None):
        """
        :param in_features: Number of input features. This parameter would affect on the amount of neurons in the input layer.
        :param out_features: Number of input features. This parameter would affect on the amount of neurons in the output layer.
        :param neurons: Number of neurons in the chaotic graph (number of nodes).
        :param connect_percentage: The percentage of connections would be generated, where 1.0 is fully connected graph/network and 0.01 is 1% of all possible connections in graph.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Random_RNN, self).__init__()
        self.graph = nx.DiGraph()
        self.in_features = in_features
        self.out_features = out_features
        self.associative = neurons
        self.connect_percentage = connect_percentage
        self.total_neurons = neurons + in_features + out_features

        # Randomly initialize graph structure
        self.random_graph_init()

        # Initialize weights for input and associative neurons as two vectors
        conn_num = self.get_connections_num()
        self.input_weights = Parameter(torch.empty((conn_num["input"], 1), **factory_kwargs))
        self.associative_weights = Parameter(torch.empty((conn_num["associative"], 1), **factory_kwargs))
        self.arrange_weights()




    def forward(self, x):
        # Initialize output tensor
        output = torch.zeros((x.shape[0], self.out_features), device=x.device)

        # forward all input neurons
        for i in range(0, self.in_features):
            pass
