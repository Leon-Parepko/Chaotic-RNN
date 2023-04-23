import math
import torch
import torch.nn as nn
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

    def random_graph_init(self):

        # --------------------- NEURONS INITIALIZATION ---------------------

        # Generate input, output and associative (between input and output) population of neurons
        input_neurons = []
        associative_neurons = []
        output_neurons = []
        for i in range(self.in_features):
            input_neurons.append((i, {"type": "input", "activated": False, "forward": None}))

        for i in range(self.associative):
            associative_neurons.append((i+self.in_features, {"type": "associative", "activated": False, "forward": None}))

        for i in range(self.out_features):
            output_neurons.append((i + self.in_features + self.associative, {"type": "output", "activated": False, "forward": None}))

        # Initialize all nodes of the graph
        self.graph.add_nodes_from(input_neurons + associative_neurons + output_neurons)


        # --------------------- EDGES INITIALIZATION ---------------------
        edges = []
        percentile_connections = math.ceil(self.total_neurons * self.connections)   # User choosed amount of maximum connections for Input neurons



        # Generate edges for INPUT neurons
        for i in range(self.in_features):

            # For each neuron track its possible connections
            possible_connections = list(range(0, self.total_neurons))

            # Firstly generate minimum required connection i'th input neuron.
            # It could be Input -> Associative, Input -> Output or Input -> Input (but not to itself).
            connection_index = 0
            while True:
                connection_index = np.random.randint(0, self.total_neurons - 1)
                if connection_index != i:
                    break
            possible_connections.remove(connection_index)
            edges.append((i, connection_index))

            # Select not repeating random connections for i'th input neuron in given connections percentile.
            # It could be Input -> Associative, Input -> Output or Input -> Input (Including itself).
            # Hence, the maximum amount of connections for each input neuron is total_neurons.
            for j in range(percentile_connections):
                connection_index = np.random.choice(possible_connections)
                possible_connections.remove(connection_index)
                edges.append((i, connection_index))



       #  # Generate edges for ASSOCIATIVE neurons
       #  for i in range(self.associative):
       #      i += self.in_features
       #
       #      # Check for minimum required connections of i'th associative neuron.
       #      # Should have 1 associative (non-recurrent) or 1 input input connection.
       #      # Firstly search in existing edges
       #      requirement_1 = False
       #      for edge in edges:
       #          if edge[1] == i:
       #              requirement_1 = True
       #              break
       #      # If there is no already existing connection, create it from random input or associative neuron.
       #      if not requirement_1:
       #          presynaptic_neuron_index = 0
       #          while True:
       #              presynaptic_neuron_index = np.random.randint(0, self.in_features + self.associative - 1)
       #              if presynaptic_neuron_index != i:
       #                  break
       #          edges.append((presynaptic_neuron_index, i))
       #
       #      # Should have 1 associative (non-recurrent) or 1 input or one output output connection.
       #      possible_connections = list(range(0, self.total_neurons))
       #
       #      connection_index = 0
       #      while True:
       #          connection_index = np.random.randint(0, self.total_neurons - 1)
       #          if connection_index != i:
       #              break
       #      possible_connections.remove(connection_index)
       #      edges.append((i, connection_index))
       #
       #      # Select not repeating random connections for i'th associative neuron in given connections percentile.
       #      # It could be Associative -> Associative (Including itself), Associative -> Output or Associative -> Input.
       #      # Hence, the maximum amount of connections for each Associative neuron is total_neurons.
       #      for j in range(percentile_connections):
       #          connection_index = np.random.choice(possible_connections)
       #          possible_connections.remove(connection_index)
       #          edges.append((i, connection_index))
       #
       #
       #
       # # Generate edges for OUTPUT neurons
       #  for i in range(self.out_features):
       #      i += self.in_features + self.associative
       #
       #
       #
        self.graph.add_edges_from(edges)



    def __init__(self, in_features, out_features, neurons, connections, device=None, dtype=None):
        """
        :param in_features: Number of input features. This parameter would affect on the amount of neurons in the input layer.
        :param out_features: Number of input features. This parameter would affect on the amount of neurons in the output layer.
        :param neurons: Number of neurons in the chaotic graph (number of nodes).
        :param connections: The percentage of connections would be generated, where 1.0 is fully connected graph/network and 0.01 is 1% of all possible connections in graph.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}

        super(Random_RNN, self).__init__()
        self.graph = nx.Graph()                                                                 # Randomly initialize graph strucure
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.associative = neurons
        self.connections = connections
        self.total_neurons = neurons + in_features + out_features
        self.v = torch.zeros(in_features, **factory_kwargs)  # Initial membrane potential -70mv ~ 0
        self.threshold = 20  # Threshold value -50mv ~ 20

        # Randomly initialize graph structure
        self.random_graph_init()



    def forward(self, x, t):
        # init spikes as zeros
        spikes = torch.zeros(self.in_features)

        # init pre spike history as zeros if it is none
        if self.spike_history["pre"] is None:
            self.spike_history["pre"] = torch.zeros(x.shape[1])

        # Update pre history if any pre spikes
        for i in range(x.shape[1]):
            if x[0][i] > 0:
                self.spike_history["pre"][i] = 1

            elif self.spike_history["pre"][i] >= 1:
                self.spike_history["pre"][i] += 1

        # Weighted normalized sum of x for each neuron
        sum = torch.sum(x, 1)

        # Leak
        self.v *= math.exp(
            -0.01 / self.tau)  # Analytical solution to the differential equation dv/dt = -1/tau * (v - v_0)
        # Integrate
        self.v += sum * 10  # maximum 10mv per spike
        # Fire
        for i in range(0, self.v.shape[0]):
            if self.v[i] >= self.threshold:  # Check if any neuron has fired
                spikes[i] = 1
                self.v[i] = 0

        # Update spike history
        self.spike_history["cur"] = spikes

        # Normalize output
        return (spikes * self.weight) / self.in_features






    def history_update(self, post_spikes=None):
        if post_spikes is not None:
            for i, post_spike in enumerate(post_spikes):
                # If post spike, reset time for post spike in history
                if post_spike > 0:
                    self.spike_history["post"][i] = 1

                # Else increase time for post spike in history
                elif self.spike_history["post"][i] >= 1:
                    self.spike_history["post"][i] += 1
        return self.spike_history["cur"]



    def get_membrane_potential(self):
        return self.v

    def get_spike_history(self):
        return self.spike_history

    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()

    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1


class TemporalEncoder(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TemporalEncoder, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.spike_history = SpikeHistory(post=torch.zeros(out_features, **factory_kwargs))

    def forward(self, x, t):
        # Initialize the output
        spikes = torch.zeros(self.in_features)

        # Normalize input and set it from 0 to 255
        x = (x / torch.max(x)) * 255

        # Discretize input using tau
        x = torch.floor(x / self.tau)

        for i in range(x.shape[1]):
            if torch.tensor(t, dtype=torch.int32) == torch.tensor(x, dtype=torch.int32)[0][i]:
                spikes[i] = 1

        # Update spike history
        self.spike_history["cur"] = spikes
        return self.weight * spikes


    def history_update(self, post_spikes):
        for i, post_spike in enumerate(post_spikes):
            # If post spike, reset time for post spike in history
            if post_spike > 0:
                self.spike_history["post"][i] = 1

            # Else increase time for post spike in history
            elif self.spike_history["post"][i] >= 1:
                self.spike_history["post"][i] += 1


    def neuron_states(self):
        return self.spikes

    def get_spike_history(self):
        return self.spike_history

    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()

    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1


class RateEncoder(nn.Module):
    def __init__(self, in_features, out_features, tau=1, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(RateEncoder, self).__init__()
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.weight.data.uniform_(0, 1)
        self.in_features = in_features
        self.out_features = out_features
        self.tau = tau
        self.spike_history = SpikeHistory(post=torch.zeros(out_features, **factory_kwargs))

    def forward(self, x, t):
        # Initialize the output
        spikes = torch.zeros(self.in_features)

        # Normalize input and set it from 0 to 255
        x = (x / torch.max(x)) * 255

        # Discretize input using tau
        x = torch.floor(x / self.tau)

        for i in range(x.shape[1]):
            if torch.tensor(t, dtype=torch.int32) == torch.tensor(x, dtype=torch.int32)[0][i]:
                spikes[i] = 1

        # Update spike history
        self.spike_history["cur"] = spikes
        return self.weight * spikes


    def history_update(self, post_spikes):
        for i, post_spike in enumerate(post_spikes):
            # If post spike, reset time for post spike in history
            if post_spike > 0:
                self.spike_history["post"][i] = 1

            # Else increase time for post spike in history
            elif self.spike_history["post"][i] >= 1:
                self.spike_history["post"][i] += 1


    def neuron_states(self):
        return self.spikes

    def get_spike_history(self):
        return self.spike_history

    def weight_reset(self):
        with torch.no_grad():
            self.weight.zero_()

    def weight_to_one(self):
        with torch.no_grad():
            for i in range(0, self.weight.shape[0]):
                for j in range(0, self.weight.shape[1]):
                    self.weight[i][j] = 1
