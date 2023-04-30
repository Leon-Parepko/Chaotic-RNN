import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import networkx as nx
import numpy as np

class ChaoticRNN(nn.Module):
    r"""
    Applies a custom chaotic recurrent layer (Elman RNN but with
     reduced number of connections) over an input signal.
     It generates a graph with a given number of neurons and connections
     between them with a given probability.

     Args:
        :param in_features: Number of input features (input neurons).
        :param out_features: Number of output features (output neurons).
        :param neurons: Number of associative neurons (hidden neurons).
        :param connect_percentage: Percentage of connections between
         neurons (0.0 - 1.0). Here the vqlue coud be interpreted as a
         probability of connection between neurons.
        :param activation: Activation function of the network.
        :param device: Device to run the network.
        :param dtype: Data type of the network.

    Examples::
        >>> m = ChaoticRNN(10, 50, 4, 0.3)
        >>> input = torch.randn(1, 10)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([50])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    associative: int
    out_features: int
    input_weights: Tensor
    associative_weights: Tensor
    graph: nx.DiGraph
    activation: callable
    connect_percentage: float

    #TODO: Add type of data forwarding (one -> one, one -> many, many -> one, many -> many)

    def __init__(self, in_features, out_features, neurons, connect_percentage, activation=torch.tanh, device='cpu', dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(ChaoticRNN, self).__init__()
        self.device = device
        self.activation = activation
        self.in_features = in_features
        self.associative = neurons
        self.out_features = out_features
        self.connect_percentage = connect_percentage
        self.total_neurons = neurons + in_features + out_features

        # Randomly initialize graph structure until it is stable
        while True:
            self.graph = nx.DiGraph()
            self.__random_graph_init()
            if self.propagation_check():
                break

        # Initialize weights for input and associative neurons as two vectors
        conn_num = self.__get_connections_num()
        self.input_weights = Parameter(torch.empty((conn_num["input"], 1), **factory_kwargs))
        self.associative_weights = Parameter(torch.empty((conn_num["associative"], 1), **factory_kwargs))
        self.__arrange_weights()
        self.__init_weights()


    def to(self, *args, **kwargs):
        """
        Override the default to() method to set the device
        :return: self
        """
        self = super().to(*args, **kwargs)
        self.device = args[0]
        return self


    def forward(self, x, return_work_time=False):
        """
        Forward pass of the network. Here each neuron in a graph
         might have 3 states (inactive, working, activated). The network
         is working until all neurons are activated.
        :param x: Input tensor
        :param return_work_time: If True, the function returns the work time of the network
        :return: Output tensor
        """



        # Reset all neurons to be not activated
        self.reset_neurons_states()
        # self.reset_neurons_memory()

        # Increase dimensions of the input tensor
        X = x.clone()[0].unsqueeze(1)
        Y = torch.zeros((self.out_features, ))

        # Forward all input neurons
        for i, (neuron_index, data) in enumerate(self.graph.nodes(data=True)):
            if data["type"] == 'input':
                forward_fn = data["forward"]

                # Get all edges of the current neuron
                for edge in self.graph.edges(neuron_index, data=True):
                    weight = edge[2]["weight"]
                    conn_neuron = self.graph.nodes(data=True)[edge[1]]
                    conn_neuron["memory"].append(forward_fn(X[i], weight))

                    # Activate all connected neurons to working state
                    self.__activate_neuron(edge[1], working=True)

                # Set the current neuron as activated (skipping working state)
                self.__activate_neuron(neuron_index, working=False)



        # Forward all associative neurons until all neurons are activated
        ass_active = [False for _ in range(self.associative)]
        work_time = 1
        max_work_time = self.associative + 2

        while not all(ass_active):
            for neuron_index, data in self.graph.nodes(data=True):
                if data["type"] == 'associative' and data["status"] == "working":
                    forward_fn = data["forward"]
                    memory = data["memory"]

                    # Get all edges of the current neuron
                    for edge in self.graph.edges(neuron_index, data=True):
                        weight = edge[2]["weight"]
                        conn_neuron = self.graph.nodes(data=True)[edge[1]]

                        # Send signal to the memory
                        conn_neuron["memory"].append(forward_fn(memory, weight))

                        # Activate connected neuron to working state if it is inactive
                        if conn_neuron["status"] == "inactive":
                            self.__activate_neuron(edge[1], working=True)

                    self.__activate_neuron(neuron_index, working=False)
                    ass_active[neuron_index - self.in_features] = True

            work_time += 1

            # If the work time excite maximum possible, return zeros Y and +inf work time
            if work_time >= max_work_time:
                # print("WORK TIME ERROR!!!!")
                return (Y, float("inf")) if return_work_time else Y

        # Forward all output neurons and write the result
        i = 0
        for neuron_index, data in self.graph.nodes(data=True):
            if data["type"] == 'output':
                forward_fn = data["forward"]
                memory = data["memory"]
                Y[i] = forward_fn(memory, torch.tensor([1]))

                self.__activate_neuron(neuron_index, working=False)
                i += 1

        # Check if all neurons are activated
        # for i, data in self.graph.nodes(data=True):
        #     if data['status'] != 'activated':
        #         print(f"ERROR: {i} neuron - {data}")

        return (Y, work_time) if return_work_time else Y


    def restruct(self, connect_percentage=None):
        """
        Restructure the network by randomly initializing
         the graph structure and weights until it is stable.
        :param connect_percentage: Percentage of connections in the graph
         if it is None, the default value is used from the constructor.
        :return: None
        """
        if connect_percentage is not None:
            self.connect_percentage = connect_percentage

        while True:
            self.graph = nx.DiGraph()
            self.__random_graph_init()
            if self.propagation_check():
                break
        conn_num = self.__get_connections_num()
        self.input_weights = Parameter(torch.empty((conn_num["input"], 1)))
        self.associative_weights = Parameter(torch.empty((conn_num["associative"], 1)))
        self.__arrange_weights()
        self.__init_weights()
        return


    def reset_neurons_memory(self):
        """
        Reset all neurons to have empty memory
        :return: none
        """
        for i, data in self.graph.nodes(data=True):
            self.graph.nodes[i]["memory"] = []
        return


    def reset_neurons_states(self):
        """
        Reset all neurons to be inactive.
        :return: none
        """
        for i, data in self.graph.nodes(data=True):
            self.graph.nodes[i]["status"] = "inactive"
        return


    def propagation_check(self):
        """
        Check if the network is able to propagate the signal from input to output
        :return: True if the network is able to propagate the signal, False otherwise
        """
        X_test = torch.ones(self.in_features).view(-1, self.in_features)
        if self.forward(X_test, return_work_time=True)[1] != float("inf"):
            return True

        else:
            return False


    def __activate_neuron(self, index, working=True):
        """
        Activate the neuron with the given index.
         If working is True, the neuron is set to
         working state (memory is not erased). If working
         is False, the neuron is set to activated state
         and memory is erased.
        :param index: Index of the neuron to activate
        :param working: If True, the neuron is set to working state
        :return: none
        """
        neuron = self.graph.nodes[index]

        if working:
            neuron["status"] = "working"

        else:
            neuron["status"] = "activated"
            neuron["memory"] = []

        return


    def __get_connections_num(self):
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


    def __arrange_weights(self):
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
                    ass_w_counter += 1\


    def __init_weights(self):
        """
        Initialize all weights in the network
        as random values from uniform distribution.
        :return: None
        """
        nn.init.uniform_(self.input_weights, -1, 1)
        nn.init.uniform_(self.associative_weights, -1, 1)
        return


    def __random_graph_init(self):
        """
        Randomly initialize graph structure
         of the network based on the number of
         neurons and connection percentage.
        :return: None
        """

        # --------------------- NEURONS INITIALIZATION ---------------------
        associative_start_index = self.in_features
        output_start_index = self.in_features + self.associative

        neurons = []

        def simple_forward(x, w):
            """
            Function used for forwarding each neuron.
            :param x: Input array of signals.
            :param w: Weight.
            :return: Output of a neuron (single value).
            """
            w = w.clone()
            return self.activation(w.to(self.device) * sum(x).to(self.device))

        input_neuron_data = {"type": "input",
                             "status": "inactive",
                             "forward": simple_forward,
                             "memory": [],
                             'color': 'lightblue',
                             'layer': 0}
        output_neuron_data = {"type": "output",
                              "status": "inactive",
                              "memory": [],
                              "forward": simple_forward,
                              'color': 'lightgreen',
                              'layer': 2}
        associative_neuron_data = {"type": "associative",
                                   "status": "inactive",
                                   "memory": [],
                                   "forward": simple_forward,
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
                    edges.append((input_index, ass_index))
                    in_connected_neurons.append(ass_index)
                    req_out_connection = True

            # If no connection created, randomly chose one associative neuron and create a connection
            if not req_out_connection:
                ass_index = np.random.randint(associative_start_index, output_start_index)
                edges.append((input_index, ass_index))


        # --- Generate edges for ASSOCIATIVE neurons
        for ass_index in range(associative_start_index, output_start_index):
            req_out_connection = False

            # Enumerate over associative (for recurrent connections) and output neurons
            for connection_index in range(associative_start_index, self.total_neurons):

                # Create a connection with probability
                if np.random.random() < self.connect_percentage:
                    edges.append((ass_index, connection_index))

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
                        edges.append((ass_index, connection_index))
                        break

            # If no 'in' connection, randomly chose one input or associative neuron and create a connection
            if ass_index not in in_connected_neurons:

                # Iterate until get not recurrent connection
                while True:
                    connection_index = np.random.randint(0, output_start_index)
                    if ass_index != connection_index:
                        edges.append((connection_index, ass_index))
                        break


        # --- Generate edges for OUTPUT neurons
        for output_index in range(output_start_index, self.total_neurons):

            if output_index not in in_connected_neurons:
                connection_index = np.random.randint(associative_start_index, output_start_index)
                edges.append((connection_index, output_index))


        # Finally, load edges into the graph
        self.graph.add_edges_from(edges)
        nx.set_edge_attributes(self.graph, torch.tensor([0.0]), "weight")
        return

