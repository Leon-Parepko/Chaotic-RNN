# Chaotic RNN

The main idea of this project is to obtain basic Recurrent Neural Network (RNN) architecture to initialize as a chaotic directed graph (with some rules) via probabilistic Monte Carlo simulation and create a corresponding experimental BP-like training algorithm. The key goal here is to state the hypothesis that these kind of reduced/chaotic architectures are trainable at all.

### Architecture:
Considering networks of such type consist of three main layers:
 - Input layer – ‘n’ neurons where ‘n’ is the number of input features. Input neurons could not have recurrent connections, but could be connected at minimum to a single next-layered neuron or at maximum fully connected to each next-layered neuron.
 - Chaotic recurrent layer - ‘m’ hidden neurons structured into a chaotic graph (or might be represented as classical fully connected RNN (or Hopfield) layer but with reduced forward and recurrent connections). The minimum required connections for this neuron are at least one input connection (from the input neuron or other hidden neuron except itself) and one output (output neuron or another hidden neuron except itself). At maximum, it seems to be Hopfield network-styled architecture but with added recurrent to itself connection.
 - Output layer - ‘k’ output neurons at a minimum should have one input (form hidden neuron) connection. At maximum - fully connected.

### Biological plausibility:
Here the input–hidden–output neural chain represents an afferent – interneuron - eferente chain. Even though one occurs in the peripheral nervous system (PNS), my approach considers the same idea for the central nervous system (CNS). So interneuron in this scheme would be replaced with a very complex cognitive structure (chaotic recurrent layer). Also one could note that McCulloch-Pitts neurons are not so Biologically plausible and not suitable for native recurrent processing. So for this purpose, it is needed to make my architecture scalable for replacing the neurons with the spiking ones (LIF or Izhikevich neurons).

### Purpose:
This experimental architecture with some spiking modifications would be used in the research of construction algorithms (self-constructed neural networks). So it is the basis for my further research.
