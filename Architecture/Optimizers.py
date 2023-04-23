import torch
from torch.optim import Optimizer


class STDP(Optimizer):

    def __init__(self, params, spike_history, lr=0.01, device=None, dtype=None, tau=1):
        super(STDP, self).__init__(params, {})
        self.lr = lr
        self.tau = tau
        self.spikes = spike_history

    def step(self, closure=None):
        for group in self.param_groups:
            for i in range(len(group['params'])):
                layer_w = group['params'][i].data

                # Initialize previous layer weights
                prev_layer_w = None
                if i > 0:
                    prev_layer_w = group['params'][i-1].data

                layer_spike_history = self.spikes[i]()

                for j, neuron_spike in enumerate(layer_spike_history['cur']):
                    if neuron_spike > 0:

                        # LTP for all pre spikes activity (update previous layer weights)
                        if layer_spike_history['pre'] is not None:
                            delta_w = self.lr * torch.exp(-layer_spike_history['pre'] / self.tau)
                            prev_layer_w[j] += delta_w

                        # LTD for all pre spikes activity (update current layer weights)
                        if layer_spike_history['post'] is not None:
                            delta_w = self.lr * torch.exp(-layer_spike_history['post'] / self.tau)
                            layer_w.T[j] -= delta_w

                            # Replace negative weights with 0.001
                            # TODO: check if correct
                            layer_w.T[j][layer_w.T[j] < 0] = .001

                # Normalize all weights
                for j, row in enumerate(layer_w):
                    layer_w[j] = row / torch.sum(row)




        #         print(f"Layer {i} weights: {layer_w}\n")
        # print("=============================================")



        #         splitter = "------------------------"
        #         print(f"PRE: {layer_spike_history['pre']}\n"
        #               f"CURR: {layer_spike_history['cur']}\n"
        #               f"POST: {layer_spike_history['post']}\n"
        #               f"{splitter}")
        # print("=====================================")