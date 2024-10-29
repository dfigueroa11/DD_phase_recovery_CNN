import numpy as np
import torch.nn as nn

def calc_multi_layer_CNN_complexity(conv_layers: nn.ModuleList, sig_len: int=2**20):
    ''' Calculates the complexity of a CNN measured in number of multiplications per output per output channel

    Arguments:
    conv_layers:    list of convolutional layers applied in the given order in the CNN
    sig_len:        length of the input signal, default 2^20

    Returns:
    num_mult_per_output_per_output_channel
    '''
    assert all([isinstance(module, nn.Conv1d) for module in conv_layers]), "all layers must be instance of nn.Conv1d"
    num_m = 0
    cl: nn.Conv1d
    for cl in conv_layers:
        sig_len = (sig_len + 2*cl.padding[0] - cl.dilation[0]*(cl.kernel_size[0]-1) - 1)//cl.stride[0] + 1
        num_m += cl.out_channels * (cl.in_channels/cl.groups) * cl.kernel_size[0] * sig_len
    return np.ceil(num_m/(sig_len*cl.out_channels))