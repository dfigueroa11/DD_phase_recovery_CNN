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

def design_CNN_structures(complexity: int, complexity_profile: np.ndarray, CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray, groups: np.ndarray):
    structures = [1]
    # assert np.isclose(sum(complexity_profile),1), "complexity_profile must add up to 1"
    assert complexity_profile.size == strides.size and strides.size == groups.size, "complexity_profile, strides and groups must have the same size"
    layers_complexity_budget = complexity*complexity_profile
    for layer_idx, l_comp in enumerate(layers_complexity_budget):
        new_structures = []
        for structure in structures:
            new_structures += design_conv_layer(l_comp,structure,layer_idx)
        structures = new_structures
    return structures

def design_conv_layer(complexity: float, structure: np.ndarray, layer_idx: int, num_new_structures: int, CNN_ch_out: int):
    layer_ch_in = structure[0,layer_idx]
    strides = structure[3,:]
    groups_current = structure[4, layer_idx]
    # if we are in the last layer:
    if layer_idx+1 == strides.size:
        structure[1,layer_idx] = round_ch_out(CNN_ch_out, groups_current)
        structure[2,layer_idx] = complexity*groups_current/(layer_ch_in*np.prod(strides[layer_idx+1:]))
        return [structure,]
    groups_next = structure[4, layer_idx+1]
    prod_layer_ch_out_ker_sz = complexity*groups_current*CNN_ch_out/(layer_ch_in*np.prod(strides[layer_idx+1:]))
    layer_ch_out_options = np.logspace((1/(num_new_structures+1)), np.log10(prod_layer_ch_out_ker_sz), num_new_structures, endpoint=False)
    k_size_options = prod_layer_ch_out_ker_sz/layer_ch_out_options
    structures = []
    for layer_ch_out, k_size in zip(layer_ch_out_options, k_size_options):
        new_structure = structure.copy()
        new_structure[1,layer_idx] = round_ch_out(layer_ch_out, groups_current, groups_next)
        new_structure[2,layer_idx] = round_kernel_size(k_size)
        new_structure[0,layer_idx+1] = round_ch_out(layer_ch_out, groups_current, groups_next)       # input of the next layer is output of the current one
        structures.append(new_structure)
    return structures

def round_ch_out(ch_out: float, groups_pre: int=1, groups_post: int=1):
    ''' Returns the smallest integer divisible by groups_current_pre and groups_current_post bigger than ch_out
    '''
    divisible_by = np.lcm(groups_pre, groups_post)
    remainder = np.mod(np.ceil(ch_out), divisible_by)
    if remainder == 0:
        return int(np.ceil(ch_out))
    return int(np.ceil(ch_out)+divisible_by-remainder)

def round_kernel_size(kernel_size: float):
    ''' Returns the smallest odd number bigger than kernel_size
    '''
    return int(np.ceil(kernel_size) + np.mod(np.ceil(kernel_size)+1,2))

if __name__=="__main__":
    xd = design_conv_layer(500, np.array([[1,-1,-1,-1],[-1,-1,-1,-1],[-1,-1,-1,-1],[1,1,1,1],[1,3,5,1]]),0,5,1)
    xd = design_conv_layer(500,xd[1],1,5,1)
    xd = design_conv_layer(500,xd[1],2,5,1)
    # xd = design_conv_layer(500,xd[1],3,5,1)
    
    for xxd in xd:
        print(xxd)
