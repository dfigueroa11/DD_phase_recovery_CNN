import numpy as np
import torch.nn as nn

def calc_multi_layer_CNN_complexity(conv_layers: nn.ModuleList, sig_len: int=2**20):
    ''' Calculates the complexity of a CNN measured in number of multiplications per output symbol

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
    return np.ceil(num_m/(sig_len))

def design_CNN_structures_fix_comp(complexity: int, complexity_profile: np.ndarray, CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray, groups: np.ndarray,
                          n_str_layer: int=4):
    ''' Design many structures of CNN with a given complexity

    Arguments:
    complexity:             number of multiplications per CNN output
    complexity_profile:     fraction of the complexity spent in each layer (must add up to 1)
    CNN_ch_in:              number of input channels of the CNN
    CNN_ch_out:             number of output channels of the CNN
    strides:                stride for each layer
    groups:                 groups for each layer
    n_str_layer:            number of new structures designed per layer.
                            At the end there will be n_str_layer^(num layers -1) different structures since the last layer is determined by CNN_ch_out

    Returns:
    list containing all the designed structures
    '''
    assert np.isclose(sum(complexity_profile),1), "complexity_profile must add up to 1"
    assert complexity_profile.size == strides.size and strides.size == groups.size, "complexity_profile, strides and groups must have the same size"
    layers_complexity_budget = complexity*complexity_profile
    first_structure = -np.ones((5,strides.size))
    first_structure[0,0] = CNN_ch_in
    first_structure[3,:] = strides
    first_structure[4,:] = groups
    structures = [first_structure,]
    for layer_idx, l_comp in enumerate(layers_complexity_budget):
        new_structures = []
        for structure in structures:
            new_structures += design_conv_layer_fix_comp(l_comp, structure, layer_idx, n_str_layer, CNN_ch_out)
        structures = new_structures
    return structures

def design_conv_layer_fix_comp(complexity: float, structure: np.ndarray, layer_idx: int, num_new_structures: int, CNN_ch_out: int):
    '''Starting from a given structure of the first (i-1) conv layers, design 'num' new structures for the i-th conv layer,
    while having approximately the given complexity for the designed layer, and meeting the requirements of stride and groups.

    Arguments:
    complexity:     complexity for the current layer, measured in number of multiplications per CNN output
    structure:      structure of the CNN from the first layer up to the (i-1)-th layer (shape (5,CNN_N_layers),
                    1st row: in_channels, 2nd row: out_channels, 3rd row: kernel_size, 4th row: stride, 5th row: groups)
    layer_idx:      index of the layer to design (0 -> first layer, 1 -> second, ...)
    num_new_structures:     number of new structures to create
    CNN_ch_out:     number of output channels at the end of the CNN

    Returns:
    structures:     list of structures designed
    '''
    layer_ch_in = structure[0,layer_idx]
    strides = structure[3,:]
    groups_current = structure[4, layer_idx]
    # if we are in the last layer:
    if layer_idx+1 == strides.size:
        structure[1,layer_idx] = round_ch_out(CNN_ch_out, int(groups_current))
        structure[2,layer_idx] = round_kernel_size(complexity*groups_current/(CNN_ch_out*layer_ch_in))
        return [structure.astype(int),]
    groups_next = structure[4, layer_idx+1]
    prod_layer_ch_out_ker_sz = complexity*groups_current/(layer_ch_in*np.prod(strides[layer_idx+1:]))
    layer_ch_out_options = np.logspace((1/(num_new_structures+1)), np.log10(prod_layer_ch_out_ker_sz), num_new_structures, endpoint=False)
    structures = []
    for layer_ch_out in layer_ch_out_options:
        new_structure = structure.copy()
        new_structure[1,layer_idx] = round_ch_out(layer_ch_out, int(groups_current), int(groups_next))
        new_structure[2,layer_idx] = round_kernel_size(prod_layer_ch_out_ker_sz/round_ch_out(layer_ch_out, int(groups_current), int(groups_next)))
        # input of the next layer is output of the current one
        new_structure[0,layer_idx+1] = round_ch_out(layer_ch_out, int(groups_current), int(groups_next))
        structures.append(new_structure.astype(int))
    return structures

def design_CNN_structures_fix_geom(complexities: np.ndarray, complexity_profile: np.ndarray, ch_out_ker_sz_ratios: np.ndarray,
                                   CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray, groups: np.ndarray):
    assert np.isclose(sum(complexity_profile),1), "complexity_profile must add up to 1"
    assert complexity_profile.size == strides.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert strides.size == groups.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert complexity_profile.size == ch_out_ker_sz_ratios.size+1, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    structures = []
    for complexity in complexities:
        layers_complexity = complexity*complexity_profile
        new_structure = -np.ones((5,strides.size))
        new_structure[0,0] = CNN_ch_in
        new_structure[3,:] = strides
        new_structure[4,:] = groups
        for i, ch_out_ker_sz_ratio in enumerate(ch_out_ker_sz_ratios):
            prod_layer_ch_out_ker_sz = layers_complexity[i]*groups[i]/(new_structure[0,i]*np.prod(strides[i+1:]))
            new_structure[1,i] = round_ch_out(np.sqrt(prod_layer_ch_out_ker_sz*ch_out_ker_sz_ratio), int(groups[i]), int(groups[i+1]))
            new_structure[2,i] = round_kernel_size(new_structure[1,i]/ch_out_ker_sz_ratio)
            new_structure[0,i+1] = new_structure[1,i]
        new_structure[1,-1] = round_ch_out(CNN_ch_out, int(groups[-1]))
        new_structure[2,-1] = round_kernel_size(layers_complexity[-1]*groups[-1]/(CNN_ch_out*new_structure[0,-1]))
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
