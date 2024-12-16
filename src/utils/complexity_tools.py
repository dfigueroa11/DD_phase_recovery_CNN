import numpy as np
import torch.nn as nn

################ CNN ####################

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

def design_CNN_structures_fix_geom(complexities: np.ndarray, complexity_profile: np.ndarray, exp_chs: np.ndarray,
                                   CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray, groups: np.ndarray):
    assert np.isclose(sum(complexity_profile),1), "complexity_profile must add up to 1"
    assert complexity_profile.size == strides.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert strides.size == groups.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert complexity_profile.size == exp_chs.size+1, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    structures = []
    for complexity in complexities:
        layers_complexity = complexity*complexity_profile
        new_structure = -np.ones((5,strides.size), dtype=int)
        new_structure[0,0] = CNN_ch_in
        new_structure[3,:] = strides
        new_structure[4,:] = groups
        for i, exp_ch in enumerate(exp_chs):
            prod_layer_ch_out_ker_sz = layers_complexity[i]*groups[i]/(new_structure[0,i]*np.prod(strides[i+1:]))
            new_structure[1,i] = round_ch_out(np.power(prod_layer_ch_out_ker_sz,exp_ch), int(groups[i]), int(groups[i+1]))
            new_structure[2,i] = round_kernel_size(prod_layer_ch_out_ker_sz/new_structure[1,i])
            new_structure[0,i+1] = new_structure[1,i]
        new_structure[1,-1] = round_ch_out(CNN_ch_out, int(groups[-1]))
        new_structure[2,-1] = round_kernel_size(layers_complexity[-1]*groups[-1]/(CNN_ch_out*new_structure[0,-1]))
        structures.append(new_structure)
    return structures

def design_CNN_structures_fix_comp2(complexity: int, complexity_profile: np.ndarray, CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray,
                                    groups: np.ndarray, ker_sz_lims: np.ndarray, n_str_layer: int=4):
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
            new_structures += design_conv_layer_fix_comp2(l_comp, structure, layer_idx, n_str_layer, CNN_ch_out, ker_sz_lims)
        structures = new_structures
    return structures

def design_conv_layer_fix_comp2(complexity: float, structure: np.ndarray, layer_idx: int, num_new_structures: int, CNN_ch_out: int, ker_sz_lims: np.ndarray):
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
    ker_sz_options = np.linspace(ker_sz_lims[0], ker_sz_lims[1], num_new_structures, endpoint=True)
    structures = []
    for ker_sz in ker_sz_options:
        new_structure = structure.copy()
        new_structure[2,layer_idx] = round_kernel_size(ker_sz)
        new_structure[1,layer_idx] = round_ch_out(prod_layer_ch_out_ker_sz/new_structure[2,layer_idx], int(groups_current), int(groups_next))
        # input of the next layer is output of the current one
        new_structure[0,layer_idx+1] = new_structure[1,layer_idx]
        structures.append(new_structure.astype(int))
    return structures

def design_CNN_structures_fix_geom2(complexities: np.ndarray, complexity_profile: np.ndarray, ker_sz_s: np.ndarray,
                                   CNN_ch_in: int, CNN_ch_out: int, strides: np.ndarray, groups: np.ndarray):
    assert np.isclose(sum(complexity_profile),1), "complexity_profile must add up to 1"
    assert complexity_profile.size == strides.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert strides.size == groups.size, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    assert complexity_profile.size == ker_sz_s.size+1, "complexity_profile, strides and groups must have the same size, and ch_out_ker_sz_ratios must have one element less"
    structures = []
    for complexity in complexities:
        layers_complexity = complexity*complexity_profile
        new_structure = -np.ones((5,strides.size), dtype=int)
        new_structure[0,0] = CNN_ch_in
        new_structure[3,:] = strides
        new_structure[4,:] = groups
        for i, ker_sz in enumerate(ker_sz_s):
            prod_layer_ch_out_ker_sz = layers_complexity[i]*groups[i]/(new_structure[0,i]*np.prod(strides[i+1:]))
            new_structure[2,i] = round_kernel_size(ker_sz)
            new_structure[1,i] = round_ch_out(prod_layer_ch_out_ker_sz/new_structure[2,i], int(groups[i]), int(groups[i+1]))
            new_structure[0,i+1] = new_structure[1,i]
        new_structure[1,-1] = round_ch_out(CNN_ch_out, int(groups[-1]))
        new_structure[2,-1] = round_kernel_size(layers_complexity[-1]*groups[-1]/(CNN_ch_out*new_structure[0,-1]))
        structures.append(new_structure)
    return structures

def round_ch_out(ch_out: float, groups_pre: int=1, groups_post: int=1):
    ''' Returns the nearest integer divisible by groups_current_pre and groups_current_post to ch_out
    '''
    ch_out = np.round(ch_out)
    if ch_out == 0:
        ch_out = 1
    divisible_by = np.lcm(groups_pre, groups_post)
    remainder = np.mod(ch_out, divisible_by)
    if remainder == 0:
        return int(ch_out)
    if remainder < divisible_by/2:
        return int(ch_out+divisible_by-remainder)
    if int(ch_out-remainder) < 1:
        return int(ch_out+divisible_by-remainder)
    return int(ch_out-remainder)

def round_kernel_size(kernel_size: float):
    ''' Returns the nearest odd number to the kernel_size
    '''
    if np.mod(np.round(kernel_size),2) == 1:
        return int(np.round(kernel_size))
    if np.round(kernel_size) < kernel_size:
        return int(np.round(kernel_size) + 1)
    return int(np.round(kernel_size) - 1)

################ FCN ####################
def calc_multi_layer_FCN_complexity(layers: nn.ModuleList, sym_out):
    assert all([isinstance(module, nn.Linear) or isinstance(module, nn.Bilinear) for module in layers]), "all layers must be instance of nn.Linear or  nn.Bilinear"
    return np.ceil(sum([l.weight.data.numel() for l in layers])/sym_out)

################ RNN ####################
def calc_RNN_complexity(TVRNN_layers, Lin_layer):
    num_m = 0
    for TVRNN_layer in TVRNN_layers:
        num_m += calc_TVRNN_layer_complexity(TVRNN_layer)
    num_m += Lin_layer.weight.data.numel()
    return num_m

def calc_TVRNN_layer_complexity(TVRNN_layer):
    num_m = 0
    for cell_fw, cell_bw in zip(TVRNN_layer.cells_fw, TVRNN_layer.cells_bw):
        num_m += calc_RNN_Cell_complexity(cell_fw)
        num_m += calc_RNN_Cell_complexity(cell_bw)
    return num_m

def calc_RNN_Cell_complexity(RNN_cell):
    return RNN_cell.Lin_layer_input_to_hidden.weight.data.numel() + RNN_cell.Lin_layer_hidden_to_hidden.weight.data.numel()