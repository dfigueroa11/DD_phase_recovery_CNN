import torch
import torch.nn as nn
import data_conversion_tools as dconv_tools

TRAIN_MSE_U_SYMBOLS = 0
TRAIN_MSE_U_MAG_PHASE = 1
TRAIN_MSE_U_SLDMAG_PHASE = 2
TRAIN_CE_U_SYMBOLS = 3
TRAIN_TYPES = [TRAIN_MSE_U_SYMBOLS,
               TRAIN_MSE_U_MAG_PHASE,
               TRAIN_MSE_U_SLDMAG_PHASE,
               TRAIN_CE_U_SYMBOLS]

class CNN_equalizer(nn.Module):
    '''
    Class implementing the CNN equalizer
    '''
    
    def __init__(self, num_ch, ker_lens, strides, activ_func, groups_list=None):
        '''
        Arguments:
        num_ch:         list with the number of input channels of each layer and number of output channel of the last layer 
                        so CNN channels behave as: num_ch[0] -> conv1d -> num_ch[1] -> ... -> conv1d -> num_ch[-1] (list of length L-1)
        ker_lens:       list with the length of the kernel of each layer (list of length L, each number should be odd)
        strides:        list with the stride of each layer (list of length L)
        activ_func:     activation function applied after each layer, except the last one
        groups_list:    groups applied to each layer of the CNN (list of length L, default: list of ones)
        '''
        super().__init__()
        if groups_list is None:
            groups_list = [1]*len(ker_lens)

        self.conv_layers = nn.ModuleList()
        for ch_in, ch_out, ker_len, stride, groups in zip(num_ch[:-1], num_ch[1:], ker_lens, strides, groups_list):
            self.conv_layers.append(nn.Conv1d(ch_in, ch_out, ker_len, stride, (ker_len-1)//2, groups=groups))
        self.activ_func = activ_func

    def forward(self, y):
        '''
        Arguments:
        y:      input signal (Tensor of size (batch_size, in_channels, signal_in_length), in_channels as defined when creating the CNN)
        
        Returns:
        out:    output signal (Tensor of size (batch_size, out_channels, signal_out_length),
                out_channels as defined when creating the CNN, and signal_out_length depends on the stride of each layer)
        '''
        out = y
        for conv_layer in self.conv_layers[:-1]:
            out = self.activ_func(conv_layer(out))
        out = self.conv_layers[-1](out)
        return out
    
cnn_out_2_u_hat_funcs = {TRAIN_MSE_U_SYMBOLS: dconv_tools.mag_phase_2_complex,
                         TRAIN_MSE_U_MAG_PHASE: dconv_tools.mag_phase_2_complex,
                         TRAIN_MSE_U_SLDMAG_PHASE: dconv_tools.SLDmag_phase_2_complex,
                         TRAIN_CE_U_SYMBOLS: dconv_tools.APPs_2_u}