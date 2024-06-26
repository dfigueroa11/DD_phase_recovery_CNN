import torch
import torch.nn as nn

class CNN_equalizer(nn.Module):
    '''
    Class implementing the CNN equalizer
    '''
    
    def __init__(self, num_ch, ker_lens, strides, activ_func):
        '''
        Arguments:
        num_ch:     list with the number of input channels of each layer and number of output channel of the last layer 
                    so CNN channels behave as: num_ch[0] -> conv1d -> num_ch[1] -> ... -> conv1d -> num_ch[-1] (list of length L-1)
        ker_lens:   list with the length of the kernel of each layer (list of length L, each number should be odd)
        strides:    list with the stride of each layer (list of length L)
        activ_func: activation function applied after each layer, except the last one
        '''
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for ch_in, ch_out, ker_len, stride in zip(num_ch[:-1], num_ch[1:], ker_lens, strides):
            self.conv_layers.append(nn.Conv1d(ch_in, ch_out, ker_len, stride, (ker_len-1)//2))
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
    