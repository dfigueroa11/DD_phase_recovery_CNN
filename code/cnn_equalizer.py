import torch
import torch.nn as nn

class CNN_equalizer(nn.Module):
    
    def __init__(self, num_ch, ker_lens, strides, paddings, activ_func):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for ch_in, ch_out, ker_len, stride ,padding in zip(num_ch[:-1], num_ch[1:], ker_lens, strides, paddings):
            self.conv_layers.append(nn.Conv1d(ch_in, ch_out, ker_len, stride, padding))
        self.activ_func = activ_func

    def forward(self, y):
        out = y
        for conv_layer in self.conv_layers[:-1]:
            out = self.activ_func(conv_layer(out))
        out = self.conv_layers[-1](out)
        return out
    