import torch
import torch.nn as nn
import data_conversion_tools as dconv_tools

from complexity_tools import calc_multi_layer_FCN_complexity

TRAIN_MSE = 0
TRAIN_MSE_PHASE_FIX = 1
TRAIN_CE = 2
TRAIN_TYPES = {TRAIN_MSE: "TRAIN_MSE",
               TRAIN_MSE_PHASE_FIX: "TRAIN_MSE_PHASE_FIX",
               TRAIN_CE: "TRAIN_CE"}

class FCN_ph(nn.Module):

    def __init__(self, y_len: int, a_len: int, sym_out: int, outs_per_symbol: int, hidden_layers_len: list[int], layer_is_bilin: list[bool], activ_func, activ_func_last_layer=None):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_in_lens = [y_len+a_len] + hidden_layers_len[:]
        layer_out_lens = hidden_layers_len[:] + [sym_out*outs_per_symbol]
        for in_len, out_len, l_is_bilin in zip(layer_in_lens, layer_out_lens, layer_is_bilin):
            if l_is_bilin:
                self.layers.append(nn.Bilinear(in_len, in_len, out_len))
            else:
                self.layers.append(nn.Linear(in_len, out_len))
        self.activ_func = activ_func
        self.activ_func_last_layer = activ_func_last_layer
        self.complexity = calc_multi_layer_FCN_complexity(self.layers, sym_out)

    def forward(self, y: torch.Tensor, a: torch.Tensor):
        out = torch.cat((y,a), dim=-1)
        for layer in self.layers[:-1]:
            out = self.activ_func(self.apply_layer(out, layer))
        out = self.apply_layer(out, self.layers[-1])
        if self.activ_func_last_layer is not None:
            return self.activ_func_last_layer(out)
        return out

    def apply_layer(self, out, layer):
        if isinstance(layer, nn.Bilinear):
            return layer(out, out)
        return layer(out)

fcn_out_2_u_hat_funcs = {TRAIN_MSE: dconv_tools.MSE_FCN_out_2_complex,
                         TRAIN_MSE_PHASE_FIX: dconv_tools.MSE_FCN_out_2_complex,
                         TRAIN_CE: dconv_tools.CE_FCN_out_2_complex}