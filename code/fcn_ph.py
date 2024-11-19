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

    def __init__(self, y_len: int, a_len: int, fcn_out_len: int, hidden_layers_len: list[int], activ_func, activ_func_last_layer=None):
        super().__init__()
        self.lin_layers = nn.ModuleList()
        layer_in_lens = [y_len+a_len] + hidden_layers_len[:]
        layer_out_lens = hidden_layers_len[:] + [fcn_out_len]
        for in_len, out_len in zip(layer_in_lens, layer_out_lens):
            self.lin_layers.append(nn.Linear(in_len, out_len))
        self.activ_func = activ_func
        self.activ_func_last_layer = activ_func_last_layer
        self.complexity = calc_multi_layer_FCN_complexity(self.lin_layers)

    def forward(self, y: torch.Tensor, a: torch.Tensor):
        out = torch.cat((y,a), dim=-1)
        for lin_layer in self.lin_layers[:-1]:
            out = self.activ_func(lin_layer(out))
        out = self.lin_layers[-1](out)
        if self.activ_func_last_layer is not None:
            return self.activ_func_last_layer(out)
        return out

fcn_out_2_u_hat_funcs = {TRAIN_MSE: dconv_tools.MSE_FCN_out_2_complex,
                         TRAIN_MSE_PHASE_FIX: dconv_tools.MSE_FCN_out_2_complex,
                         TRAIN_CE: dconv_tools.CE_FCN_out_2_complex}