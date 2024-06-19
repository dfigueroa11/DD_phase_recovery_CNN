import torch
from torch.nn.functional import conv1d

import help_functions as hlp


class DD_system():

    def __init__(self, N_os, N_sim, constellation, diff_encoder, pulse_shape, ch_imp_resp, rx_filt, ):
        self.N_os = N_os
        self.N_sim = N_sim  
        self.d = N_sim//N_os 
        self.constellation = constellation
        self.diff_encoder = diff_encoder
        self.tx_filt = hlp.cascade_filters(pulse_shape, ch_imp_resp).view(1,1,-1)
        self.rx_filt = rx_filt.view(1,1,-1)

    def simulate_transmission(self, batch_size, N_sym, Ptx):
        u = torch.sqrt(Ptx)*self.constellation[torch.randint(torch.numel(self.constellation),[batch_size, 1, N_sym])]
        if self.diff_encoder is not None:
            x = self.diff_encoder.encode(u)
        else:
            x = u
        x_up = torch.kron(x,torch.eye(self.N_sim)[-1])
        z = torch.square(torch.abs(hlp.convolve(x_up, self.tx_filt)))
        var_n = torch.tensor([self.N_sim/2])
        y = z + torch.sqrt(var_n)*torch.randn_like(z)
        y = hlp.convolve(y, self.rx_filt)
        return u, x, y[:,:,self.d-1::self.d]
        




if __name__ == "__main__":

    constellation = torch.tensor([1 + 1j, 1 - 1j, -1 + 1j, -1 -1j])
    pulse_shape = torch.tensor([0, 0, 0, 1, 0, 0, 0])
    ch_imp_resp = torch.tensor([0, 0, 0, 1, 0, 0, 0])
    rx_filt = torch.tensor([1.])
    N_os = 5
    N_sim = 10

    system = DD_system(N_os, N_sim, constellation , None, pulse_shape, ch_imp_resp, rx_filt)
    u, x = system.simulate_transmission(3,20,torch.tensor([10000]))
    
