import torch

import help_functions as hlp


class DD_system():

    def __init__(self, N_os, N_sim, constellation, diff_encoder, pulse_shape, ch_imp_resp, rx_filt):
        self.N_os = N_os
        self.N_sim = N_sim  
        self.d = N_sim//N_os 
        self.constellation = constellation
        self.diff_encoder = diff_encoder
        if pulse_shape is not None and ch_imp_resp is not None:
            self.tx_filt = hlp.cascade_filters(pulse_shape, ch_imp_resp).view(1,1,-1)
        else:
            self.tx_filt = None
        self.rx_filt = rx_filt.view(1,1,-1)

    def simulate_transmission(self, batch_size, N_sym, Ptx_dB):
        Ptx_lin = 10**(Ptx_dB/10)
        u = torch.sqrt(Ptx_lin)*self.constellation[torch.randint(torch.numel(self.constellation),[batch_size, 1, N_sym])]
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
        
def set_up_DD_system(N_os, N_sim, **kwargs):
    if {"mod_format", "M"} <= kwargs.keys():
        constellation = hlp.common_constellation(kwargs["mod_format"], kwargs["M"])
    elif "constellation" in kwargs.keys():
        constellation = kwargs["constellation"]
    else:
        constellation = None
    if {"alpha", "N_taps"} <= kwargs.keys():
        pulse_shape = hlp.rcos_filt(kwargs["alpha"], kwargs["N_taps"], N_sim, 1)
    elif "pulse_shape" in kwargs.keys():
        pulse_shape = kwargs["pulse_shape"]
    else:
        pulse_shape = None
    if {"L_link", "R_sym", "beta2", "N_taps"} <= kwargs.keys():
        ch_imp_resp = hlp.chrom_disp_filt(kwargs["L_link"], kwargs["R_sym"], kwargs["beta2"], kwargs["N_taps"], N_sim)
    elif "ch_imp_resp" in kwargs.keys():
        ch_imp_resp = kwargs["ch_imp_resp"]
    else:
        ch_imp_resp = None
    if "diff_encoder" in kwargs.keys():
        diff_encoder = kwargs["diff_encoder"]
    else:
        diff_encoder = None
    if "rx_filt" in kwargs.keys():
        rx_filt = kwargs["rx_filt"]
    elif N_sim > N_os:
        rx_filt = hlp.rcos_filt(0, len(pulse_shape), N_os, 1, dtype=torch.float32)
    else:
        rx_filt = torch.tensor([1.])
    return DD_system(N_os, N_sim, constellation , diff_encoder, pulse_shape, ch_imp_resp, rx_filt)





if __name__ == "__main__":

    system = set_up_DD_system(N_os= 2, N_sim=2,
                              mod_format="PAM", M=4,
                              N_taps=7,     # for all
                              alpha=0.2, 
                              L_link=10e3, R_sym=20e9, beta2=-2e26)
    u, x, y = system.simulate_transmission(3,20,torch.tensor([10000]))

    
