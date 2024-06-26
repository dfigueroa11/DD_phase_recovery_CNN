import torch

import help_functions as hlp


class DD_system():
    '''
    Class implementing the system with Direct Detection
    '''

    def __init__(self, N_os, N_sim, constellation, diff_encoder, pulse_shape, ch_imp_resp, rx_filt):
        '''
        Arguments:
            N_os:           oversampling factor of the physical system (integer)
            N_sim:          oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)
            constellation:  constellation alphabet containing the possible symbols (1D tensor)
            diff_encoder:   instance of a Differential_encoder or None
            pulse_shape:    taps pulse shaping FIR filter (1D tensor with odd length)
            ch_imp_resp:   taps of the channel impulse response (1D tensor with length equal to pulse_shape)
            rx_filt:        taps of the receiver FIR filter (1D tensor with odd length)
        '''
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

    def simulate_transmission(self, batch_size, N_sym, SNR_dB):
        ''' Simulates the transmission of B batches and N symbols per batch with the given SNR 

        Arguments:
        batch_size:     number of batches to simulate (integer)
        N_sym:          number of symbols per batch (integer)
        SNR_dB:         SNR in dB used for the simulation (float)
        
        Returns:
        u:  the symbols before the differential encoding (Tensor of size (batch_size, 1, N_sym))
        x:  the symbols after the differential encoding (Tensor of size (batch_size, 1, N_sym)) if diff_encoder is None u = x  
        y:  the samples after the transmission (Tensor of size (batch_size, 1, N_sym*N_os))
        '''
        SNR_lin = 10**(SNR_dB/10)
        u = self.constellation[torch.randint(torch.numel(self.constellation),[batch_size, 1, N_sym])]
        if self.diff_encoder is not None:
            x = self.diff_encoder.encode(u)
        else:
            x = u
        x_up = torch.kron(x,torch.eye(self.N_sim)[-1])
        z = torch.square(torch.abs(hlp.convolve(x_up, self.tx_filt)))
        var_n = torch.tensor([self.N_sim/SNR_lin])
        y = z + torch.sqrt(var_n)*torch.randn_like(z)
        y = hlp.convolve(y, self.rx_filt)
        return u, x, y[:,:,self.d-1::self.d]
        
def set_up_DD_system(N_os, N_sim, **kwargs):
    '''Returns a DD_system with the given configuration for common constellations,
    pulse shapes, channel impulse response and receiver filter

    Arguments:
    N_os:   oversampling factor of the physical system (integer)
    N_sim:  oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)

    kwargs:
    mod_format:     constellation type (string: PAM, ASK, SQAM, QAM or DDQAM)
    M:              constellation order (integer) give together with mod_format
    constellation:  constellation to be used (1D tensor)
    alpha:          roll off factor of a raised cosine filter used as a pulse shape (float in [0,1])
    N_taps:         number of taps used for the pulse shape filter and channel impulse response filter (integer)
    pulse_shape:    particular pulse shape to be used (1D tensor)
    L_link:         length of the SMF in meters (float) use if the channel presents CD
    R_sym:          symbol rate in Hz (float) use if the channel presents CD
    beta2:          beta2 parameter of the SMF in s^2/m (float)
    ch_imp_resp:    particular chanel impulse response to be used (1D tensor)
    diff_encoder:   differential encoder to be used (Differential_encoder instance)
    rx_filt:        particular receiver filter to be used (1D tensor), 
                    if not specified uses a ideal LP filter simulate sampler BW, and if N_sim=N_os is a delta function
    '''
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

    
