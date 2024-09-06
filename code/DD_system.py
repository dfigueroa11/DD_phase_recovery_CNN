import torch

import help_functions as hlp


class DD_system():
    '''
    Class implementing the system with Direct Detection

    Methods:

    simulate_transmission(batch_size, N_sym, SNR_dB)
    '''

    def __init__(self, N_os, N_sim, constellation, diff_encoder, pulse_shape, ch_imp_resp, rx_filt, device):
        '''
        Arguments:
            N_os:           oversampling factor of the physical system (integer)
            N_sim:          oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)
            constellation:  constellation alphabet containing the possible symbols (1D tensor)
            diff_encoder:   instance of a Differential_encoder or None
            pulse_shape:    taps pulse shaping FIR filter (1D tensor with odd length)
            ch_imp_resp:   taps of the channel impulse response (1D tensor with length equal to pulse_shape)
            rx_filt:        taps of the receiver FIR filter (1D tensor with odd length)
            device:         the device to use (cpu or cuda)
        '''
        self.device = device
        self.N_os = N_os
        self.N_sim = N_sim
        self.d = N_sim//N_os
        self.constellation = constellation.to(device)
        self.multi_mag_const = torch.unique(torch.round(torch.abs(self.constellation), decimals=5)).numel() > 1
        self.multi_phase_const = torch.unique(torch.round(torch.angle(self.constellation), decimals=5)).numel() > 1
        self.diff_encoder = diff_encoder
        if pulse_shape is not None and ch_imp_resp is not None:
            self.tx_filt = hlp.norm_filt(N_sim, hlp.cascade_filters(pulse_shape, ch_imp_resp)).view(1,1,-1).to(device)
        else:
            self.tx_filt = None
        self.rx_filt = hlp.norm_filt(N_sim, rx_filt).view(1,1,-1).to(device)

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
        u = self.constellation[torch.randint(torch.numel(self.constellation),[batch_size, 1, N_sym])]
        if self.diff_encoder is not None:
            x = self.diff_encoder.encode(u)
        else:
            x = u
        x_up = torch.kron(x,torch.eye(self.N_sim, device=self.device)[-1])
        z = torch.square(torch.abs(hlp.convolve(x_up, self.tx_filt)))
        SNR_lin = torch.tensor([10**(SNR_dB/10)], device=self.device, dtype=torch.float32)
        var_n = torch.tensor([self.N_sim/(2*SNR_lin)], dtype=torch.float32, device=self.device)
        y = z + torch.sqrt(var_n)*torch.randn_like(z)
        y = hlp.convolve(y, self.rx_filt)
        return u, x, y[:,:,self.d-1::self.d]
    