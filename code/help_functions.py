import numpy as np
import torch
from torch.nn.functional import conv1d
import DD_system
import Differential_encoder

def cascade_filters(filt_1, filt_2):
    '''combine 2 filters by multiplication in frequency domain

    Arguments: 
    filt_1, filt_2:     impulse response of the filters to combine, they must be of the same length (1D tensor)

    Return:
    filt_out:           combination of the filters (1D tensor of length equal to the input filters)
    '''
    FILT_1 =  torch.fft.fft(torch.fft.ifftshift(filt_1))
    FILT_2 =  torch.fft.fft(torch.fft.ifftshift(filt_2))
    return  torch.fft.fftshift(torch.fft.ifft(FILT_1*FILT_2))

def convolve(signal, filter):
    '''Apply the convolution to the signals using the conv1d torch function
    
    Arguments: 
    signal:     tensor of size (batch_size, in_channels, signal_length)
    filter:     tensor of size (out_channels, in_channels, filter_length)

    Return:
    out:        tensor of size (batch_size, out_channels, signal_length)   
    '''
    filter = torch.resolve_conj(torch.flip(filter, [-1]))
    return conv1d(signal, filter, padding='same')

def common_constellation(mod, M, dtype=torch.cfloat):
    '''Returns the constellation specified (1D tensor of size M)

    Arguments:
    mod:    String with the modulation format, valid options are 'PAM', 'ASK', 'SQAM', 'QAM' or 'DDQAM'
    M:      order of the modulation
    dtype:  data type (optional, default torch.cfloat)
    '''
    if mod == "PAM":
        constellation = np.linspace(0, 1, num=M, endpoint=True)
    elif mod == "ASK":
        constellation = np.linspace(-1, 1, num=M, endpoint=True)
    elif mod == "SQAM":
        X_base = np.array([1, 1j, -1, -1j])
        constellation = np.array([], dtype=np.complex64)
        for ii in range(M // len(X_base)):
            constellation = np.append(constellation, (ii + 1) * X_base)
    elif mod == "QAM":
        Mp = int(np.sqrt(M))
        constellation1D = np.linspace(-1, 1, num=Mp, endpoint=True)
        constellation = np.reshape(np.add.outer(constellation1D.T, 1j * constellation1D), -1)
    elif mod == "DDQAM":
        angle = np.arccos(1/3)
        constellation = np.kron(np.arange(1, M//4+1),np.exp(np.array([0,angle,np.pi,np.pi+angle])*1j))
    else:
        raise ValueError("mod should be PAM, ASK, SQAM, QAM or DDQAM")

    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    return torch.tensor(constellation, dtype=dtype)

def common_diff_encoder(mod, constellation):
    '''Returns the constellation specified (1D tensor of size M)

    Arguments:
    mod:            String with the modulation format, valid options are 'PAM', 'ASK', 'SQAM', 'QAM' or 'DDQAM'
    constellation:  constellation:  constellation to be used (1D tensor)
    '''
    if mod == "PAM":
        return None
    elif mod == "ASK":
        diff_mapping = torch.tensor([[1,0],[0,1]])
    elif mod == "SQAM":
        diff_mapping = torch.tensor([[3,0,1,2],[0,1,2,3],[1,2,3,0],[2,3,0,1]])
    elif mod == "QAM":
        return None
    elif mod == "DDQAM":
        diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])
    else:
        raise ValueError("mod should be PAM, ASK, SQAM, QAM or DDQAM")
    return Differential_encoder.Differential_encoder(constellation, diff_mapping)

def rcos_filt(alpha, N_taps, fs, sym_time, dtype=torch.cfloat):
    ''' Returns a raised cosine filter (1D tensor of length N_taps)

    Arguments:
    alpha:      roll off factor (float between 0 and 1)
    N_taps:     number of coefficients (integer must be odd)
    fs:         sampling frequency (float)
    sym_time:   symbol time (float)
    dtype:      data type (optional, default torch.cfloat)
    '''
    t_vec = (np.arange(N_taps)-(N_taps-1)/2)/fs
    if alpha == 0:
        return torch.tensor(np.sinc(t_vec/sym_time), dtype=dtype)
    rcos =  np.where(np.abs(t_vec) == sym_time/(2*alpha), np.pi/4*np.sinc(1/(2*alpha)), \
                     np.sinc(t_vec/sym_time)*(np.cos(np.pi*alpha*t_vec/sym_time))/(1-(2*alpha*t_vec/sym_time)**2))
    return torch.tensor(rcos, dtype=dtype)

def chrom_disp_filt(L_link, R_sym, beta2, N_taps, N_sim, dtype=torch.cfloat):
    ''' Returns the impulse response of a SMF with CD (1D tensor of length N_taps)
    
    Arguments:
    L_link:     length of the SMF in meters (float) use if the channel presents CD
    R_sym:      symbol rate in Hz (float) use if the channel presents CD
    beta2:      beta2 parameter of the SMF in s^2/m (float)
    N_taps:     number of coefficients (integer must be odd)
    N_sim:      oversampling factor used during the simulation
    dtype:      data type (optional, default torch.cfloat)
    '''
    delta_f = (N_sim*R_sym)/N_taps
    f = (np.arange(N_taps) - np.floor(N_taps/2))*delta_f
    H_cd = np.exp(1j*((2*np.pi*f)**2*beta2*L_link/2))
    h_cd = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(H_cd)))
    return torch.tensor(h_cd, dtype=dtype)

def set_up_DD_system(N_os, N_sim, **kwargs):
    '''Returns a DD_system with the given configuration for common constellations,
    pulse shapes, channel impulse response and receiver filter

    Arguments:
    N_os:   oversampling factor of the physical system (integer)
    N_sim:  oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)

    kwargs:
    mod_format:     constellation type (string: PAM, ASK, SQAM, QAM or DDQAM)
    M:              constellation order (integer) give together with mod_format
    diff_encoder:   boolean to use or no differential encoding give together with mod_format and M
    constellation:  constellation to be used (1D tensor)
    alpha:          roll off factor of a raised cosine filter used as a pulse shape (float in [0,1])
    N_taps:         number of taps used for the pulse shape filter and channel impulse response filter (integer)
    pulse_shape:    particular pulse shape to be used (1D tensor)
    L_link:         length of the SMF in meters (float) use if the channel presents CD
    R_sym:          symbol rate in Hz (float) use if the channel presents CD
    beta2:          beta2 parameter of the SMF in s^2/m (float)
    ch_imp_resp:    particular chanel impulse response to be used (1D tensor)
    rx_filt:        particular receiver filter to be used (1D tensor), 
                    if not specified uses a ideal LP filter simulate sampler BW, and if N_sim=N_os is a delta function
    '''
    constellation = None
    diff_encoder = None
    pulse_shape = None
    ch_imp_resp = None
    if {"mod_format", "M"} <= kwargs.keys():
        constellation = common_constellation(kwargs["mod_format"], kwargs["M"])
        if "diff_encoder" in kwargs.keys():
            if kwargs["diff_encoder"]:
                diff_encoder = common_diff_encoder(kwargs["mod_format"], constellation)
    elif "constellation" in kwargs.keys():
        constellation = kwargs["constellation"]    
    if {"alpha", "N_taps"} <= kwargs.keys():
        pulse_shape = rcos_filt(kwargs["alpha"], kwargs["N_taps"], N_sim, 1)
    elif "pulse_shape" in kwargs.keys():
        pulse_shape = kwargs["pulse_shape"]
    if {"L_link", "R_sym", "beta2", "N_taps"} <= kwargs.keys():
        ch_imp_resp = chrom_disp_filt(kwargs["L_link"], kwargs["R_sym"], kwargs["beta2"], kwargs["N_taps"], N_sim)
    elif "ch_imp_resp" in kwargs.keys():
        ch_imp_resp = kwargs["ch_imp_resp"]
    if "rx_filt" in kwargs.keys():
        rx_filt = kwargs["rx_filt"]
    elif N_sim > N_os:
        rx_filt = rcos_filt(0, len(pulse_shape), N_os, 1, dtype=torch.float32)
    else:
        rx_filt = torch.tensor([1.])
    return DD_system.DD_system(N_os, N_sim, constellation , diff_encoder, pulse_shape, ch_imp_resp, rx_filt)

