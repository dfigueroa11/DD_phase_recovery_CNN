import numpy as np
import torch
from torch.nn.functional import conv1d

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

def chrom_disp_filt(L_link, R_sym, beta2, N_taps, N_sim, dtype=torch.cfloat): # expect odd N_taps, SI 
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
