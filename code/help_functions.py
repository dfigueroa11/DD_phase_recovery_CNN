import numpy as np

import torch
from torch.nn.functional import conv1d

def cascade_filters(filt_1, filt_2):
    FILT_1 =  torch.fft.fft(torch.fft.ifftshift(filt_1))
    FILT_2 =  torch.fft.fft(torch.fft.ifftshift(filt_2))
    return  torch.fft.fftshift(torch.fft.ifft(FILT_1*FILT_2))

def convolve(signal, filt):
    filt = torch.resolve_conj(torch.flip(filt, [-1]))
    return conv1d(signal, filt, padding='same')

def common_constellation(mod, M):
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
        ValueError("mod should be PAM, ASK, SQAM, QAM or DDQAM")

    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    return torch.tensor(constellation)

def rcos_filt(alpha, len, fs, sym_time):
    t_vec = (np.arange(len)-(len-1)/2)/fs
    if alpha == 0:
        return torch.tensor(np.sinc(t_vec/sym_time))
    rcos =  np.where(np.abs(t_vec) == sym_time/(2*alpha), np.pi/4*np.sinc(1/(2*alpha)), \
                     np.sinc(t_vec/sym_time)*(np.cos(np.pi*alpha*t_vec/sym_time))/(1-(2*alpha*t_vec/sym_time)**2))
    return torch.tensor(rcos)