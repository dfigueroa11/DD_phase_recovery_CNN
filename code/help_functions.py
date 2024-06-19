import torch
from torch.nn.functional import conv1d

def cascade_filters(filt_1, filt_2):
    FILT_1 =  torch.fft.fft(torch.fft.ifftshift(filt_1))
    FILT_2 =  torch.fft.fft(torch.fft.ifftshift(filt_2))
    return  torch.fft.fftshift(torch.fft.ifft(FILT_1*FILT_2))

def convolve(signal, filt):
    filt = torch.resolve_conj(torch.flip(filt, [-1]))
    return conv1d(signal, filt, padding='same')
