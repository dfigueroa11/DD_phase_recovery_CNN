import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.axes as axes

from torch.nn.functional import conv1d
from sklearn.metrics import mutual_info_score


import DD_system
import Differential_encoder

def cascade_filters(filt_1, filt_2):
    '''combine 2 filters by multiplication in frequency domain

    Arguments: 
    filt_1, filt_2:     impulse response of the filters to combine, they must be of the same length (1D tensor)

    Return:
    filt_out:           combination of the filters (1D tensor of length equal to the input filters)
    '''
    FILT_1 = torch.fft.fft(torch.fft.ifftshift(filt_1))
    FILT_2 = torch.fft.fft(torch.fft.ifftshift(filt_2))
    return torch.fft.fftshift(torch.fft.ifft(FILT_1*FILT_2))

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

def norm_filt(N_sim, filt):
    ''' Returns the filter normalized to unitary energy, taking into acount the simulation oversampling

    Arguments:
    N_sim:      oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)
    filt:       filter to normalize (1D tensor)       
    '''
    filt = filt * torch.sqrt(N_sim / torch.sum(torch.abs(filt) ** 2))
    return filt

def filt_windowing(filt, energy_criteria=99):  
    ''' Returns the smallest window of the original filter centered around zero, than contains x% of the energy
    of the original filter, and the number of taps of such filter

    Returns: filt_w, N_taps

    Arguments:
    filt:               filter to window (1D tensor)
    energy_criteria:    float between 0 and 100, interpreted as a percentage (default 99%)
    '''  
    filt_len = torch.numel(filt)
    samp_idx = torch.arange(-(filt_len-1)/2,(filt_len-1)/2+1)
    energy_tot = torch.sum(torch.abs(filt)**2)
    energy_w = 0
    n = -1
    while energy_w < energy_tot*energy_criteria/100:
        n = n+1
        filt_w = filt[abs(samp_idx) <= n]
        energy_w = torch.sum(torch.abs(filt_w)**2)
    return filt_w, 2*n+1

def analyse_channel_length(N_os, N_sim, N_taps, alpha, L_link, R_sym, beta2=-2.168e-26, energy_criteria = 99):
    ''' Function to determine the number of required taps for the channel filter, assuming a raised cosine and a chromatic dispersion channel

    Arguments:
    N_os:               oversampling factor of the physical system (integer)
    N_sim:              oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)
    N_taps:             number of taps used for the test filter (integer)
    alpha:              roll off factor of a raised cosine filter used as a pulse shape (float in [0,1])
    L_link:             length of the SMF in meters (float) use if the channel presents CD
    R_sym:              symbol rate in Hz (float) use if the channel presents CD
    beta2:              beta2 parameter of the SMF in s^2/m (float default 2.168e-26)
    energy_criteria:    float between 0 and 100, interpreted as a percentage (default 99%)
    '''
    dd_system = set_up_DD_system(N_os= N_os, N_sim=N_sim,
                                N_taps=N_taps,
                                alpha=alpha, 
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

    filt, N_taps = filt_windowing(torch.squeeze(dd_system.tx_filt), energy_criteria)
    print(f"{N_taps} tap are needed to contain the {energy_criteria}% of the energy")
    plt.figure()
    t = np.arange(-np.floor(torch.numel(filt)/2),np.floor(torch.numel(filt)/2)+1)
    plt.stem(t, np.abs(filt)**2)
    t = np.arange(-np.floor(torch.numel(dd_system.tx_filt)/2),np.floor(torch.numel(dd_system.tx_filt)/2)+1)
    plt.stem(t, np.abs(torch.squeeze(dd_system.tx_filt))**2, linefmt=':')
    plt.show()

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
    rcos = np.where(np.abs(t_vec) == sym_time/(2*alpha), np.pi/4*np.sinc(1/(2*alpha)), \
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

def set_up_DD_system(N_os, N_sim, device, **kwargs):
    '''Returns a DD_system with the given configuration for common constellations,
    pulse shapes, channel impulse response and receiver filter

    Arguments:
    N_os:   oversampling factor of the physical system (integer)
    N_sim:  oversampling factor used during the simulation to avoid aliasing (integer multiple of N_os)
    device: the device to use (cpu or cuda)

    kwargs:
    mod_format:     constellation type (string: PAM, ASK, SQAM, QAM or DDQAM)
    M:              constellation order (integer) give together with mod_format
    sqrt_flag:      whether to apply the sqrt to all symbol's magnitudes or not (boolean default: False)
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
        if "sqrt_flag" in kwargs.keys():
            constellation = common_constellation(kwargs["mod_format"], kwargs["M"], sqrt_flag=kwargs["sqrt_flag"])
        else:
            constellation = common_constellation(kwargs["mod_format"], kwargs["M"])
        if "diff_encoder" in kwargs.keys():
            if kwargs["diff_encoder"]:
                diff_encoder = common_diff_encoder(kwargs["mod_format"], constellation, device)
    elif "constellation" in kwargs.keys():
        constellation = kwargs["constellation"]    
    if {"alpha", "N_taps"} <= kwargs.keys():
        pulse_shape = rcos_filt(kwargs["alpha"], kwargs["N_taps"], N_sim, 1)
        N_sim = 4 if kwargs["alpha"] > 0 else N_sim
    elif "pulse_shape" in kwargs.keys():
        pulse_shape = kwargs["pulse_shape"]
    if {"L_link", "R_sym", "beta2", "N_taps"} <= kwargs.keys():
        ch_imp_resp = chrom_disp_filt(kwargs["L_link"], kwargs["R_sym"], kwargs["beta2"], kwargs["N_taps"], N_sim)
    elif "ch_imp_resp" in kwargs.keys():
        ch_imp_resp = kwargs["ch_imp_resp"]
    if "rx_filt" in kwargs.keys():
        rx_filt = kwargs["rx_filt"]
    elif N_sim > N_os:
        rx_filt = rcos_filt(0, len(pulse_shape), N_sim, 1/2, dtype=torch.float32)
    else:
        rx_filt = torch.tensor([1.])
    return DD_system.DD_system(N_os, N_sim, constellation , diff_encoder, pulse_shape, ch_imp_resp, rx_filt, device)

def common_constellation(mod, M, dtype=torch.cfloat, sqrt_flag=False):
    '''Returns the constellation specified (1D tensor of size M)

    Arguments:
    mod:        String with the modulation format, valid options are 'PAM', 'ASK', 'SQAM', 'QAM' or 'DDQAM'
    M:          order of the modulation
    dtype:      data type (optional, default torch.cfloat)
    sqrt_flag:  whether to apply the sqrt to all symbol's magnitudes or not (boolean default: False)
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

    if sqrt_flag:
        constellation = np.sqrt(np.abs(constellation))*constellation/(np.abs(constellation)+1e-30)
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    return torch.tensor(constellation, dtype=dtype)

def common_diff_encoder(mod, constellation, device):
    '''Returns the constellation specified (1D tensor of size M)

    Arguments:
    mod:            String with the modulation format, valid options are 'PAM', 'ASK', 'SQAM', 'QAM' or 'DDQAM'
    constellation:  constellation:  constellation to be used (1D tensor)
    device:         the device to use (cpu or cuda)
    '''
    if mod == "PAM":
        return None
    elif mod == "ASK":
        def diff_mapping(u_ph_idx):
            return torch.remainder(torch.cumsum(u_ph_idx, dim=-1),2)
    elif mod == "SQAM":
        def diff_mapping(u_ph_idx):
            return torch.remainder(torch.cumsum(u_ph_idx, dim=-1),4)
    elif mod == "QAM":
        def diff_mapping(u_ph_idx):
            return torch.remainder(torch.cumsum(u_ph_idx, dim=-1),4)
    elif mod == "DDQAM":
        diff_mapping = torch.tensor([[1,0,3,2],[0,1,2,3],[3,2,1,0],[2,3,0,1]])  ## update
    else:
        raise ValueError("mod should be PAM, ASK, SQAM, QAM or DDQAM")
    return Differential_encoder.Differential_encoder(constellation, diff_mapping, device)

def DD_1sym_ISI(x, h0=1, h1=1/2, device='cpu'):
    '''Apply ideal DD for a channel with one symbol ISI, with Tx_filter = [h1, h0, h1]
    
    Arguments:
    x:      input signal (tensor of size (batch_size, 1, N_sym))
    h0:     center filter coefficient, default: 1
    h1:     side filter coefficients, default: 1/2
    device: the device to use (cpu or cuda)

    Returns:
    y:      signal (tensor of size ((batch_size, 1, 2*N_sym))
    '''
    y_1sym_ISI = torch.zeros(x.size(dim=0), x.size(dim=1), 2*x.size(dim=2), device=device)
    y_1sym_ISI[:,:,1::2] = torch.square(torch.abs(h0*x))
    y_1sym_ISI[:,:,0::2] = torch.square(torch.abs(h1*x+h1*torch.roll(x, 1, dims=-1)))
    return y_1sym_ISI

def create_ideal_y(u, multi_mag, multi_phase, h0_tx=1, h0_rx=1):
    ''' creates the ideal output of the CNN for given symbols u

    Arguments:
    u:              transmitted symbols estimates (shape (batch_size, 1, N_sym))
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    h0_tx:          center tap of the transmitter filter, used to scale y_hat properly
    h0_rx:          center tap of the receiver filter, used to scale y_hat properly

    Returns:
    y:              ideal output of CNN (shape (batch_size, 1 or 2, N_sym) depending on multi_mag, multi_phase)
    '''
    if not multi_mag:
        return torch.angle(u)
    if not multi_phase:
        return torch.square(torch.abs(h0_tx*u))*h0_rx
    return torch.cat((torch.square(torch.abs(h0_tx*u))*h0_rx,torch.angle(u)), dim=1)

def y_hat_2_u_hat(y_hat, multi_mag, multi_phase, h0_tx=1, h0_rx=1):
    ''' Converts the output of the CNN to the estimates of the transmitted symbols u

    Arguments:
    y_hat:          output of the CNN (shape (batch_size, 1 or 2, N_sym) depending on multi_mag, multi_phase)
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    h0_tx:          center tap of the transmitter filter, used to scale y_hat properly
    h0_rx:          center tap of the receiver filter, used to scale y_hat properly

    Returns:
    u_hat:          transmitted symbols estimates (shape (batch_size, 1, N_sym))
    '''
    if not multi_mag:
        return torch.exp(1j*y_hat)
    if not multi_phase:
        return torch.sqrt(torch.abs(y_hat)/h0_rx)/torch.abs(h0_tx)
    return torch.sqrt(torch.abs(y_hat[:,0,:])/h0_rx)/torch.abs(h0_tx)*torch.exp(1j*y_hat[:,1,:])

def abs_phase_diff(x, dim=-1):
    '''Computes the phase difference between adjacent symbols
    
    Arguments:
    x:      input signal (tensor of size (batch_size, 1, N_sym))
    dim:    dimension to do the phase difference (default -1)

    Returns:
    signal (tensor of size ((batch_size, 1, N_sym))
    '''
    return torch.abs(torch.remainder(torch.abs(torch.diff(torch.angle(x))+torch.pi),2*torch.pi)-torch.pi)

def mag_phase_2_complex(x):
    ''' Converts from mag phase representation to complex number
    
    Argument:
    x:      Tensor to convert (shape (batch_size, 2, N_sym) [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
    '''
    return x[:,0,:]*torch.exp(1j*x[:,1,:])

def get_ER(Tx, Rx, tol=1e-5):
    ''' Calculate the error rate between Tx and Rx with a given tolerance, that means count
    the number of positions where Tx and Rx differ, and divide by the number of elements

    Arguments:
    Tx:     tensor of transmitted elements (same size as Rx)
    Rx:     tensor of received elements (same size as Tx)
    tol:    tolerance, so Tx is considered equal to Rx if |Tx-Rx| < tol (float, default 1e-5)
    '''
    assert Tx.size() == Rx.size() , "Tx and Rx must have the same size"
    return torch.sum(abs(Tx-Rx)>tol)/torch.numel(Tx)

def min_distance_dec(alphabet, Rx):
    ''' Decodes the elements in Rx using minimum distance criteria with respect to the elements in alphabet

    Arguments:
    alphabet:   1D tensor with the elements to determine the minimum distance
    Rx:         tensor with the symbols to decode (arbitrary size)

    Return:
    hard_dec_idx:   index if the symbol in alphabet
    sym_hat:        decoded symbols
    '''
    hard_dec_idx = torch.argmin(torch.abs(alphabet - Rx[...,None]), dim=-1)
    return hard_dec_idx, alphabet[hard_dec_idx]

def decode_and_ER(Tx, Rx, precision=5):
    ''' Decodes under minimum distance criteria and calculates the error rate between Tx and Rx

    Arguments:
    Tx:         noiseless transmitted symbols (the alphabet is computed from Tx ass all different values that Tx take, that is why Tx must be noiseless)
    Rx:         received symbols (same size as Tx)
    precision:  number of decimals used to determine the alphabet in the rounding process (default=5)

    Return:
    decoding alphabet (1D tensor), error rate
    '''
    alphabet = torch.unique(torch.round(torch.flatten(Tx), decimals=precision))
    _, Rx_deco = min_distance_dec(alphabet, Rx)
    return alphabet, get_ER(Tx,Rx_deco)

def decode_and_ER_mag_phase(Tx, Rx, precision=5):
    ''' Decodes under minimum distance criteria and calculates the error rate between Tx and Rx

    Arguments:
    Tx:         noiseless transmitted symbols (shape (batch_size, 2, N_sym) [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
                (the alphabet is computed from Tx ass all different values that Tx take, that is why Tx must be noiseless)
    Rx:         received symbols (same shape as Tx)
    precision:  number of decimals used to determine the alphabet in the rounding process (default=5)

    Return:
    decoding alphabet (1D tensor), error rate
    '''
    alphabet_mag = torch.unique(torch.round(torch.flatten(Tx[:,0,:]), decimals=precision))
    alphabet_phase = torch.unique(torch.round(torch.flatten(Tx[:,1,:]), decimals=precision))
    alphabet = (alphabet_mag*torch.exp(1j*alphabet_phase)[...,None]).flatten()
    Tx = mag_phase_2_complex(Tx)
    Rx = mag_phase_2_complex(Rx)
    _, Rx_deco = min_distance_dec(alphabet, Rx)
    return alphabet, get_ER(Tx,Rx_deco,tol=torch.min(alphabet_mag)/10)

def calc_progress(y_ideal, y_hat, multi_mag, multi_phase):
    '''Print the training progress

    Arguments:
    y_ideal:        Tensor containing the ideal magnitudes and phase differences (shape (batch_size, 1 or 2, N_sym) depending on multi_mag, multi_phase)
    y_hat:          output of the CNN (same shape as y_ideal)
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not

    Returns:
    alphabets:      [mag alphabet, phase alphabet, symbol alphabet] or [alphabet] depending on multi_mag, multi_phase
    SERs:           [mag ER, phase, ER, SER] or [SER] depending on multi_mag, multi_phase
    '''
    if multi_mag and multi_phase:
        alphabet_mag, mag_ER = decode_and_ER(y_ideal[:,0,:], y_hat[:,0,:])
        alphabet_phase, phase_ER = decode_and_ER(y_ideal[:,1,:], y_hat[:,1,:])
        alphabet, SER = decode_and_ER_mag_phase(y_ideal, y_hat)
        alphabets = [alphabet_mag, alphabet_phase, alphabet]
        SERs = [mag_ER, phase_ER, SER]
    else:
        alphabet, SER = decode_and_ER(y_ideal, y_hat)
        alphabets = [alphabet]
        SERs = [SER]
    return alphabets, SERs

def get_MI(u, u_hat, constellation, Ptx_dB):
    ''' Calculates the mutual information between u and u_hat (1D arrays) after minimum distance 
    hard decoding with respect to constellation (1D tensor) with a transmit power of Ptx_dB
    '''
    Ptx_lin = torch.tensor([10**(Ptx_dB/10)], dtype=torch.float32)
    constellation = torch.sqrt(Ptx_lin)*constellation
    u_idx, _ = min_distance_dec(constellation, u.flatten())
    u_hat_idx, _ = min_distance_dec(constellation, u_hat.flatten())
    return mutual_info_score(u_hat_idx, u_idx)/np.log(2)
