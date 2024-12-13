import numpy as np
import torch

from sklearn.metrics import mutual_info_score

from comm_sys.DD_system import DD_system


def min_distance_dec(Rx: torch.Tensor, alphabet: torch.Tensor):
    ''' Decodes the elements in Rx using minimum distance criteria with respect to the elements in alphabet

    Arguments:
    Rx:         tensor with the symbols to decode (arbitrary shape)
    alphabet:   1D tensor with the elements to determine the minimum distance

    Returns:
    hard_dec_idx:   index if the symbol in alphabet
    sym_hat:        decoded symbols same shape as Rx
    '''
    hard_dec_idx = torch.argmin(torch.abs(alphabet - Rx[...,None]), dim=-1)
    return hard_dec_idx, alphabet[hard_dec_idx]

def decode_and_SER(Tx: torch.Tensor, Rx: torch.Tensor, alphabet: torch.Tensor):
    ''' Decodes the received symbols using minimum distance criteria with respect to the elements in alphabet, and calculates the SER

    Arguments:
    Tx:         transmitted symbols Tensor with arbitrary shape
    Rx:         received symbols Tensor with same shape as Tx
    alphabet:   1D tensor with the elements to determine the minimum distance

    Returns:
    SER:    float
    '''
    Tx_idx, _ = min_distance_dec(Tx, alphabet)
    Rx_idx, _ = min_distance_dec(Rx, alphabet)
    return torch.sum(Tx_idx != Rx_idx)/torch.numel(Tx_idx)

def get_alphabets(dd_system: DD_system, Ptx_dB: float):
    ''' returns the alphabets used for modulation, taking into account the transmitted power

    Returns:
    mag_alphabet, phase_alphabet, constellation
    '''
    Ptx_lin = torch.tensor([10**(Ptx_dB/10)], dtype=torch.float32)
    constellation = torch.sqrt(Ptx_lin)*dd_system.constellation.detach().cpu()
    mag_alphabet = torch.unique(torch.round(torch.abs(constellation), decimals=4))
    phase_alphabet = torch.unique(torch.round(torch.angle(constellation/torch.abs(constellation)), decimals=4))
    return mag_alphabet, phase_alphabet, constellation

def get_all_SERs(u: torch.Tensor, u_hat: torch.Tensor, dd_system: DD_system, Ptx_dB: float):
    ''' returns the magnitude error rate, phase error rate and SER
    
    Arguments:
    u:      tensor with the transmitted symbols (arbitrary shape)
    u_hat:  tensor with the received symbols (same shape as u)

    Returns:
    [mag_ER, ph_ER, SER]
    '''
    SERs = -torch.ones(3)
    mag_alphabet, phase_alphabet, const = get_alphabets(dd_system, Ptx_dB)
    if dd_system.multi_mag_const and dd_system.multi_phase_const:
        SERs[0] = decode_and_SER(torch.abs(u), torch.abs(u_hat), mag_alphabet)
        SERs[1] = decode_and_SER(u/torch.abs(u), u_hat/torch.abs(u_hat), torch.exp(1j*phase_alphabet))
    SERs[2] = decode_and_SER(u, u_hat, const)
    return SERs

def get_MI_HD(u: torch.Tensor, u_hat: torch.Tensor, dd_system: DD_system, Ptx_dB: float):
    ''' Calculates the mutual information between u and u_hat after minimum distance hard decoding

    Arguments:
    u:      tensor with the transmitted symbols (arbitrary shape)
    u_hat:  tensor with the received symbols (same shape as u)

    Returns:
    mutual information
    '''
    _, _, constellation = get_alphabets(dd_system, Ptx_dB)
    u_idx, _ = min_distance_dec(u.flatten(), constellation)
    u_hat_idx, _ = min_distance_dec(u_hat.flatten(), constellation)
    return mutual_info_score(u_hat_idx, u_idx)/np.log(2)

def get_MI_SD(u:np.ndarray, APPs:np.ndarray, dd_system: DD_system, Ptx_dB: float):
    ''' Calculates the mutual information between u_idx and APPs

    Arguments:
    u_idx:  ndarray with the index of the transmitted symbols (1D array of length N_sym)
    APPs:   ndarray with the APPs for each class and symbol (shape (N_sym, M) M: modulation order)

    Returns:
    mutual information
    '''
    _, _, constellation = get_alphabets(dd_system, Ptx_dB)
    u_idx, _ = min_distance_dec(u, constellation)
    p_t = np.array([np.mean(u_idx.numpy()==i) for i in range(APPs.shape[1])])
    return np.mean(np.log2(np.take_along_axis(APPs.numpy(),u_idx.numpy(),1)).squeeze() - np.log2(np.sum(APPs.numpy()*p_t[...,None],axis=1)))
