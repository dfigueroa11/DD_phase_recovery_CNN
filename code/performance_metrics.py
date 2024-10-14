import numpy as np
import torch

from sklearn.metrics import mutual_info_score

from DD_system import DD_system


def min_distance_dec(Rx, alphabet):
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

def decode_and_SER(Tx, Rx, alphabet):
    Tx_idx, _ = min_distance_dec(Tx, alphabet)
    Rx_idx, _ = min_distance_dec(Rx, alphabet)
    return torch.sum(Tx_idx != Rx_idx)/torch.numel(Tx_idx)

def get_alphabets(dd_system: DD_system):
    const = dd_system.constellation.detach().cpu()
    mag_alphabet = torch.unique(torch.round(torch.abs(const), decimals=4))
    phase_alphabet = torch.unique(torch.round(torch.angle(const/torch.abs(const)), decimals=4))
    return mag_alphabet, phase_alphabet, const

def get_all_SERs(u, u_hat, dd_system: DD_system):
    SERs = -torch.ones(3)
    mag_alphabet, phase_alphabet, const = get_alphabets(dd_system)
    if dd_system.multi_mag_const:
        SERs[0] = decode_and_SER(torch.abs(u), torch.abs(u_hat), mag_alphabet)
    if dd_system.multi_phase_const:
        SERs[1] = decode_and_SER(u/torch.abs(u), u_hat/torch.abs(u_hat), phase_alphabet)
    SERs[2] = decode_and_SER(u, u_hat, const)
    return SERs

def get_MI(u, u_hat, constellation, Ptx_dB):
    ''' Calculates the mutual information between u and u_hat (1D arrays) after minimum distance 
    hard decoding with respect to constellation (1D tensor) with a transmit power of Ptx_dB
    '''
    u_idx, _ = min_distance_dec(u.flatten(), constellation)
    u_hat_idx, _ = min_distance_dec(u_hat.flatten(), constellation)
    return mutual_info_score(u_hat_idx, u_idx)/np.log(2)
