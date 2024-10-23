import torch

import help_functions as hlp
import matplotlib.pyplot as plt

class Differential_encoder():
    '''
    Class implementing the differential encoder

    Methods:

    encode(u)

    Differential encoding process:
    given the ordered list of the possible phases that a symbol can have phase_list=[a_0,a_1,..a_n-1]
    define a function that returns the index at which the phase of a symbol is in the phase_list
        phase_to_idx(x)=i <==> phase_list[i]=a_i=angle(x)

    then the phase and magnitude of x_k are given by: 

    x_k_phase_idx = diff_mapping(phase_to_idx(u_k))
    
    angle(x_k) = phase_list[ x_k_phase_idex ]
    
    abs(x_k) = abs(u)
    '''

    def __init__(self, constellation, diff_mapping, device):
        '''
        Arguments:
        constellation:  constellation alphabet containing the possible symbols (1D tensor)
        diff_mapping:   function that apply the differential encoding, given the indices of the phases
        device:         the device to use (cpu or cuda)
        '''
        self.device= device
        self.constellation = constellation.to(device)
        self.diff_mapping = diff_mapping
        self.phase_list = torch.torch.unique(torch.round(torch.angle(constellation), decimals=10)).to(device)
        self.phase_list,_ = torch.sort(torch.remainder(self.phase_list,2*torch.pi))
        
    def encode(self, u: torch.Tensor): 
        '''
        Apply the differential encoding

        Arguments:
        u:      the incoming symbols (Tensor of size (batch_size, 1, N_sym))

        Returns:
        x:      output symbols (Tensor of size (batch_size, 1, N_sym))
        '''
        u_phase_idx = torch.argmin(torch.abs(self.phase_list[...,None]-torch.remainder(torch.angle(u),2*torch.pi)), dim=1, keepdim=True)
        x_phase_idx = self.diff_mapping(u_phase_idx)
        x = torch.abs(u)*torch.exp(1j*self.phase_list[x_phase_idx])
        return x

