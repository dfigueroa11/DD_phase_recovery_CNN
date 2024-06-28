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

    x_k_phase_idx = diff_mapping[ phase_to_idx(u_k), phase_to idx(x_k-1) ]
    
    angle(x_k) = phase_list[ x_k_phase_idex ]
    
    abs(x_k) = abs(u)
    
    where diff_mapping is a (n x n) matrix defining the phase transitions depending on
    the previous symbol x_k-1 and the current symbol u_k, but in terms of the indexes
    
    for example a classic DBPSK is described by:
    phase_list = [pi,0]
    diff_mapping = [[1,0],[0,1]] 
    '''

    def __init__(self, constellation, diff_mapping):
        '''
        Arguments:
        constellation:  constellation alphabet containing the possible symbols (1D tensor)
        diff_mapping:   square matrix describing the differential encoding (2D tensor ptype int)
        '''
        self.constellation = constellation
        self.diff_mapping = diff_mapping
        self.phase_list = torch.unique(torch.round(torch.angle(constellation), decimals=10)).view(-1,1)
        
    def encode(self, u, x0_phase_idx=0): 
        '''
        Apply the differential encoding 

        Arguments:
        u:              the incoming symbols (Tensor of size (batch_size, 1, N_sym))
        x0_phase_idx:   the index of the initial phase (integer, default 0)
        '''
        u_phase_idx = torch.argmin(torch.abs(self.phase_list-torch.angle(u)), dim=1, keepdim=True)
        x = torch.empty_like(u)
        x_prev_phase_idx = x0_phase_idx*torch.ones(u.size(dim=0), dtype=torch.int)
        for i in range(u.size(dim=-1)):
            x_prev_phase_idx = self.diff_mapping[u_phase_idx[:,0,i],x_prev_phase_idx]
            x[:,:,i] = torch.abs(u[:,:,i])*torch.exp(1j*self.phase_list[x_prev_phase_idx])
        return x

