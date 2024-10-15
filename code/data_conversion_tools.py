import torch
import torch.nn.functional as F

from DD_system import DD_system


def mag_phase_2_complex(x, dd_system: DD_system):
    ''' Converts from mag phase representation to complex number
    
    Argument:
    x:      Tensor to convert (shape (batch_size, 2, N_sym) [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
    '''
    if not dd_system.multi_mag_const:
        return torch.exp(1j*x)
    if not dd_system.multi_phase_const:
        return x
    return x[:,0,:]*torch.exp(1j*x[:,1,:])

def SLDmag_phase_2_complex(x, dd_system: DD_system):
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
    if not dd_system.multi_mag_const:
        return torch.exp(1j*x)
    h0_tx=dd_system.tx_filt[0,0,dd_system.N_taps//2]
    h0_rx=torch.max(dd_system.rx_filt)
    if not dd_system.multi_phase_const:
        return torch.sign(x)*torch.sqrt(torch.abs(x)/h0_rx)/torch.abs(h0_tx)
    return torch.sign(x[:,0,:])*torch.sqrt(torch.abs(x[:,0,:])/h0_rx)/torch.abs(h0_tx)*torch.exp(1j*x[:,1,:])

def complex_2_mag_phase(x, dd_system: DD_system):
    ''' Converts from mag phase representation to complex number
    
    Argument:
    x:      Tensor to convert (shape (batch_size, 2, N_sym) [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
    '''
    if not dd_system.multi_mag_const:
        return torch.angle(x)
    if not dd_system.multi_phase_const:
        return torch.abs(x)
    return torch.cat((torch.abs(x),torch.angle(x)), dim=1)

def complex_2_SLDmag_phase(u, dd_system: DD_system):
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
    if not dd_system.multi_mag_const:
        return torch.angle(u)
    h0_tx=dd_system.tx_filt[0,0,dd_system.N_taps//2]
    h0_rx=torch.max(dd_system.rx_filt)
    if not dd_system.multi_phase_const:
        return torch.square(torch.abs(h0_tx*u))*h0_rx
    return torch.cat((torch.square(torch.abs(h0_tx*u))*h0_rx,torch.angle(u)), dim=1)

def idx_2_one_hot(idx):
    one_hot = F.one_hot(idx.squeeze(1))
    return one_hot.permute(0, 2, 1).to(torch.float32)  # Shape becomes (100, 4, 300)

def APPs_2_u(APPs, dd_system: DD_system):
    return dd_system.constellation[torch.argmax(APPs, dim=1, keepdim=True)]