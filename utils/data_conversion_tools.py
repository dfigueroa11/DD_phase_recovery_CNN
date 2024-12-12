import torch
import torch.nn.functional as F

from comm_sys.DD_system import DD_system

############### CNN ###############
def mag_phase_2_complex(x: torch.Tensor, dd_system: DD_system, **kwargs):
    ''' Converts from mag phase representation to complex number
    
    Arguments:
    x:          Tensor to convert (shape (batch_size, 1|2, N_sym), [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
    dd_system:  DD_system used to generate the data

    Returns:
    y:      Tensor of same shape (batch_size, 1, N_sym)
    '''
    if not dd_system.multi_mag_const:
        return torch.exp(1j*x)
    if not dd_system.multi_phase_const:
        return x
    return x[:,0:1,:]*torch.exp(1j*x[:,1:,:])

def SLDmag_phase_2_complex(x: torch.Tensor, dd_system: DD_system, **kwargs):
    ''' Converts from mag phase representation to complex number taking into acount the SLD operator and the channel amplification
    
    Arguments:
    x:          Tensor to convert (shape (batch_size, 1|2, N_sym), [:,0,:] interpreted as mag, [:,1,:] interpreted as phase)
    dd_system:  DD_system used to generate the data

    Returns:
    y:      Tensor of same shape (batch_size, 1, N_sym)
    '''
    if not dd_system.multi_mag_const:
        return torch.exp(1j*x)
    h0_tx = dd_system.tx_filt[0,0,dd_system.N_taps//2].cpu()
    h0_rx = torch.max(dd_system.rx_filt).cpu()
    if not dd_system.multi_phase_const:
        return torch.sign(x)*torch.sqrt(torch.abs(x)/h0_rx)/torch.abs(h0_tx)
    return torch.sign(x[:,0:1,:])*torch.sqrt(torch.abs(x[:,0:1,:])/h0_rx)/torch.abs(h0_tx)*torch.exp(1j*x[:,1:,:])

def complex_2_mag_phase(x: torch.Tensor, dd_system: DD_system):
    ''' Converts from complex representation to mag phase
    
    Arguments:
    x:          Tensor to convert (shape (batch_size, 1, N_sym))
    dd_system:  DD_system used to generate the data

    Returns:
    y:      Tensor of shape (batch_size, 1|2, N_sym) depending on the characteristics of DD_system
    '''
    if not dd_system.multi_mag_const:
        return torch.angle(x)
    if not dd_system.multi_phase_const:
        return torch.abs(x)
    return torch.cat((torch.abs(x),torch.angle(x)), dim=1)

def complex_2_SLDmag_phase(u: torch.Tensor, dd_system: DD_system):
    ''' Converts from complex representation to mag phase taking into acount the SLD operator and the channel amplification
    
    Arguments:
    u:          Tensor to convert (shape (batch_size, 1, N_sym))
    dd_system:  DD_system used to generate the data

    Returns:
    y:      Tensor of shape (batch_size, 1|2, N_sym) depending on the characteristics of DD_system
    '''
    if not dd_system.multi_mag_const:
        return torch.angle(u)
    h0_tx=dd_system.tx_filt[0,0,dd_system.N_taps//2]
    h0_rx=torch.max(dd_system.rx_filt)
    if not dd_system.multi_phase_const:
        return torch.square(torch.abs(h0_tx*u))*h0_rx
    return torch.cat((torch.square(torch.abs(h0_tx*u))*h0_rx,torch.angle(u)), dim=1)

def idx_2_one_hot(idx: torch.Tensor):
    ''' Converts indices to one hot representation
    
    Arguments:
    idx:          indices to convert (shape (batch_size, 1, N_sym))

    Returns:
    one_hot:      Tensor of shape (batch_size, N_class, N_sym) where N_class is the number of different classes or indices
    '''
    one_hot = F.one_hot(idx.squeeze(1))
    return one_hot.permute(0, 2, 1).to(torch.float32)

def APPs_2_u(APPs: torch.Tensor, dd_system: DD_system, Ptx_dB: float=0):
    ''' Converts APPs into a symbol estimate using the maximum likelihood rule

    Arguments:
    APPs:       Tensor of shape (batch_size, N_class, N_sym)
    dd_system:  DD_system used to generate the data
    Ptx_dB:     transmitted power (float)

    Returns:
    u_hat:      Tensor of shape (batch_size, 1, N_sym)
    '''
    Ptx_lin = torch.tensor([10**(Ptx_dB/10)], dtype=torch.float32)
    return torch.sqrt(Ptx_lin)*dd_system.constellation[torch.argmax(APPs, dim=1, keepdim=True)].cpu()

############### FCN ###############
def reshape_data_for_FCN(ui: torch.Tensor, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor, a_len):
    N_os = y.shape[-1]//u.shape[-1]
    ui = torch.reshape(ui.squeeze()[:,a_len:-a_len], (-1,a_len))
    u = torch.reshape(u.squeeze()[:,a_len:-a_len], (-1,a_len))
    x = torch.reshape(x.squeeze()[:,a_len:-a_len], (-1,a_len))
    y = torch.reshape(y.squeeze()[:,N_os*a_len:-N_os*a_len], (-1,N_os*a_len))
    return ui, u, x, y

def MSE_FCN_out_2_complex(fcn_out: torch.Tensor, a: torch.Tensor, dd_system: DD_system):
    return a*torch.exp(1j*fcn_out.squeeze())

def CE_FCN_out_2_complex(fcn_out: torch.Tensor, a: torch.Tensor, dd_system: DD_system):
    return a*torch.exp(1j*dd_system.phase_list[fcn_out.argmax(dim=-1)]).detach().cpu()