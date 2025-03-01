import torch
from torch.nn.functional import mse_loss, cross_entropy

from . import data_conversion_tools as dconv_tools
from comm_sys.DD_system import DD_system
from nn_equalizers import cnn_equalizer, fcn_ph_equalizer

################ CNN #####################

def MSE_u_symbols_2_cnn_out(u_idx: torch.Tensor, u: torch.Tensor, cnn_out: torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the complex symbols
    
    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, 1|2, N_sym)

    Returns:
    loss
    '''
    u_hat_cnn = dconv_tools.mag_phase_2_complex(cnn_out, dd_system)
    return torch.mean(torch.square(torch.abs(u - u_hat_cnn)))

def MSE_u_mag_phase_2_cnn_out(u_idx: torch.Tensor, u: torch.Tensor, cnn_out: torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the magnitude and phase of the symbols
    
    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, 1|2, N_sym)

    Returns:
    loss
    '''
    u_mag_phase = dconv_tools.complex_2_mag_phase(u, dd_system)
    return mse_loss(u_mag_phase, cnn_out)

def MSE_u_mag_phase_2_cnn_out_phase_fix(u_idx: torch.Tensor, u: torch.Tensor, cnn_out:torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the magnitude and phase of the symbols using the phase fix to avoid asymmetries
    
    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, 1|2, N_sym)

    Returns:
    loss
    '''
    u_mag_phase = dconv_tools.complex_2_mag_phase(u, dd_system)
    loss_ph = 0
    loss_mag = 0
    if dd_system.multi_phase_const:
        phase_diff = torch.abs(torch.remainder(torch.abs(u_mag_phase[:,-1,:]-cnn_out[:,-1,:]+torch.pi),2*torch.pi)-torch.pi)
        loss_ph = torch.mean(torch.square(phase_diff))
    if dd_system.multi_mag_const:
        loss_mag = mse_loss(u_mag_phase[:,0,:], cnn_out[:,0,:])
    return (loss_mag+loss_ph)/cnn_out.size(dim=1)

def MSE_u_SLDmag_phase_2_cnn_out(u_idx: torch.Tensor, u: torch.Tensor, cnn_out: torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the magnitude (after SLD and channel amplification) and phase of the symbols
    
    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, 1|2, N_sym)

    Returns:
    loss
    '''
    u_SLDmag_phase = dconv_tools.complex_2_SLDmag_phase(u, dd_system)
    return mse_loss(u_SLDmag_phase, cnn_out)

def MSE_u_SLDmag_phase_2_cnn_out_phase_fix(u_idx: torch.Tensor, u: torch.Tensor, cnn_out:torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the magnitude (after SLD and channel amplification) and phase of the symbols
    using the phase fix to avoid asymmetries 

    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, 1|2, N_sym)

    Returns:
    loss
    '''
    u_mag_phase = dconv_tools.complex_2_SLDmag_phase(u, dd_system)
    loss_ph = 0
    loss_mag = 0
    if dd_system.multi_phase_const:
        phase_diff = torch.abs(torch.remainder(torch.abs(u_mag_phase[:,-1,:]-cnn_out[:,-1,:]+torch.pi),2*torch.pi)-torch.pi)
        loss_ph = torch.mean(torch.square(phase_diff))
    if dd_system.multi_mag_const:
        loss_mag = mse_loss(u_mag_phase[:,0,:], cnn_out[:,0,:])
    return (loss_mag+loss_ph)/cnn_out.size(dim=1)

def ce_u_symbols_cnn_out(u_idx: torch.Tensor, u: torch.Tensor, cnn_out: torch.Tensor, dd_system: DD_system):
    ''' Calculates the loss such that the CNN_output tends towards the APPs approximated by the one hot notation
    
    Arguments:
    u_idx:      Tensor with the index of the transmitted symbols (shape (batch_size, 1, N_sym))
    u:          Tensor with the transmitted symbols (shape (batch_size, 1, N_sym))
    cnn_out:    Tensor with the output of the CNN with shape (batch_size, M, N_sym) M: modulation order

    Returns:
    loss
    '''
    return cross_entropy(input=cnn_out, target=u_idx.squeeze())

loss_funcs_cnn = {cnn_equalizer.TRAIN_MSE_U_SYMBOLS: MSE_u_symbols_2_cnn_out,
                  cnn_equalizer.TRAIN_MSE_U_MAG_PHASE: MSE_u_mag_phase_2_cnn_out,
                  cnn_equalizer.TRAIN_MSE_U_MAG_PHASE_PHASE_FIX: MSE_u_mag_phase_2_cnn_out_phase_fix,
                  cnn_equalizer.TRAIN_MSE_U_SLDMAG_PHASE: MSE_u_SLDmag_phase_2_cnn_out,
                  cnn_equalizer.TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX: MSE_u_SLDmag_phase_2_cnn_out_phase_fix,
                  cnn_equalizer.TRAIN_CE_U_SYMBOLS: ce_u_symbols_cnn_out}

################ FCN #####################
def MSE_fcn(u: torch.Tensor, fcn_out: torch.Tensor, dd_system: DD_system):
    return mse_loss(fcn_out.squeeze(), torch.angle(u[:,u.shape[-1]//2]))

def MSE_fcn_phase_fix(u: torch.Tensor, fcn_out: torch.Tensor, dd_system: DD_system):
    phase_diff = torch.abs(torch.remainder(torch.abs(torch.angle(u[:,u.shape[-1]//2])-fcn_out.squeeze()+torch.pi),2*torch.pi)-torch.pi)
    return torch.mean(torch.square(phase_diff))

def CE_fcn(u: torch.Tensor, fcn_out: torch.Tensor, dd_system: DD_system):
    phase_idx = torch.argmin(torch.abs(dd_system.phase_list[...,None]-torch.remainder(torch.angle(u[:,u.shape[-1]//2]),2*torch.pi)), dim=0)
    return cross_entropy(input=fcn_out, target=phase_idx)

loss_funcs_fcn = {fcn_ph_equalizer.TRAIN_MSE: MSE_fcn,
                  fcn_ph_equalizer.TRAIN_MSE_PHASE_FIX: MSE_fcn_phase_fix,
                  fcn_ph_equalizer.TRAIN_CE: CE_fcn}