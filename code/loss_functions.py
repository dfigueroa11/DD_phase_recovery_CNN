import torch
from torch.nn.functional import mse_loss, cross_entropy

import data_conversion_tools as dconv_tools
from DD_system import DD_system
import cnn_equalizer

def MSE_u_symbols_2_cnn_out(u_idx, u, cnn_out, dd_system: DD_system):
    u_hat_cnn = dconv_tools.mag_phase_2_complex(cnn_out, dd_system)
    return torch.mean(torch.square(torch.abs(u - u_hat_cnn)))

def MSE_u_mag_phase_2_cnn_out(u_idx, u, cnn_out, dd_system: DD_system):
    u_mag_phase = dconv_tools.complex_2_mag_phase(u, dd_system)
    return mse_loss(u_mag_phase, cnn_out)

def MSE_u_mag_phase_2_cnn_out_phase_fix(u_idx, u, cnn_out:torch.Tensor, dd_system: DD_system):
    u_mag_phase = dconv_tools.complex_2_mag_phase(u, dd_system)
    loss_ph = 0
    loss_mag = 0
    if dd_system.multi_phase_const:
        phase_diff = torch.abs(torch.remainder(torch.abs(u_mag_phase[:,-1,:]-cnn_out[:,-1,:]+torch.pi),2*torch.pi)-torch.pi)
        loss_ph = torch.mean(torch.square(phase_diff))
    if dd_system.multi_mag_const:
        loss_mag = mse_loss(u_mag_phase[:,0,:], cnn_out[:,0,:])
    return (loss_mag+loss_ph)/cnn_out.size(dim=1)

def MSE_u_SLDmag_phase_2_cnn_out(u_idx, u, cnn_out, dd_system: DD_system):
    u_SLDmag_phase = dconv_tools.complex_2_SLDmag_phase(u, dd_system)
    return mse_loss(u_SLDmag_phase, cnn_out)

def MSE_u_SLDmag_phase_2_cnn_out_phase_fix(u_idx, u, cnn_out:torch.Tensor, dd_system: DD_system):
    u_mag_phase = dconv_tools.complex_2_SLDmag_phase(u, dd_system)
    loss_ph = 0
    loss_mag = 0
    if dd_system.multi_phase_const:
        phase_diff = torch.abs(torch.remainder(torch.abs(u_mag_phase[:,-1,:]-cnn_out[:,-1,:]+torch.pi),2*torch.pi)-torch.pi)
        loss_ph = torch.mean(torch.square(phase_diff))
    if dd_system.multi_mag_const:
        loss_mag = mse_loss(u_mag_phase[:,0,:], cnn_out[:,0,:])
    return (loss_mag+loss_ph)/cnn_out.size(dim=1)

def ce_u_symbols_cnn_out(u_idx, u, cnn_out, dd_system: DD_system):
    u_one_hot = dconv_tools.idx_2_one_hot(u_idx)
    return cross_entropy(input=cnn_out, target=u_one_hot)

loss_funcs = {cnn_equalizer.TRAIN_MSE_U_SYMBOLS: MSE_u_symbols_2_cnn_out,
              cnn_equalizer.TRAIN_MSE_U_MAG_PHASE: MSE_u_mag_phase_2_cnn_out,
              cnn_equalizer.TRAIN_MSE_U_MAG_PHASE_PHASE_FIX: MSE_u_mag_phase_2_cnn_out_phase_fix,
              cnn_equalizer.TRAIN_MSE_U_SLDMAG_PHASE: MSE_u_SLDmag_phase_2_cnn_out,
              cnn_equalizer.TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX: MSE_u_SLDmag_phase_2_cnn_out_phase_fix,
              cnn_equalizer.TRAIN_CE_U_SYMBOLS: ce_u_symbols_cnn_out}