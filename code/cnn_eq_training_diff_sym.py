import torch
import torch.optim as optim
from torch.nn import MSELoss 
import matplotlib.pyplot as plt

import numpy as np
import os
import sys

import help_functions as hlp
import DD_system
import CNN_equalizer

def create_results_folder(path,n_copy):
    try:
        real_path = f"{path}_{n_copy}" if n_copy > 0 else path
        os.makedirs(real_path)
        return real_path
    except Exception as e:
        n_copy += 1
        print(f"directory '{path}' already exist, try to create '{path}_{n_copy}'")
        return create_results_folder(path,n_copy)

def initialize_dd_system():
    return hlp.set_up_DD_system(N_os=N_os, N_sim=N_sim, device=device,
                                mod_format=mod_format, M=M, sqrt_flag=True,
                                diff_encoder=True,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_CNN_optimizer(lr):
    groups_list = None
    num_ch_aux = num_ch.copy()
    # if modulation have multiple phases and magnitudes stack two CNN in parallel for each component.
    if dd_system.multi_mag_const and dd_system.multi_phase_const:
        groups_list = [1]+[2]*(len(ker_lens)-1)
        num_ch_aux[1:] = num_ch_aux[1:]*2
    cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch_aux, ker_lens, strides, activ_func, groups_list)
    cnn_equalizer.to(device)
    optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07, lr=lr)
    return cnn_equalizer, optimizer

def train_CNN():
    loss_evolution = [-1]
    loss_func = MSELoss(reduction='mean')
    cnn_equalizer.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            y_hat = cnn_equalizer(y)
            
            y_ideal = hlp.create_ideal_y(u, dd_system.multi_mag_const, dd_system.multi_phase_const, h0=dd_system.tx_filt[0,0,N_taps//2], h_rx=torch.max(dd_system.rx_filt))
            loss = loss_func(y_ideal[:,:,1:], y_hat[:,:,1:])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                hlp.print_progress(y_ideal[:,:,1:].detach().cpu(), y_hat[:,:,1:].detach().cpu(), batch_size, (i+1)/batches_per_epoch, loss_evolution[-1], dd_system.multi_mag_const, dd_system.multi_phase_const)
        print()

def eval_n_save_CNN():
    u, x, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_equalizer.eval()
    y_hat = cnn_equalizer(y).detach().cpu()

    y_ideal = hlp.create_ideal_y(u, dd_system.multi_mag_const, dd_system.multi_phase_const, h0=dd_system.tx_filt[0,0,N_taps//2], h_rx=torch.max(dd_system.rx_filt)).detach().cpu()
    
    alphabets,_ = hlp.print_save_summary(y_ideal[:,:,1:], y_hat[:,:,1:],
                           dd_system.multi_mag_const, dd_system.multi_phase_const,
                           lr, L_link, alpha, SNR_dB, f"{folder_path}/SER_results.txt")

    if SNR_dB in SNR_save_fig and lr in lr_save_fig and L_link in L_link_save_fig and alpha in alpha_save_fig:
        hlp.save_fig_summary(y.detach().cpu(), y_hat, dd_system.multi_mag_const, dd_system.multi_phase_const, alphabets, folder_path, lr, L_link, alpha, SNR_dB,)
        

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

arguments = sys.argv[1:]
### System definition
N_os = 2
N_sim = 2
mod_format = "ASK"#arguments[0]
M = 8#int(arguments[1])
sqrt_flag = False
diff_encoder = False
N_taps = 41
R_sym = 35e9
beta2 = -2.168e-26
alpha_steps = np.arange(0,1)               # for sweep over alpha
alpha_save_fig = alpha_steps
L_link_steps = np.arange(0,35,6)*1e3      # for sweep over L_link
L_link_save_fig = L_link_steps
SNR_dB_steps = np.arange(5,26,5)                          # for sweep over SNR
SNR_save_fig = SNR_dB_steps#[[2,-1]]

### CNN definition
num_ch = np.array([1,15,7,1])
ker_lens = np.array([11,11,7])
strides = np.array([1,1,2])
activ_func = torch.nn.ELU()

### Training hyperparameter
batches_per_epoch = 30
batch_size_per_epoch = [10]
N_sym = 1000
lr_steps = np.array([0.0007, 0.001, 0.003, 0.005])       # for sweep over lr
lr_save_fig = lr_steps#[[2,]]

checkpoint_per_epoch = 1

folder_path = create_results_folder(f"results/{mod_format}{M:}_odd_samp",0)
for lr in lr_steps:
    for L_link in L_link_steps:
        for alpha in alpha_steps:
            for SNR_dB in SNR_dB_steps:
                print(f'training model with lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB} dB')
                dd_system = initialize_dd_system()
                cnn_equalizer, optimizer = initialize_CNN_optimizer(lr)
                train_CNN()
                eval_n_save_CNN()
