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
                                mod_format=mod_format, M=M, sqrt_flag=False,
                                diff_encoder=False,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_CNN_optimizer(lr):
    cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch, ker_lens, strides, activ_func)
    cnn_equalizer.to(device)
    optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07, lr=lr)
    return cnn_equalizer, optimizer

def train_CNN():
    loss_evolution = [-1]
    loss_func = MSELoss(reduction='mean')
    cnn_equalizer.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            _, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            y_hat = cnn_equalizer(torch.cat((y,torch.kron(torch.abs(x),torch.eye(N_sim, device=device)[-1])), dim=1))
            
            phase_diff = hlp.abs_phase_diff(x)
            loss = loss_func(phase_diff, y_hat[:,:,1:])
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                _, SER = hlp.decode_and_ER(phase_diff.detach().cpu(),y_hat[:,:,1:].detach().cpu())
                print(f"\tBatch size {batch_size:_}\tprogress {(i+1)/batches_per_epoch:>6.1%}\t loss: {loss_evolution[-1]:.3e}\t SER: {SER:.3e}",end='\r')
        print()

def eval_n_save_CNN():
    _, x, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_equalizer.eval()
    y_hat = cnn_equalizer(torch.cat((y,torch.kron(torch.abs(x),torch.eye(N_sim, device=device)[-1])), dim=1))

    phase_diff = hlp.abs_phase_diff(x)
    alphabet, SER = hlp.decode_and_ER(phase_diff,y_hat[:,:,1:])
    print(f"\tfinal SER: {SER:.3e}")

    with open(f"{folder_path}/SER_results.txt", 'a') as file:
        file.write(f"lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB}dB --> SER:{SER:.10e}\n")
    
    if SNR_dB in SNR_save_fig and lr in lr_save_fig and L_link in L_link_save_fig and alpha in alpha_save_fig:
        y_hat = y_hat.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        plt.figure()
        plt.title("Phase difference")
        for val in alphabet.detach().cpu().numpy():
            line = plt.axvline(x=val, color='red', linestyle='--')
        _, _, hist1 = plt.hist(y[:,:,0::2].flatten(), 200, alpha=0.5, density=True)
        _, _, hist2 = plt.hist(y_hat.flatten(), 200, alpha=0.5, density=True)
        plt.legend([line, hist1[0], hist2[0]],['ideal phase diff', 'DD out', 'CNN out'], loc='upper right')
        lr_str = f"{lr:}".replace('.', 'p')
        alpha_str = f"{alpha:.1f}".replace('.', 'p')
        plt.savefig(f"{folder_path}/lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB.png")
        plt.close()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

arguments = sys.argv[1:]
### System definition
N_os = 2
N_sim = 2
mod_format = arguments[0]
M = int(arguments[1])
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
SNR_save_fig = SNR_dB_steps[[2,-1]]

### CNN definition
num_ch = [2,6,8,3,1]
ker_lens = [21,15,9,9]
strides = [1,1,1,2]
activ_func = torch.nn.ELU()

### Training hyperparameter
batches_per_epoch = 300
batch_size_per_epoch = [100, 300, 500]
N_sym = 1000
lr_steps = np.array([0.0007, 0.001, 0.003, 0.005])       # for sweep over lr
lr_save_fig = lr_steps[[4,]]

checkpoint_per_epoch = 20

folder_path = create_results_folder(f"results/{mod_format}{M:}_phase_diff_mag_in",0)
for lr in lr_steps:
    for L_link in L_link_steps:
        for alpha in alpha_steps:
            for SNR_dB in SNR_dB_steps:
                print(f'training model with lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB} dB')
                dd_system = initialize_dd_system()
                cnn_equalizer, optimizer = initialize_CNN_optimizer(lr)
                train_CNN()
                eval_n_save_CNN()
