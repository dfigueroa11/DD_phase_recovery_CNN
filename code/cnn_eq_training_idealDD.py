import torch
import torch.optim as optim
from torch.nn import MSELoss 
import matplotlib.pyplot as plt
import numpy as np

import help_functions as hlp
import DD_system
import CNN_equalizer

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
            y_hat = cnn_equalizer(y)
            
            y_1_ISI = hlp.DD_1sym_ISI(x,dd_system.tx_filt[0,0,N_taps//2],dd_system.tx_filt[0,0,N_taps//2+1])*dd_system.rx_filt[0,0,0]
            loss = loss_func(y_1_ISI, y_hat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                print(f"\tBatch size {batch_size:_}\tprogress {(i+1)/batches_per_epoch:>6.1%}\t loss: {loss_evolution[-1]:.3e}",end='\r')
        print()

def eval_n_save_CNN():
    _, x, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_equalizer.eval()
    y_hat = cnn_equalizer(y)

    y_1_ISI = hlp.DD_1sym_ISI(x,dd_system.tx_filt[0,0,N_taps//2],dd_system.tx_filt[0,0,N_taps//2+1])*dd_system.rx_filt[0,0,0]
    y_1_ISI = y_1_ISI.detach().cpu().numpy()
    y_hat = y_hat.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    plt.figure()
    plt.title("Phase sample")
    plt.hist(y_1_ISI[:,:,0::2].flatten(), 20, alpha=1, label='ideal DD')
    plt.hist(y[:,:,0::2].flatten(), 200, alpha=0.5, label='DD out')
    plt.hist(y_hat[:,:,0::2].flatten(), 200, alpha=0.5, label='CNN out')
    plt.ylim(0,2e4)
    plt.legend(loc='upper right')
    lr_str = f"{lr:}".replace('.', 'p')
    plt.savefig(f"allPhase_{mod_format}{M:}_{SNR_dB}dB_lr{lr_str}_Llink{L_link*1e-3:.0f}km.png")

    plt.figure()
    plt.title("Magnitude sample")
    plt.hist(y_1_ISI[:,:,1::2].flatten(), 20, alpha=1, label='ideal DD',)
    plt.hist(y[:,:,1::2].flatten(), 200, alpha=0.5, label='DD out')
    plt.hist(y_hat[:,:,1::2].flatten(), 200, alpha=0.5, label='CNN out')
    plt.ylim(0,2e4)
    plt.legend(loc='upper right')
    plt.savefig(f"allMag_{mod_format}{M:}_{SNR_dB}dB_lr{lr_str}_Llink{L_link*1e-3:.0f}km.png")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

### System definition
N_os = 2
N_sim = 2
mod_format = "ASK"
M = 2
sqrt_flag = False
diff_encoder = False
N_taps = 41
alpha = 0
R_sym = 35e9
beta2 = -2.168e-26
L_link_steps = np.array([*range(25,35,5)])*1e3      # for sweep over L_link
SNR_dB_steps = [*range(40,42)]                          # for sweep over SNR

### CNN definition
num_ch = [1,6,8,3,1]
ker_lens = [21,15,9,9]
strides = [1,1,1,1]
activ_func = torch.nn.ELU()

### Training hyperparameter
batches_per_epoch = 30
batch_size_per_epoch = [100, 300]
N_sym = 1000
lr_steps = [0.001, 0.002]                               # for sweep over lr

checkpoint_per_epoch = 10
for lr in lr_steps:
    for SNR_dB in SNR_dB_steps:
        for L_link in L_link_steps:
            print(f'training model with lr={lr}, L_link={L_link*1e-3:.0f}km, SNR{SNR_dB} dB')
            dd_system = initialize_dd_system()
            cnn_equalizer, optimizer = initialize_CNN_optimizer(lr)
            train_CNN()
            eval_n_save_CNN()
