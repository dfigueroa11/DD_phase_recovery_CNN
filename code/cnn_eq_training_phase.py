import torch
import torch.optim as optim
from torch.nn import MSELoss 
import matplotlib.pyplot as plt
import numpy as np

import help_functions as hlp
import DD_system
import CNN_equalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

N_taps = 41
mod_format = "QAM"
M = 4
dd_system = hlp.set_up_DD_system(N_os= 2, N_sim=2,
                            mod_format=mod_format, M=M, sqrt_flag=False,
                            diff_encoder=False,
                            N_taps=N_taps,     
                            alpha=0, 
                            L_link=30e3, R_sym=35e9, beta2=-2.168e-26)

### to determine an appropriate number of taps for the tx_filter choose a big number of taps above
### and use the code below to determine an appropriate number of taps
# energy_criteria = 99
# filt, N_taps = hlp.filt_windowing(torch.squeeze(dd_system.tx_filt), energy_criteria)
# print(f"{N_taps} tap are needed to contain the {energy_criteria}% of the energy")
# plt.figure()
# t = np.arange(-np.floor(torch.numel(filt)/2),np.floor(torch.numel(filt)/2)+1)
# plt.stem(t, np.abs(filt)**2)
# t = np.arange(-np.floor(torch.numel(dd_system.tx_filt)/2),np.floor(torch.numel(dd_system.tx_filt)/2)+1)
# plt.stem(t, np.abs(torch.squeeze(dd_system.tx_filt))**2, linefmt=':')
# plt.show()

num_ch = [1,6,8,3,1]
ker_lens = [21,15,9,9]
strides = [1,1,1,2]
activ_func = torch.nn.ELU()
cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch, ker_lens, strides, activ_func)

optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07)


batches_per_epoch = 300
batch_size_per_epoch = [100, 300]
N_sym = 1000
SNR_dB_steps = [*range(40,41)]
loss_func = MSELoss(reduction='mean')

checkpoint_per_epoch = 10
loss_evolution = [-1]
cnn_equalizer.train()
for SNR_dB in SNR_dB_steps:
    print(f'train model for SNR {SNR_dB} dB')
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            y_hat = cnn_equalizer(y)
            
            y_1_ISI = hlp.DD_1sym_ISI(x,dd_system.tx_filt[0,0,N_taps//2],dd_system.tx_filt[0,0,N_taps//2+1])*dd_system.rx_filt[0,0,0]
            loss = loss_func(y_1_ISI[:,:,0::2], y_hat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:  
                print(f"\tBatch size {batch_size:_}, Train step {i:_} from {batches_per_epoch:_}, {loss_evolution[-1]}")
            


u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
cnn_equalizer.eval()
y_hat = cnn_equalizer(y)

y_1_ISI = hlp.DD_1sym_ISI(x,dd_system.tx_filt[0,0,N_taps//2],dd_system.tx_filt[0,0,N_taps//2+1])*dd_system.rx_filt[0,0,0]
y_1_ISI = y_1_ISI.detach().cpu().numpy()
y_hat = y_hat.detach().cpu().numpy()




plt.figure()
plt.title("Phase sample")
plt.hist(y_1_ISI[:,:,0::2].flatten(), 20, alpha=1, label='ideal DD')
plt.hist(y[:,:,0::2].flatten(), 200, alpha=0.5, label='DD out')
plt.hist(y_hat.flatten(), 200, alpha=0.5, label='CNN out')
plt.ylim(0,2e4)
plt.legend(loc='upper right')
plt.savefig(f"{mod_format}{M:}_phase_bigCNN.png")
# plt.show()