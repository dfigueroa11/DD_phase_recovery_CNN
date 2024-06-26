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


dd_system = DD_system.set_up_DD_system(N_os= 2, N_sim=2,
                            mod_format="PAM", M=4,
                            N_taps=11,     
                            alpha=0, 
                            L_link=0e3, R_sym=20e9, beta2=-2e26)


num_ch = [1,3,3,2]
ker_lens = [11,9,9]
strides = [1,1,2]
activ_func = torch.nn.ELU()
cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch, ker_lens, strides, activ_func)

optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07)


batches_per_epoch = 1
batch_size_per_epoch = [1,]
N_sym = 2000
SNR_dB_steps = [*range(20,21)]
loss_func = MSELoss(reduction='mean')


loss_evolution = []
cnn_equalizer.train()
for SNR_dB in SNR_dB_steps:
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            x_hat = cnn_equalizer(y)
            
            x_real_imag = torch.cat((torch.real(x),torch.imag(x)), dim=1)
            loss = loss_func(x_real_imag,x_hat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())



u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
cnn_equalizer.eval()
x_hat = cnn_equalizer(y)

x = x.detach().cpu().numpy()
x_hat = x_hat.detach().cpu().numpy()

plt.figure()
plt.scatter(np.real(x)**2,np.imag(x), alpha=0.5)
plt.scatter(y[:,:,1::2],np.zeros_like(y[:,:,1::2]), alpha=0.5)
plt.scatter(x_hat[:,0,:],x_hat[:,1,:], alpha=0.5)
plt.show()