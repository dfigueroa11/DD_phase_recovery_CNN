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


dd_system = hlp.set_up_DD_system(N_os= 2, N_sim=2,
                            mod_format="PAM", M=4, sqrt_flag=False,
                            diff_encoder=False,
                            N_taps=41,     
                            alpha=0, 
                            L_link=30e3, R_sym=35e9, beta2=-2.168e-26)

# ### to determine an appropriate number of taps for the tx_filter choose a big number of taps above
# ### and use the code below to determine an appropriate number of taps
# energy_criteria = 98
# filt, N_taps = hlp.filt_windowing(torch.squeeze(dd_system.tx_filt), energy_criteria)
# print(f"{N_taps} tap are needed to contain the {energy_criteria}% of the energy")
# plt.figure()
# plt.stem(torch.abs(filt)**2)
# plt.show()

num_ch = [1,3,3,2]
ker_lens = [21,15,9]
strides = [1,1,2]
activ_func = torch.nn.ELU()
cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch, ker_lens, strides, activ_func)

optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07)


batches_per_epoch = 300
batch_size_per_epoch = [100, 300, 500]
N_sym = 1000
SNR_dB_steps = [*range(40,41)]
loss_func = MSELoss(reduction='mean')

checkpoint_per_epoch = 10
loss_evolution = []
cnn_equalizer.train()
for SNR_dB in SNR_dB_steps:
    print(f'train model for SNR {SNR_dB} dB')
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
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:  
                print(f"\tBatch size {batch_size:_}, Train step {i:_} from {batches_per_epoch:_}, {loss_evolution[-1]}")
            


u, x, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
cnn_equalizer.eval()
x_hat = cnn_equalizer(y)

x = x.detach().cpu().numpy()
x_hat = x_hat.detach().cpu().numpy()

plt.figure()
plt.scatter(x_hat[:,0,:],x_hat[:,1,:], alpha=0.5, label="CNN_eq")
plt.scatter(np.real(x),np.imag(x), alpha=0.5, label="DD_input")
plt.legend()
plt.show()