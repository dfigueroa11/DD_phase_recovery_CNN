import torch
import torch.optim as optim
from torch.nn import MSELoss 

import help_functions as hlp
import DD_system
import CNN_equalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)


dd_system = DD_system.set_up_DD_system(N_os= 2, N_sim=2,
                            mod_format="PAM", M=4,
                            N_taps=11,     # for all
                            alpha=0, 
                            L_link=10e3, R_sym=20e9, beta2=-2e26)


num_ch = [1,3,3,2]
ker_lens = [11,9,9]
strides = [1,1,2]
paddings = [(ker_len-1)//2 for ker_len in ker_lens]
activ_func = torch.nn.ELU()
cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch, ker_lens, strides, paddings, activ_func)

optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07)



batches_per_epoch = 30
batch_size_per_epoch = [100, 400]
Ptx_dB_steps = torch.tensor([*range(20,21)])
N_sym = 200

cnn_equalizer.train()
loss_func = MSELoss()
for Ptx_dB in Ptx_dB_steps:
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u, x, y = dd_system.simulate_transmission(batch_size, N_sym, Ptx_dB)
            x_hat = cnn_equalizer(y)

            x_real_imag = torch.cat((torch.real(x),torch.imag(x)), dim=1)
            loss = loss_func(x_real_imag,x_hat)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
