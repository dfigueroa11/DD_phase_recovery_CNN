import torch
import torch.optim as optim
from torch.nn import MSELoss 

import numpy as np

import help_functions as hlp
import in_out_tools as io_tool
import DD_system
import CNN_equalizer

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    return cnn_equalizer, optimizer, scheduler

def train_CNN():
    loss_evolution = [-1]
    loss_func = MSELoss(reduction='mean')
    cnn_equalizer.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            _, u, _, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            y_hat = cnn_equalizer(y)[:,:,1:]
            
            y_ideal = hlp.create_ideal_y(u, dd_system.multi_mag_const, dd_system.multi_phase_const,
                                         h0_tx=dd_system.tx_filt[0,0,N_taps//2], h0_rx=torch.max(dd_system.rx_filt))[:,:,1:]
            loss = loss_func(y_ideal, y_hat)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_evolution.append(loss.detach().cpu().numpy())
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                checkpoint_tasks(y_ideal, y_hat, u, batch_size, (i+1)/batches_per_epoch, loss_evolution[-1])
        print()

def checkpoint_tasks(y_ideal, y_hat, u, batch_size, progress, loss):
    _, SERs = hlp.calc_progress(y_ideal.detach().cpu(), y_hat.detach().cpu(), dd_system.multi_mag_const, dd_system.multi_phase_const)
    scheduler.step(sum(SERs))
    curr_lr = scheduler.get_last_lr()
    u_hat = hlp.y_hat_2_u_hat(y_hat, dd_system.multi_mag_const, dd_system.multi_phase_const, h0_tx=dd_system.tx_filt[0,0,N_taps//2], h0_rx=torch.max(dd_system.rx_filt))
    u = u[:,:,1:].detach().cpu()
    MI = hlp.get_MI(u, u_hat.detach().cpu(), dd_system.constellation.detach().cpu(), SNR_dB)
    io_tool.print_progress(dd_system.multi_mag_const, dd_system.multi_phase_const, batch_size,
                            progress, curr_lr, loss, SERs, MI)
    if save_progress:
        io_tool.save_progress(progress_file_path, dd_system.multi_mag_const, dd_system.multi_phase_const,
                                batch_size, progress, curr_lr, loss, SERs, MI)

def eval_n_save_CNN():
    _, u, _, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_equalizer.eval()
    y_hat = cnn_equalizer(y)[:,:,1:]

    y_ideal = hlp.create_ideal_y(u, dd_system.multi_mag_const, dd_system.multi_phase_const,
                                 h0_tx=dd_system.tx_filt[0,0,N_taps//2], h0_rx=torch.max(dd_system.rx_filt)).detach().cpu()[:,:,1:]
    alphabets, SERs = hlp.calc_progress(y_ideal, y_hat.detach().cpu(), dd_system.multi_mag_const, dd_system.multi_phase_const)    
    
    u_hat = hlp.y_hat_2_u_hat(y_hat, dd_system.multi_mag_const, dd_system.multi_phase_const, h0_tx=dd_system.tx_filt[0,0,N_taps//2], h0_rx=torch.max(dd_system.rx_filt))
    u = u[:,:,1:].detach().cpu()
    MI = hlp.get_MI(u, u_hat.detach().cpu(), dd_system.constellation.detach().cpu(), SNR_dB)

    io_tool.print_save_summary(f"{folder_path}/SER_results.txt", dd_system.multi_mag_const, dd_system.multi_phase_const,
                               lr, L_link, alpha, SNR_dB, SERs, MI)

    if SNR_dB in SNR_save_fig and lr in lr_save_fig and L_link in L_link_save_fig and alpha in alpha_save_fig:
        io_tool.save_fig_summary(y.detach().cpu(), y_hat.detach().cpu(), dd_system.multi_mag_const, dd_system.multi_phase_const, alphabets,
                             folder_path, lr, L_link, alpha, SNR_dB)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

args = io_tool.process_args()
### System definition
N_os = 2
N_sim = 2
mod_format = args.mod_format
M = args.order
sqrt_flag = False
diff_encoder = False
N_taps = 41
R_sym = 35e9
beta2 = -2.168e-26
alpha_steps = np.arange(0,1)               # for sweep over alpha
alpha_save_fig = alpha_steps
L_link_steps = np.arange(0,35,6)*1e3      # for sweep over L_link
L_link_save_fig = L_link_steps[[0,2,-1]]
SNR_dB_steps = np.arange(-5, 12, 2)                          # for sweep over SNR
SNR_save_fig = SNR_dB_steps[[0,5,-1]]

### CNN definition
num_ch = np.array([1,15,7,1])
ker_lens = np.array([11,11,7])
strides = np.array([1,1,2])
activ_func = torch.nn.ELU()

### Training hyperparameter
batches_per_epoch = 300
batch_size_per_epoch = [100, 300, 500]
N_sym = 1000
lr_steps = np.array([0.004])       # for sweep over lr
lr_save_fig = lr_steps

checkpoint_per_epoch = 100
save_progress = True

folder_path = io_tool.create_folder(f"results/{mod_format}{M:}_sym",0)
io_tool.init_summary_file(f"{folder_path}/results.txt")

for lr in lr_steps:
    for L_link in L_link_steps:
        for alpha in alpha_steps:
            for SNR_dB in SNR_dB_steps:
                print(f'training model with lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB} dB, for {mod_format}-{M}')
                dd_system = initialize_dd_system()
                cnn_equalizer, optimizer, scheduler = initialize_CNN_optimizer(lr)
                if save_progress:
                    progress_file_path = f"{folder_path}/progress_{io_tool.make_file_name(lr, L_link, alpha, SNR_dB)}.txt"
                    io_tool.init_progress_file(progress_file_path, dd_system.multi_mag_const, dd_system.multi_phase_const)
                train_CNN()
                eval_n_save_CNN()
