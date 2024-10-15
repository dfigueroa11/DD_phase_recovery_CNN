import torch
import torch.optim as optim
from torch.nn import MSELoss 

import numpy as np

import help_functions as hlp
import performance_metrics as perf_met
import in_out_tools as io_tool
from DD_system import DD_system
import CNN_equalizer
from loss_functions import loss_funcs

def initialize_dd_system():
    return hlp.set_up_DD_system(N_os=N_os, N_sim=N_sim, device=device,
                                mod_format=mod_format, M=M, sqrt_flag=True,
                                diff_encoder=True,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_CNN_optimizer(lr):
    groups_list = [1]*len(ker_lens)
    num_ch_aux = num_ch.copy()
    strides_aux = strides.copy()
    ker_lens_aux = ker_lens.copy()
    # if modulation have multiple phases and magnitudes stack two CNN in parallel for each component.
    if dd_system.multi_mag_const and dd_system.multi_phase_const:
        groups_list = [1]+[2]*(len(ker_lens)-1)
        num_ch_aux[1:] = num_ch_aux[1:]*2
    if train_type == CNN_equalizer.TRAIN_CE_U_SYMBOLS:
        groups_list.append(1)
        num_ch_aux = np.append(num_ch_aux,M)
        strides_aux = np.append(strides_aux,1)
        ker_lens_aux = np.append(ker_lens_aux, 7)
    cnn_equalizer = CNN_equalizer.CNN_equalizer(num_ch_aux, ker_lens_aux, strides_aux, activ_func, groups_list)
    cnn_equalizer.to(device)
    optimizer = optim.Adam(cnn_equalizer.parameters(), eps=1e-07, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    return cnn_equalizer, optimizer, scheduler

def train_CNN(loss_function):
    cnn_equalizer.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u_idx, u, _, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            cnn_out = cnn_equalizer(y)
            
            loss = loss_function(u_idx, u, cnn_out, dd_system)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                checkpoint_tasks(y, u.detach().cpu(), cnn_out.detach().cpu(), batch_size, (i+1)/batches_per_epoch, loss.detach().cpu().numpy())
        print()

def checkpoint_tasks(y, u, cnn_out, batch_size, progress, loss):
    u_hat = cnn_out_2_u_hat(cnn_out, dd_system)
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    scheduler.step(sum(SERs))
    curr_lr = scheduler.get_last_lr()
    MI = perf_met.get_MI(u, u_hat, dd_system, SNR_dB)
    io_tool.print_progress(batch_size, progress, curr_lr, loss, SERs, MI)
    if save_progress:
        io_tool.save_progress(progress_file_path, batch_size, progress, curr_lr, loss, SERs, MI)

def eval_n_save_CNN():
    _, u, _, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_equalizer.eval()
    cnn_out = cnn_equalizer(y).detach().cpu()

    u_hat = cnn_out_2_u_hat(cnn_out, dd_system)
    u = u.detach().cpu()
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    MI = perf_met.get_MI(u, u_hat, dd_system, SNR_dB)

    io_tool.print_save_summary(f"{folder_path}/results.txt", lr, L_link, alpha, SNR_dB, SERs, MI)

    if all([SNR_dB in SNR_save_fig, lr in lr_save_fig, L_link in L_link_save_fig, alpha in alpha_save_fig]):
        io_tool.save_fig_summary(u, y.detach().cpu(), u_hat, cnn_out, dd_system, train_type,
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
train_type = CNN_equalizer.TRAIN_TYPES[args.loss_func]

### CNN definition
num_ch = np.array([1,15,7,1])
ker_lens = np.array([11,11,7])
strides = np.array([1,1,2])
activ_func = torch.nn.ELU()
loss_func = loss_funcs[train_type]
cnn_out_2_u_hat = CNN_equalizer.cnn_out_2_u_hat_funcs[train_type]
### Training hyperparameter
batches_per_epoch = 300
batch_size_per_epoch = [100, 300, 500]
N_sym = 1000
lr_steps = np.array([0.004])       # for sweep over lr
lr_save_fig = lr_steps

checkpoint_per_epoch = 20
save_progress = True

folder_path = io_tool.create_folder(f"results/{mod_format}{M:}",0)
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
                    io_tool.init_progress_file(progress_file_path)
                train_CNN(loss_func)
                eval_n_save_CNN()
