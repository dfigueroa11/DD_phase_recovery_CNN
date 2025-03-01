import torch
import torch.optim as optim
from torch.nn import MSELoss, Softmax

import numpy as np

import comm_sys.DD_system as DD_system
from nn_equalizers import fcn_ph_equalizer
import utils.help_functions as hlp
import utils.performance_metrics as perf_met
import utils.in_out_tools as io_tool
from utils.loss_functions import loss_funcs_fcn
from utils.data_conversion_tools import reshape_data_for_FCN

def initialize_dd_system():
    return hlp.set_up_DD_system(N_os=N_os, N_sim=N_sim, device=device,
                                mod_format=mod_format, M=M, sqrt_flag=True,
                                diff_encoder=True,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_FCN_optimizer(lr):
    m = dd_system.phase_list.numel() if train_type == fcn_ph_equalizer.TRAIN_CE else 1
    activ_func_last_layer = Softmax(dim=1) if train_type == fcn_ph_equalizer.TRAIN_CE else None
    fcn_eq = fcn_ph_equalizer.FCN_ph(y_len, a_len, fcn_out_sym, m, hidden_layers_len, layer_is_bilinear, activ_func, activ_func_last_layer)
    fcn_eq.to(device)
    optimizer = optim.Adam(fcn_eq.parameters(), eps=1e-07, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    return fcn_eq, optimizer, scheduler

def train_CNN(loss_function):
    fcn_eq.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            # use some parallelization for faster execution
            total_sym = batch_size*a_len
            N_sym = 1000
            # later we remove the first and last a_len symbols to avoid ringing artifacts
            _, u, x, y = reshape_data_for_FCN(*dd_system.simulate_transmission(total_sym//N_sym, N_sym+2*a_len, SNR_dB), a_len)
            fcn_out = fcn_eq(y,torch.abs(x))
            
            loss = loss_function(u, fcn_out, dd_system)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0 or True:
                checkpoint_tasks(y, u.detach().cpu(), fcn_out.detach().cpu(), batch_size, (i+1)/batches_per_epoch, loss.detach().cpu().numpy())
        print()

def checkpoint_tasks(y: torch.Tensor, u: torch.Tensor, fcn_out: torch.Tensor, batch_size, progress, loss):
    u = u[:,u.shape[-1]//2]
    u_hat = fcn_out_2_u_hat(fcn_out, torch.abs(u), dd_system)
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    if use_scheduler: scheduler.step(sum(SERs))
    curr_lr = scheduler.get_last_lr() if use_scheduler else [lr]
    MI = perf_met.get_MI_HD(u, u_hat, dd_system, SNR_dB)
    io_tool.print_progress(batch_size, progress, curr_lr, loss, SERs, MI)
    if save_progress:
        io_tool.save_progress(progress_file_path, batch_size, progress, curr_lr, loss, SERs, MI)

def eval_n_save_CNN():
    total_sym = 100_000//a_len*a_len
    N_sym = 1000
    # later we remove the first and last a_len symbols to avoid ringing artifacts
    _, u, x, y = reshape_data_for_FCN(*dd_system.simulate_transmission(total_sym//N_sym, N_sym+2*a_len, SNR_dB), a_len)
    fcn_out = fcn_eq(y,torch.abs(x)).detach().cpu()
    
    u = u[:,u.shape[-1]//2].detach().cpu()
    u_hat = fcn_out_2_u_hat(fcn_out, torch.abs(u), dd_system)
    u = u.detach().cpu()
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    MI = perf_met.get_MI_HD(u, u_hat, dd_system, SNR_dB)

    io_tool.print_save_summary(f"{folder_path}/results.txt", lr, L_link, alpha, SNR_dB, SERs, MI)

    if all([SNR_dB in SNR_save_fig, L_link in L_link_save_fig]):
        io_tool.save_fig_summary(u, y.unsqueeze(1).detach().cpu(), u_hat, fcn_out, dd_system, train_type,
                                 folder_path, lr, L_link, alpha, SNR_dB)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

args = io_tool.process_args(fcn_ph_equalizer.TRAIN_TYPES)
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
alpha = 0
L_link_steps = np.arange(0,35,6)*1e3      # for sweep over L_link
L_link_save_fig = L_link_steps[[0,2,-1]]
SNR_dB_steps = np.arange(-5, 12, 2)                          # for sweep over SNR
SNR_save_fig = SNR_dB_steps[[0,5,-2,-1]]
train_type = list(fcn_ph_equalizer.TRAIN_TYPES.keys())[args.loss_func]
train_type_name = fcn_ph_equalizer.TRAIN_TYPES[train_type]

### FCN definition
y_len = 30*N_os
a_len = 30
fcn_out_sym = 1
hidden_layers_len = [20,10]
layer_is_bilinear = [False]*3
activ_func = torch.nn.ELU()
loss_func = loss_funcs_fcn[train_type]
fcn_out_2_u_hat = fcn_ph_equalizer.fcn_out_2_u_hat_funcs[train_type]

### Training hyperparameter
batches_per_epoch = 300
batch_size_per_epoch = [50_000, 100_000, 200_000]
lr = 0.004
use_scheduler = False
checkpoint_per_epoch = 100
save_progress = True

folder_path = io_tool.create_folder(f"results2/{train_type_name}/{mod_format}{M:}",0)
io_tool.init_summary_file(f"{folder_path}/results.txt")

for L_link in L_link_steps:
    for SNR_dB in SNR_dB_steps:
        print(f'training model with lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB} dB, for {mod_format}-{M}, train type: {train_type_name}')
        dd_system = initialize_dd_system()
        fcn_eq, optimizer, scheduler = initialize_FCN_optimizer(lr)
        if save_progress:
            progress_file_path = f"{folder_path}/progress_{io_tool.make_file_name(lr, L_link, alpha, SNR_dB)}.txt"
            io_tool.init_progress_file(progress_file_path)
        train_CNN(loss_func)
        eval_n_save_CNN()
io_tool.write_complexity_in_summary_file(f"{folder_path}/results.txt", fcn_eq.complexity)
