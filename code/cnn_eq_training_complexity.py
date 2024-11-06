import torch
import torch.optim as optim
from torch.nn import MSELoss, Softmax
import matplotlib.pyplot as plt
from time import sleep
import numpy as np

import help_functions as hlp
import performance_metrics as perf_met
import in_out_tools as io_tool
from DD_system import DD_system
import cnn_equalizer
from loss_functions import loss_funcs
from complexity_tools import design_CNN_structures_fix_comp

def initialize_dd_system():
    return hlp.set_up_DD_system(N_os=N_os, N_sim=N_sim, device=device,
                                mod_format=mod_format, M=M, sqrt_flag=True,
                                diff_encoder=True,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_CNN_optimizer(lr, num_ch, ker_sz, strides, groups_list):
    activ_func_last_layer = None
    # if modulation have multiple phases and magnitudes stack two CNN in parallel for each component.
    if train_type == cnn_equalizer.TRAIN_CE_U_SYMBOLS:
        activ_func_last_layer = Softmax(dim=1)
    cnn_eq = cnn_equalizer.CNN_equalizer(num_ch, ker_sz, strides, activ_func, groups_list, activ_func_last_layer)
    cnn_eq.to(device)
    optimizer = optim.Adam(cnn_eq.parameters(), eps=1e-07, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5)
    return cnn_eq, optimizer, scheduler

def train_CNN(loss_function):
    cnn_eq.train()
    for batch_size in batch_size_per_epoch:
        for i in range(batches_per_epoch):
            u_idx, u, _, y = dd_system.simulate_transmission(batch_size, N_sym, SNR_dB)
            cnn_out = cnn_eq(y)
            
            loss = loss_function(u_idx, u, cnn_out, dd_system)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if (i+1)%(batches_per_epoch//checkpoint_per_epoch) == 0:
                checkpoint_tasks(y, u.detach().cpu(), cnn_out.detach().cpu(), batch_size, (i+1)/batches_per_epoch, loss.detach().cpu().numpy())
        print()

def checkpoint_tasks(y, u, cnn_out, batch_size, progress, loss):
    u_hat = cnn_out_2_u_hat(cnn_out, dd_system, Ptx_dB=SNR_dB)
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    scheduler.step(sum(SERs))
    curr_lr = scheduler.get_last_lr()
    MI = perf_met.get_MI_HD(u, u_hat, dd_system, SNR_dB)
    io_tool.print_progress(batch_size, progress, curr_lr, loss, SERs, MI)

def eval_n_save_CNN():
    _, u, _, y = dd_system.simulate_transmission(100, N_sym, SNR_dB)
    cnn_eq.eval()
    cnn_out = cnn_eq(y).detach().cpu()
    u_hat = cnn_out_2_u_hat(cnn_out, dd_system, Ptx_dB=SNR_dB)
    u = u.detach().cpu()
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    MI = perf_met.get_MI_HD(u, u_hat, dd_system, SNR_dB)
    with open(f"{folder_path}/results_L={n_layers}_C={complexity}.txt", 'a') as file:
        file.write(f"{L_link*1e-3:.0f},{SERs[0]:.10e},{SERs[1]:.10e},{SERs[2]:.10e},{MI:.10e},{cnn_eq.complexity:.0f}\n")


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
alpha = 0
L_link_steps = np.array([0,12,30])*1e3      # for sweep over L_link
SNR_dB = 9
train_type = list(cnn_equalizer.TRAIN_TYPES.keys())[args.loss_func]
train_type_name = cnn_equalizer.TRAIN_TYPES[train_type]
### Training hyperparameter
batches_per_epoch = 300
batch_size_per_epoch = [100, 300]
N_sym = 1000
lr = 0.004
checkpoint_per_epoch = 100

### CNN definition
activ_func = torch.nn.ELU()
CNN_ch_in = 1
CNN_ch_out = 1
###################################### change
n_layers = 2
complexity = 1000
complexity_profiles = np.array([[1,1],[1,2],[2,1],[1,3],[3,1]])   # for 2 layers
# complexity_profiles = np.array([[1,1,1],[1,2,3],[3,2,1],[1,2,1],[2,1,2]])   # for 3 layers
# complexity_profiles = np.array([[1,1,1,1],[1,2,3,3],[3,3,2,1],[1,2,2,1],[2,1,1,2]])   # for 4 layers
n_str_layer = 4
##########################################
loss_func = loss_funcs[train_type]
cnn_out_2_u_hat = cnn_equalizer.cnn_out_2_u_hat_funcs[train_type]
## Design structures
L_link = L_link_steps[0]
dd_system = initialize_dd_system()
groups = [1]*n_layers
# if modulation have multiple phases and magnitudes stack two CNN in parallel for each component.
if dd_system.multi_mag_const and dd_system.multi_phase_const:
    groups = [1]+[2]*(n_layers-1)
    CNN_ch_out = 2
if train_type == cnn_equalizer.TRAIN_CE_U_SYMBOLS:
    groups[-1] = 1
    CNN_ch_out = M

sys_to_simulate = len(complexity_profiles)*n_layers*n_str_layer**(n_layers -1)*len(L_link_steps)
print(f"you ara about to simulate {sys_to_simulate} times the system")
sleep(2)


structures = []
for complexity_profile in complexity_profiles:
    complexity_profile = complexity_profile/complexity_profile.sum()
    for i in range(n_layers):
        strides = np.eye(n_layers)[i]+1
        structures.extend(design_CNN_structures_fix_comp(complexity, complexity_profile, CNN_ch_in, CNN_ch_out, strides, np.array(groups), n_str_layer))

folder_path = io_tool.create_folder(f"results/{train_type_name}/{mod_format}{M:}",0)
np.save(f"{folder_path}/structures_L={n_layers}_C={complexity}.npy",np.array(structures))
with open(f"{folder_path}/results_L={n_layers}_C={complexity}.txt", 'a') as file:
        file.write("L_link_km,mag_ER,phase_ER,SER,MI,complexity\n")

for L_link in L_link_steps:
    CNN_complexities = []
    for i, structure in enumerate(structures):
        dd_system = initialize_dd_system()
        cnn_eq, optimizer, scheduler = initialize_CNN_optimizer(lr, np.append(structure[0],structure[1,-1]), structure[2], structure[3], structure[4])
        print(f'training model L_link={L_link*1e-3:.0f}km, SNR={SNR_dB} dB, for {mod_format}-{M}, train type: {train_type_name}, complexity: {cnn_eq.complexity:.0f}, structure: {i+1}/{len(structures)}')
        train_CNN(loss_func)
        eval_n_save_CNN()
        CNN_complexities.append(cnn_eq.complexity)

CNN_complexities = np.array(CNN_complexities)
io_tool.write_complexities_summary(f"{folder_path}/results_L={n_layers}_C={complexity}.txt", CNN_complexities)
print(f"min complexity: \t{CNN_complexities.min():.0f} at {CNN_complexities.argmin()}")
print(f"max complexity: \t{CNN_complexities.max():.0f} at {CNN_complexities.argmax()}")
print(f"mean complexity:\t{CNN_complexities.mean():.2f}")
print(f"std complexity: \t{CNN_complexities.std():.2f}")
# plt.hist(CNN_complexities, 500, cumulative=True, density=True)
# plt.show()