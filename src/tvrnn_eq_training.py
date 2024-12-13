import torch
import torch.optim as optim
from torch.nn import MSELoss, Softmax
import numpy as np

import comm_sys.DD_system as DD_system
from nn_equalizers import rnn
import utils.help_functions as hlp
import utils.performance_metrics as perf_met
import utils.data_conversion_tools as data_conv_tools
import utils.in_out_tools as io_tool

def initialize_dd_system():
    return hlp.set_up_DD_system(N_os=N_os, N_sim=N_sim, device=device,
                                mod_format=mod_format, M=M, sqrt_flag=sqrt_flag,
                                diff_encoder=diff_encoder,
                                N_taps=N_taps,
                                alpha=alpha,
                                L_link=L_link, R_sym=R_sym, beta2=beta2)

def initialize_RNN_optimizer(lr):
    rnn_eq = rnn.RNNRX(input_size, hidden_states_size, output_size, N_tv_cells, device)
    rnn_eq.to(device)
    optimizer = optim.Adam(rnn_eq.parameters(), eps=1e-07, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3)
    return rnn_eq, optimizer, scheduler

def train_rnn():
    pass

def eval_n_save_rnn():
    pass











device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("We are using the following device for learning:",device)

args = io_tool.process_args(rnn.TRAIN_TYPES)
### System definition
N_os = 2
N_sim = 2
mod_format = args.mod_format
complex_mod = mod_format != "PAM" and mod_format != "ASK"
M = args.order
sqrt_flag = False
diff_encoder = True
N_taps = 41
R_sym = 35e9
beta2 = -2.168e-26
alpha = 0
L_link_steps = np.arange(0,35,6)*1e3      # for sweep over L_link
L_link_save_fig = L_link_steps[[0,2,-1]]
SNR_dB_steps = np.arange(-5, 12, 2)                          # for sweep over SNR
SNR_save_fig = SNR_dB_steps[[0,5,-2,-1]]
train_type = list(rnn.TRAIN_TYPES.keys())[args.loss_func]
train_type_name = rnn.TRAIN_TYPES[train_type]

### SIC definition 
num_SIC_stages = 4
sim_stage = 1

### TVRNN definition
L_y = 32
L_ic = 16
eff_L_ic = L_ic*2 if complex_mod else L_ic
input_size = L_y if sim_stage == 1 else L_y + eff_L_ic
hidden_states_size = np.array([32,])
output_size = M
N_tv_cells = 1 if sim_stage == 1 else num_SIC_stages - sim_stage + 1

# NN Training parameters
lr = 0.02
T_rnn_raw = 36
batch_size = 512
max_num_epochs_training = 30000
num_frame_validation = 100
num_epochs_before_sched = 100
num_frame_sched_velidation = 1

folder_path = io_tool.create_folder(f"results2/{train_type_name}/{mod_format}{M:}",0)
io_tool.init_summary_file(f"{folder_path}/results.txt")

idx_mat_y_inputs, idx_mat_x_inputs = data_conv_tools.gen_idx_mat_inputs()

for L_link in L_link_steps:
    for SNR_dB in SNR_dB_steps:
        dd_system = initialize_dd_system()
        y_mean, y_var, x_mean, x_var = hlp.find_normalization_constants(dd_system, SNR_dB)
        rnn_eq, optimizer, scheduler = initialize_RNN_optimizer(lr)
        train_rnn()
        eval_n_save_rnn()
