import torch
import torch.optim as optim
from torch.nn import MSELoss, Softmax
from torch.nn.functional import cross_entropy
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
    rnn_eq = rnn.RNNRX(input_size, hidden_states_size, output_size, unknown_stages, device)
    rnn_eq.to(device)
    optimizer = optim.Adam(rnn_eq.parameters(), eps=1e-07, lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3)
    return rnn_eq, optimizer, scheduler

def train_rnn():
    rnn_eq.train()
    for i in range(max_num_epochs):
        idx_u, _, x, y = dd_system.simulate_transmission(1, n_sym_per_batch*batch_size, SNR_dB)
        x = hlp.norm_unit_var(x, x_mean, x_var)
        y = hlp.norm_unit_var(y, y_mean, y_var)
        rnn_inputs = data_conv_tools.gen_rnn_inputs(x, y, idx_mat_x_inputs, idx_mat_y_inputs, complex_mod)
        u_hat_soft = rnn_eq(rnn_inputs)
        
        idx_u_sic_curr_s = idx_u.reshape(batch_size, t_max, num_SIC_stages)[:,:,curr_stage-1]
        loss = cross_entropy(input=u_hat_soft.permute(0,2,1), target=idx_u_sic_curr_s)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

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
num_SIC_stages = 5
curr_stage = 3
known_stages = curr_stage-1
unknown_stages = num_SIC_stages - known_stages
### TVRNN definition
L_y = 32
L_ic = 16
eff_L_ic = L_ic*2 if complex_mod else L_ic
input_size = L_y if curr_stage == 1 else L_y + eff_L_ic
hidden_states_size = np.array([32,])
output_size = M

# NN Training parameters
lr = 0.02

n_unknown_sym_raw = 36
batch_size = 500
n_unknown_sym, n_sym_per_batch = hlp.calculate_effective_train_params(unknown_stages, num_SIC_stages, n_unknown_sym_raw)
t_max = n_unknown_sym//unknown_stages

max_num_epochs = 30000
num_frame_validation = 100
num_epochs_before_sched = 100
num_frame_sched_velidation = 1

# matrices to take the inputs 
idx_mat_y_inputs, idx_mat_x_inputs = data_conv_tools.gen_idx_mat_inputs(n_sym_per_batch*batch_size, N_os, L_y, L_ic, num_SIC_stages, curr_stage)
# reshape and use index of the SIC block acording to the paper: (s,t)
idx_mat_y_inputs = idx_mat_y_inputs.reshape(batch_size, t_max, unknown_stages, -1).permute(0,2,1,3)
idx_mat_x_inputs = idx_mat_x_inputs.reshape(batch_size, t_max, unknown_stages, -1).permute(0,2,1,3)

folder_path = io_tool.create_folder(f"results2/{train_type_name}/{mod_format}{M:}",0)
io_tool.init_summary_file(f"{folder_path}/results.txt")



for L_link in L_link_steps:
    for SNR_dB in SNR_dB_steps:
        dd_system = initialize_dd_system()
        _, _, _, y = dd_system.simulate_transmission(1, int(100e3), SNR_dB)
        y_mean, y_var, x_mean, x_var = hlp.find_normalization_constants(y, dd_system.constellation, SNR_dB)
        rnn_eq, optimizer, scheduler = initialize_RNN_optimizer(lr)
        train_rnn()
        eval_n_save_rnn()
        break
    break
