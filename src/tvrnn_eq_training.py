import torch
import torch.optim as optim
from torch.nn import MSELoss, Softmax
from torch.nn.functional import cross_entropy, softmax
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
        idx_u, u, x, y = dd_system.simulate_transmission(1, n_sym_per_batch*batch_size, SNR_dB)
        x = hlp.norm_unit_var(x, x_mean, x_var)
        y = hlp.norm_unit_var(y, y_mean, y_var)
        rnn_inputs = data_conv_tools.gen_rnn_inputs(x, y, idx_mat_x_inputs, idx_mat_y_inputs, complex_mod)
        u_hat_soft = rnn_eq(rnn_inputs).permute(0,2,1)
        
        idx_u_sic_curr_s = idx_u.reshape(batch_size, t_max, num_SIC_stages)[:,:,curr_stage-1]
        ce = cross_entropy(input=u_hat_soft, target=idx_u_sic_curr_s)
        ce.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i+1) % num_epochs_before_sched == 0:
            checkpoint_tasks(u_hat_soft.detach().cpu(), u.cpu(), ce.detach().cpu(), (i+1)/max_num_epochs)
            if optimizer.param_groups[0]["lr"] < 1e-5: break

def checkpoint_tasks(u_hat_soft, u, ce, progress):
    u_hat = data_conv_tools.APPs_2_u(u_hat_soft, dd_system, SNR_dB).flatten()
    u = u.reshape(batch_size, t_max, num_SIC_stages)[:,:,curr_stage-1].flatten()
    SERs = perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
    MI = np.log2(M) - ce*np.log2(np.exp(1))
    scheduler.step(MI)
    curr_lr = scheduler.get_last_lr()
    io_tool.print_progress(batch_size, progress, curr_lr, ce, SERs, MI)
    if save_progress:
        io_tool.save_progress(progress_file_path, batch_size, progress, curr_lr, ce, SERs, MI)

def eval_n_save_rnn():
    rnn_eq.eval()
    SERs = torch.zeros(3)
    MI = 0
    for i in range(num_frame_validation):
        idx_u, u, x, y = dd_system.simulate_transmission(1, n_sym_per_batch*batch_size, SNR_dB)
        x = hlp.norm_unit_var(x, x_mean, x_var)
        y = hlp.norm_unit_var(y, y_mean, y_var)
        rnn_inputs = data_conv_tools.gen_rnn_inputs(x, y, idx_mat_x_inputs, idx_mat_y_inputs, complex_mod)
        u_hat_soft = rnn_eq(rnn_inputs).permute(0,2,1).detach().cpu()
        
        idx_u_sic_curr_s = idx_u.reshape(batch_size, t_max, num_SIC_stages)[:,:,curr_stage-1].cpu()
        ce = cross_entropy(input=u_hat_soft, target=idx_u_sic_curr_s)

        u_hat = data_conv_tools.APPs_2_u(u_hat_soft, dd_system, SNR_dB).flatten()
        u = u.reshape(batch_size, t_max, num_SIC_stages)[:,:,curr_stage-1].cpu().flatten()
        SERs += perf_met.get_all_SERs(u, u_hat, dd_system, SNR_dB)
        MI += np.log2(M) - ce*np.log2(np.exp(1))

    MI = MI/num_frame_validation
    SERs = SERs/num_frame_validation

    io_tool.print_save_summary(f"{folder_path}/results_S={num_SIC_stages}_s={curr_stage}_L_y={L_y}_hs={hidden_states_size}.txt", lr, L_link, alpha, SNR_dB, SERs, MI)


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
L_link_steps = np.array([0,12,30])*1e3      # for sweep over L_link
L_link_save_fig = L_link_steps
SNR_dB_steps = np.arange(-5, 12, 2)                          # for sweep over SNR
SNR_save_fig = SNR_dB_steps[[0,5,-2,-1]]
train_type = list(rnn.TRAIN_TYPES.keys())[args.loss_func]
train_type_name = rnn.TRAIN_TYPES[train_type]

### SIC definition 
num_SIC_stages = args.numSIC
curr_stage = args.currentStage
known_stages = curr_stage-1
unknown_stages = num_SIC_stages - known_stages
### TVRNN definition
L_y_list = [10, 20, 30]
L_ic = 16
eff_L_ic = L_ic*2 if complex_mod else L_ic
input_size_list = [L_y if curr_stage == 1 else L_y + eff_L_ic for L_y in L_y_list] 
hidden_states_size_list = [np.array([10,]), np.array([20,]), np.array([30,])]
output_size = M

## NN Training parameters
lr = 0.02
n_unknown_sym_raw = 36
batch_size = 500
n_unknown_sym, n_sym_per_batch = hlp.calculate_effective_train_params(unknown_stages, num_SIC_stages, n_unknown_sym_raw)
t_max = n_unknown_sym//unknown_stages
max_num_epochs = 30_000
num_frame_validation = 100
num_epochs_before_sched = 100
num_frame_sched_validation = 1


folder_path = io_tool.create_folder(f"results2/{train_type_name}/{mod_format}{M:}",0)

for input_size, L_y in zip(input_size_list, L_y_list):
    for hidden_states_size in hidden_states_size_list:
        ## matrices to take the inputs 
        idx_mat_y_inputs, idx_mat_x_inputs = data_conv_tools.gen_idx_mat_inputs(n_sym_per_batch*batch_size, N_os, L_y, L_ic, num_SIC_stages, curr_stage)
        # reshape and use index of the SIC block acording to the paper: (s,t)
        idx_mat_y_inputs = idx_mat_y_inputs.reshape(batch_size, t_max, unknown_stages, -1).permute(0,2,1,3).to(device)
        idx_mat_x_inputs = idx_mat_x_inputs.reshape(batch_size, t_max, unknown_stages, -1).permute(0,2,1,3).to(device)

        ## saving routine
        save_progress = True

        io_tool.init_summary_file(f"{folder_path}/results_S={num_SIC_stages}_s={curr_stage}_L_y={L_y}_hs={hidden_states_size}.txt")
        for L_link in L_link_steps:
            for SNR_dB in SNR_dB_steps:
                print(f'training model with L_link={L_link*1e-3:.0f}km, SNR={SNR_dB} dB, for {mod_format}-{M} L_y={L_y}, hs={hidden_states_size}')
                dd_system = initialize_dd_system()
                _, _, _, y = dd_system.simulate_transmission(1, int(100e3), SNR_dB)
                y_mean, y_var, x_mean, x_var = hlp.find_normalization_constants(y, dd_system.constellation, SNR_dB)
                rnn_eq, optimizer, scheduler = initialize_RNN_optimizer(lr)
                if save_progress:
                    progress_file_path = f"{folder_path}/progress_S={num_SIC_stages}_s={curr_stage}_L_y={L_y}_hs={hidden_states_size}_{io_tool.make_file_name(lr, L_link, alpha, SNR_dB)}.txt"
                    io_tool.init_progress_file(progress_file_path)
                train_rnn()
                eval_n_save_rnn()
        io_tool.write_complexity_in_summary_file(f"{folder_path}/results_S={num_SIC_stages}_s={curr_stage}_L_y={L_y}_hs={hidden_states_size}.txt", rnn_eq.complexity)