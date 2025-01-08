import numpy as np

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def write_rnn_results(file_name, data: np.ndarray):
    SIC_reults = data.mean(axis=0)
    results_data = np.empty((np.shape(SIC_reults)[0],6))
    results_data[:,0] = SIC_reults[:,1]
    results_data[:,1] = SIC_reults[:,3]
    results_data[:,2] = SIC_reults[:,-1]
    results_data[:,3] = data[0,:,-1]    # SDD results
    results_data[:,4] = SIC_reults[:,-2]
    results_data[:,5] = data[0,:,-2]    # SDD results
    np.savetxt(file_name, results_data, delimiter=',', header='L_link_km,SNR,MI,SDD_MI,SER,SDD_SER')


folder_path = "/Users/diegofigueroa/Desktop/results2/TRAIN_CE"
mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
num_Stage = 4
for mod_format in mod_formats:
    data = []
    for i in range(1,num_Stage+1):
        data.append(read_data(f"{folder_path}/{mod_format}_{i}/results_S=4_s={i}.txt"))
    write_rnn_results(f"{mod_format}.txt", np.array(data))
