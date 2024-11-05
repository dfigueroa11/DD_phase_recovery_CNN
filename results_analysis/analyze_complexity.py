import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def plot_complexity(mod_format, n_layer, n_str_design):
    file_path = f"/Users/diegofigueroa/Desktop/{mod_format}/results_L={n_layer}_C=1000.txt"
    path_plabst = f"/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/Plabst_results/{mod_format}.txt"
    data = read_data(file_path).reshape(3,-1,6)
    L_link_list = data[:,0,0]
    data_plabst = (read_data(path_plabst))
    data_plabst_MI = data_plabst[:,[fc.MI_Plabst_column_ppr, fc.SDD_MI_Plabst_column_ppr]]
    idx = np.nonzero(np.isin(data_plabst[:,fc.L_link_Plabst_column_ppr],L_link_list) & (data_plabst[:,fc.SNR_Plabst_column_ppr] == 9))
    ref_MI = data_plabst_MI[idx]
    print(ref_MI[:,0])
    ax: Axes
    fig, ax = plt.subplots(figsize=(15,9))
    for i, L_link in enumerate(L_link_list):
        ax.set_title(f"Performance for different structures -- {mod_format}, 9 dB, {n_layer} layers, C=1000")
        ax.plot(data[i,:,-2],':', alpha=0.7, c=f"C{i}")
        ax.plot(data[i,:,-2],'o', c=f"C{i}")
        ax.axhline(ref_MI[i,0], c=f"C{i}", label=f"{L_link:2>.0f} km")
    ax.set_xlabel("Structure number")
    ax.set_ylabel("Rate [bpcu]")
    ax.legend()
    ax.set_xlim([0,data.shape[1]])
    ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design))
    ax.grid()
    plt.show()


mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
n_layers = [2,3]
n_str_design = 4

for n_layer in n_layers:
    for mod_format in mod_formats:
        plot_complexity(mod_format, n_layer, n_str_design)