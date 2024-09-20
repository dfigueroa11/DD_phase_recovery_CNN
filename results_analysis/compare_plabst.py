import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)
    
def rate_ax_setup(ax: Axes, xlim, ylim, xlabel, loc):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel("Rate [bpcu]")
    ax.set_ylim(ylim)
    ax.grid(visible=True)
    ax.legend(loc=loc)

def SER_ax_setup(ax: Axes, xlim, ylim, xlabel, loc):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_yscale("log")
    ax.set_ylabel("SER")
    ax.set_ylim(ylim)
    ax.grid(visible=True, which='both')
    ax.legend(loc=loc)

def plot_CNN_Plabst_comparison(path_cnn, path_plabst, mod_format):
    data_cnn = np.delete(read_data(path_cnn), (0,2), axis=1)
    data_plabst = (read_data(path_plabst))
    if data_cnn.shape[-1] > 9:
        data_cnn = np.delete(data_cnn, (3,4,5,6,7,8), axis=1)
    data_cnn = np.delete(data_cnn, (2,4,5,7,8), axis=1)

    Llink_list = np.unique(data_cnn[:,0])
    SNR_list = np.unique(data_cnn[:,1])
    data_ser = np.reshape(np.concatenate((data_cnn[:,2:3], data_plabst[:,-2:]), axis=-1), (Llink_list.size, SNR_list.size, -1))
    data_MI = np.reshape(np.concatenate((data_cnn[:,3:4], data_plabst[:,2:4]), axis=-1), (Llink_list.size, SNR_list.size, -1))
    rate_lims = (0,np.ceil(np.max(data_MI))*1.05)
    ser_lims = (1e-4, 1)
    labels = ["CNN", "SIC (4)", "SDD"]

    ax2: Axes 
    ax1: Axes 
    for i, Llink in enumerate(Llink_list):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"{mod_format} -- SER and Rate for Link length {Llink:.0f} km", fontsize=16)
        ax1.plot(SNR_list, data_MI[i], label=labels, linewidth=2, marker='o')
        ax2.plot(SNR_list, data_ser[i], label=labels, linewidth=2, marker='o')
        rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
        SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')
    
    plt.show()


if __name__=="__main__":
    mod_formats = ["ASK2","ASK4","PAM2","PAM4"]
    for mod_format in mod_formats:
        path_cnn = f"/Users/diegofigueroa/Desktop/results_post_processing/{mod_format}_sym/SER_results.txt"
        path_plabst = f"/Users/diegofigueroa/Desktop/Plabst_results/{mod_format}.txt"
        plot_CNN_Plabst_comparison(path_cnn, path_plabst, mod_format)
