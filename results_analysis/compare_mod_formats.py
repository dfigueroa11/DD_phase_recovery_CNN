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

def compare_mod_formats(path_mod1, path_mod2, labels):
    data = []
    data.append(np.delete(read_data(path_mod1), (0,2), axis=1))
    data.append(np.delete(read_data(path_mod2), (0,2), axis=1))
    for i,_ in enumerate(data):
        if data[i].shape[-1] > 9:
            data[i] = np.delete(data[i], (3,4,5,6,7,8), axis=1)
        data[i] = np.delete(data[i], (2,4,5,7,8), axis=1)

    data = np.array(data)
    Llink_list = np.unique(data[0,:,0])
    SNR_list = np.unique(data[0,:,1])
    data_ser = np.concatenate((data[0,:,2:3],data[1,:,2:3]), axis=-1).reshape((Llink_list.size, SNR_list.size, -1))
    data_MI = np.concatenate((data[0,:,3:],data[1,:,3:]), axis=-1).reshape((Llink_list.size, SNR_list.size, -1))
    rate_lims = (0,np.ceil(np.max(data_MI))*1.05)
    ser_lims = (1e-4, 1)
    
    ax2: Axes 
    ax1: Axes 
    for i, Llink in enumerate(Llink_list):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"SER and Rate for Link length {Llink} km", fontsize=16)
        ax1.plot(SNR_list, data_MI[i], label=labels, linewidth=2, marker='o')
        ax2.plot(SNR_list, data_ser[i], label=labels, linewidth=2, marker='o')
        rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
        SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')

    plt.show()


if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    folders = ["ASK2_sym","PAM2_sym"]
    file_name = "SER_results.txt"
    path_mod1 = f"{path}/{folders[0]}/{file_name}"
    path_mod2 = f"{path}/{folders[1]}/{file_name}"
    labels = ["ASK 2","PAM 2"]
    
    compare_mod_formats(path_mod1, path_mod2, labels)
