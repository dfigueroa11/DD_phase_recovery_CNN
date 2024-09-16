import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def read_data(path, folder, file_name):
    return np.loadtxt(f"{path}/{folder}/{file_name}", delimiter=",", skiprows=1)

def plot_curve_comparison(ax: Axes, x_data, y_1, y_2, labels):
    ax.plot(x_data, y_1, label=labels[0], linewidth=1)
    ax.plot(x_data, y_2, label=labels[1], linewidth=1)

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







if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    folders = ["ASK2_sym","PAM2_sym"]
    file_name = "SER_results.txt"
    data = []
    data.append(np.delete(read_data(path, folders[0], file_name), (0,2), axis=1))
    data.append(np.delete(read_data(path, folders[1], file_name), (0,2), axis=1))
    for i,_ in enumerate(data):
        if data[i].shape[-1] > 9:
            data[i] = np.delete(data[i], (3,4,5,6,7,8), axis=1)
        data[i] = np.delete(data[i], (2,4,5,7,8), axis=1)

    data = np.array(data)
    Llink_list = np.unique(data[0,:,0])
    SNR_list = np.unique(data[0,:,1])
    rate_lims = (0,2.1)
    ser_lims = (1e-3, 1)
    
    data = np.reshape(data,(2,Llink_list.size, SNR_list.size,-1))[:,:,:,2:]
    ax2: Axes 
    ax1: Axes 
    for i, Llink in enumerate(Llink_list):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"SER and Rate for Link length {Llink} km", fontsize=16)
        plot_curve_comparison(ax1, SNR_list, data[0,i,:,1], data[1,i,:,1], folders)
        plot_curve_comparison(ax2, SNR_list, data[0,i,:,0], data[1,i,:,0], folders)
        rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
        SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')
    


    plt.show()


