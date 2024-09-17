import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def plot_curve_range(ax: Axes, x_data, y_min, y_mean, y_max, color, label):
    ax.plot(x_data, y_mean, label=label, color=color, linewidth=1)
    ax.plot(x_data, y_min, color=color, linestyle=':', linewidth=0.5)
    ax.plot(x_data, y_max, color=color, linestyle=':', linewidth=0.5)
    ax.fill_between(x_data, y_min, y_max, color=color, alpha=0.1)

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

def make_all_plots(path_file):
    data = np.delete(read_data(path_file), (0,2), axis=1)
    if data.shape[-1] > 9:
        data = np.delete(data, (3,4,5,6,7,8), axis=1)

    Llink_list = np.unique(data[:,0])
    SNR_list = np.unique(data[:,1])
    rate_lims = (0,np.ceil(np.max(data[:,7]))*1.05)
    ser_lims = (1e-4, 1)
        
    data = np.reshape(data,(Llink_list.size, SNR_list.size,-1))
    ax2: Axes 
    ax1: Axes 
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
    fig.suptitle(f"{folder} SER and Rate vs SNR for alpha = 0", fontsize=16)
    for i, Llink in enumerate(Llink_list):
        plot_curve_range(ax1, SNR_list, data[i,:,5], data[i,:,6], data[i,:,7], f"C{i}", f"Link length {Llink:.0f} km")
        plot_curve_range(ax2, SNR_list, data[i,:,2], data[i,:,3], data[i,:,4], f"C{i}", f"Link length {Llink:.0f} km")
    rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
    SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
    fig.suptitle(f"{folder} SER and Rate vs link length for alpha = 0", fontsize=16)
    for i, SNR in enumerate(SNR_list):
        plot_curve_range(ax1, Llink_list, data[:,i,5], data[:,i,6], data[:,i,7], f"C{i}", f"SNR {SNR:.0f} dB")
        plot_curve_range(ax2, Llink_list, data[:,i,2], data[:,i,3], data[:,i,4], f"C{i}", f"SNR {SNR:.0f} dB")
    rate_ax_setup(ax1, (np.min(Llink_list), np.max(Llink_list)), rate_lims, 'Link length km', 'upper left')
    SER_ax_setup(ax2, (np.min(Llink_list), np.max(Llink_list)), ser_lims, 'Link length km', 'lower left')
    

    plt.show()


if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    folder = "ASK4_sym"
    file_name = "SER_results.txt"
    path_file = f"{path}/{folder}/{file_name}"

    make_all_plots(path_file)
