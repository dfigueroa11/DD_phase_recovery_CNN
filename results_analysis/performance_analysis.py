import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

import file_constants as fc

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

def make_all_plots(path_file, SNR_plot=True, Llink_plot=True, pdf=None):
    data = read_data(path_file)
    Llink_list = np.unique(data[:,fc.L_link_column_ppr])
    SNR_list = np.unique(data[:,fc.SNR_dB_column_ppr])
    rate_lims = (0,np.ceil(np.nanmax(data[:,fc.max_MI_column_ppr]))*1.05)
    ser_lims = (1e-4, 1)
        
    data = np.reshape(data,(Llink_list.size, SNR_list.size,-1))
    rate_data = data[:,:,[fc.min_MI_column_ppr, fc.mean_MI_column_ppr, fc.max_MI_column_ppr]]
    SER_data = data[:,:,[fc.min_SER_column_ppr, fc.mean_SER_column_ppr, fc.max_SER_column_ppr]]
            
    ax2: Axes 
    ax1: Axes 
    if SNR_plot:
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"{folder} SER and Rate vs SNR for alpha = 0", fontsize=16)
        for i, Llink in enumerate(Llink_list):
            plot_curve_range(ax1, SNR_list, rate_data[i,:,0], rate_data[i,:,1], rate_data[i,:,2], f"C{i}", f"Link length {Llink:.0f} km")
            plot_curve_range(ax2, SNR_list, SER_data[i,:,0], SER_data[i,:,1], SER_data[i,:,2], f"C{i}", f"Link length {Llink:.0f} km")
        rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
        SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')
        if pdf is not None:
            pdf.savefig()

    if Llink_plot:
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"{folder} SER and Rate vs link length for alpha = 0", fontsize=16)
        for i, SNR in enumerate(SNR_list):
            plot_curve_range(ax1, Llink_list, rate_data[:,i,0], rate_data[:,i,1], rate_data[:,i,2], f"C{i}", f"SNR {SNR:.0f} dB")
            plot_curve_range(ax2, Llink_list, SER_data[:,i,0], SER_data[:,i,1], SER_data[:,i,2], f"C{i}", f"SNR {SNR:.0f} dB")
        rate_ax_setup(ax1, (np.min(Llink_list), np.max(Llink_list)), rate_lims, 'Link length km', 'upper left')
        SER_ax_setup(ax2, (np.min(Llink_list), np.max(Llink_list)), ser_lims, 'Link length km', 'lower left')
        if pdf is not None:
            pdf.savefig()
    if pdf is not None:
        plt.close()
    else:
        plt.show()


if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    loss_funcs = ["TRAIN_MSE_U_SYMBOLS",
                  "TRAIN_MSE_U_MAG_PHASE",
                  "TRAIN_MSE_U_MAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SLDMAG_PHASE",
                  "TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX"]
                #   "TRAIN_CE_U_SYMBOLS"]

    mod_formats = ["ASK2","ASK4","PAM2","PAM4", "QAM4"]
    file_name = "results.txt"
    save_fig = False
    pdf = PdfPages(f"cnn_results_all.pdf") if save_fig else None
    for loss_func in loss_funcs:
        for mod_format in mod_formats:
            folder = f"{loss_func}/{mod_format}"
            path_file = f"{path}/{folder}/{file_name}"
            make_all_plots(path_file, SNR_plot=True, Llink_plot=False, pdf=pdf)
        if save_fig:
            pdf.close()