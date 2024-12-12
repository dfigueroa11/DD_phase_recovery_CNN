import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

import file_constants as fc

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

def compare_results(paths, labels, title, save_fig=False):
    data = np.array([read_data(path) for path in paths])
    Llink_list = np.unique(data[0,:,fc.L_link_column_ppr])
    SNR_list = np.unique(data[0,:,fc.SNR_dB_column_ppr])
    data_ser = data[:,:,fc.min_SER_column_ppr].T.reshape((Llink_list.size, SNR_list.size, -1))
    data_MI = data[:,:,fc.max_MI_column_ppr].T.reshape((Llink_list.size, SNR_list.size, -1))
    rate_lims = (0,np.ceil(np.nanmax(data_MI))*1.05)
    ser_lims = (1e-4, 1)
    
    ax2: Axes 
    ax1: Axes 
    if save_fig:
        pdf = PdfPages(f"{labels[0]}_vs_{labels[1]}.pdf")
    for i, Llink in enumerate(Llink_list):
        fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,9))
        fig.suptitle(f"{title} -- Link length {Llink} km", fontsize=16)
        ax1.plot(SNR_list, data_MI[i], label=labels, linewidth=2, marker='o')
        ax2.plot(SNR_list, data_ser[i], label=labels, linewidth=2, marker='o')
        rate_ax_setup(ax1, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]', 'upper left')
        SER_ax_setup(ax2, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]', 'lower left')
        if save_fig:
            pdf.savefig()
    if save_fig:
        plt.close()
        pdf.close()
    else:
        plt.show()


if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    file_name = "results.txt"
    loss_funcs = ["TRAIN_MSE_U_SYMBOLS",
                  "TRAIN_MSE_U_MAG_PHASE",
                  "TRAIN_MSE_U_MAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SLDMAG_PHASE",
                  "TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX",
                  "TRAIN_CE_U_SYMBOLS"]
    mod_formats = ["ASK2","ASK4","PAM2","PAM4","QAM4"]
    
    ###### compare modulation formats
    # mod_formats_comp = [["ASK2","PAM2"],["ASK4","PAM4"]]
    # for loss_func in loss_funcs:
    #     for mod_format in mod_formats_comp:
    #         path_mod1 = f"{path}/{loss_func}/{mod_format[0]}/{file_name}"
    #         path_mod2 = f"{path}/{loss_func}/{mod_format[1]}/{file_name}"
    #         title = f"{loss_func} -- {mod_format[0]} vs {mod_format[1]}"
    #         compare_results([path_mod1, path_mod2], mod_format, title, save_fig=False)
    
    ###### compare loss functions
    for mod_format in mod_formats[-2:]:
        paths = [f"{path}/{loss_func}/{mod_format}/{file_name}" for loss_func  in loss_funcs]
        labels = [f"{loss_func[6:]}" for loss_func  in loss_funcs]
        title = f"Comparison of loss functions for {mod_format}"
        compare_results(paths, labels, title, save_fig=False)
    
