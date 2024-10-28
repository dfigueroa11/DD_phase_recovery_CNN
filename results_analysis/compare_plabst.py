import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)
    
def rate_ax_setup(ax: Axes, xlim, ylim, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_ylabel("Rate [bpcu]")
    ax.set_ylim(ylim)
    ax.grid(visible=True)

def SER_ax_setup(ax: Axes, xlim, ylim, xlabel):
    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)
    ax.set_yscale("log")
    ax.set_ylabel("SER")
    ax.set_ylim(ylim)
    ax.grid(visible=True, which='both')

def plot_CNN_Plabst_comparison(paths_cnn, path_plabst, mod_format, labels, MI_plot=True, SER_plot=False, pdf=None):
    data_cnn = np.array([read_data(path) for path in paths_cnn])
    Llink_list = np.unique(data_cnn[0,:,fc.L_link_column_ppr])
    SNR_list = np.unique(data_cnn[0,:,fc.SNR_dB_column_ppr])
    data_cnn_ser = data_cnn[:,:,fc.min_SER_column_ppr].T
    data_cnn_MI = data_cnn[:,:,fc.max_MI_column_ppr].T
    data_plabst = (read_data(path_plabst))
    data_plabst_ser = data_plabst[:,[fc.SER_Plabst_column_ppr, fc.SDD_SER_Plabst_column_ppr]]
    data_plabst_MI = data_plabst[:,[fc.MI_Plabst_column_ppr, fc.SDD_MI_Plabst_column_ppr]]
    data_ser = np.reshape(np.concatenate((data_cnn_ser, data_plabst_ser), axis=-1), (Llink_list.size, SNR_list.size, -1))
    data_MI = np.reshape(np.concatenate((data_cnn_MI, data_plabst_MI), axis=-1), (Llink_list.size, SNR_list.size, -1))
    rate_lims = (0,np.ceil(np.max(data_MI))*1.05)
    ser_lims = (1e-4, 1)

    if MI_plot:
        fig, axs = plt.subplots(2, 3, figsize=(15,9), sharex=True, sharey=True)
        fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
        for i, (Llink, ax) in enumerate(zip(Llink_list, axs.flat)):
            ax.plot(SNR_list, data_MI[i], label=labels, linewidth=2, marker='o')
            ax.set_title(f"{Llink:.0f} km")
            rate_ax_setup(ax, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, fontsize='large')
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
        
    if SER_plot:
        fig, axs = plt.subplots(2, 3, figsize=(15,9), sharex=True, sharey=True)
        fig.suptitle(f"{mod_format} -- SER", fontsize=16)
        for i, (Llink, ax) in enumerate(zip(Llink_list, axs.flat)):
            ax.plot(SNR_list, data_ser[i], label=labels, linewidth=2, marker='o')
            ax.set_title(f"{Llink:.0f} km")
            SER_ax_setup(ax, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]')
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, fontsize='large')
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
        

if __name__=="__main__":
    mod_formats = ["ASK2","ASK4","PAM2","PAM4","QAM4"]
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    save_fig = True
    loss_funcs = ["TRAIN_MSE_U_MAG_PHASE",
                  "TRAIN_MSE_U_MAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SLDMAG_PHASE",
                  "TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SYMBOLS",
                  "TRAIN_CE_U_SYMBOLS",
                  "BIG_CNN"]
    file_name = "results.txt"
    labels = [f"{loss_func[6:]}" for loss_func  in loss_funcs[:-1]]+loss_funcs[-1:]+["SIC 4", "SDD"]
    pdf = PdfPages(f"big_comparison.pdf") if save_fig else None
    for mod_format in mod_formats:
        paths_cnn = [f"{path}/{loss_func}/{mod_format}/{file_name}" for loss_func in loss_funcs]
        path_plabst = f"/Users/diegofigueroa/Desktop/Plabst_results/{mod_format}.txt"
        plot_CNN_Plabst_comparison(paths_cnn, path_plabst, mod_format, labels, SER_plot=False, MI_plot=True, pdf=pdf)
    if save_fig:
        pdf.close()
    