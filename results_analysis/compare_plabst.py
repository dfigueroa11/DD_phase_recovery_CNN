import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)
    
def rate_ax_setup(ax: Axes, xlim, ylim, xlabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(visible=True)

def SER_ax_setup(ax: Axes, xlim, ylim, xlabel):
    ax.set_xlim(xlim)
    ax.set_yscale("log")
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
        
def plot_FCN_CNN_Plabst_comparison(paths, mod_format, labels, MI_plot=True, SER_plot=False, pdf=None):
    data_cnn = np.array([read_data(path) for path in paths[1]])
    Llink_list = np.unique(data_cnn[0,:,fc.L_link_column_ppr])
    SNR_list = np.unique(data_cnn[0,:,fc.SNR_dB_column_ppr])
    data_cnn_ser = np.reshape(data_cnn[:,:,fc.min_SER_column_ppr].T, (Llink_list.size, SNR_list.size, -1))
    data_cnn_MI = np.reshape(data_cnn[:,:,fc.max_MI_column_ppr].T, (Llink_list.size, SNR_list.size, -1))
    data_fcn = np.array([read_data(path) for path in paths[0]])
    data_fcn_ser = np.reshape(data_fcn[:,:,-2].T, (Llink_list.size, SNR_list.size, -1))
    data_fcn_MI = np.reshape(data_fcn[:,:,-1].T, (Llink_list.size, SNR_list.size, -1))
    data_plabst = (read_data(paths[2]))
    data_plabst_ser = np.reshape(data_plabst[:,[fc.SER_Plabst_column_ppr, fc.SDD_SER_Plabst_column_ppr]], (Llink_list.size, SNR_list.size, -1))
    data_plabst_MI = np.reshape(data_plabst[:,[fc.MI_Plabst_column_ppr, fc.SDD_MI_Plabst_column_ppr]], (Llink_list.size, SNR_list.size, -1))
    rate_lims = (0,np.ceil(np.max(data_plabst_MI))*1.05)
    ser_lims = (1e-4, 1)
    ax: Axes
    if MI_plot:
        fig, axs = plt.subplots(2, 3, figsize=(15,9))
        fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
        for i, (Llink, ax) in enumerate(zip(Llink_list, axs.flat)):
            ax.plot(SNR_list, data_fcn_MI[i], label=labels[0], linewidth=2, marker='o')
            for j in range(data_cnn_MI.shape[-1]):
                ax.plot(SNR_list, data_cnn_MI[i,:,j], label=labels[1][j], linewidth=1, marker='o', color=f"C{9-j}")
            ax.plot(SNR_list, data_plabst_MI[i,:,0], label=labels[2][0], linewidth=1, marker='o', mfc='none', linestyle=":", color="gray")
            ax.plot(SNR_list, data_plabst_MI[i,:,1], label=labels[2][0], linewidth=1, marker='o', mfc='none', linestyle="-.", color="gray")
            ax.set_title(f"{Llink:.0f} km")
            rate_ax_setup(ax, (np.min(SNR_list), np.max(SNR_list)), rate_lims, 'SNR [dB]')
        for i in range(3): axs[1][i].set_xlabel('SNR [dB]')
        for i in range(2): axs[i][0].set_ylabel("Rate [bpcu]")
        handles, lbs = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, lbs, loc='lower center', ncol=5, fontsize='large')
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
        
    if SER_plot:
        fig, axs = plt.subplots(2, 3, figsize=(15,9))
        fig.suptitle(f"{mod_format} -- SER", fontsize=16)
        for i, (Llink, ax) in enumerate(zip(Llink_list, axs.flat)):
            ax.plot(SNR_list, data_fcn_ser[i], label=labels[0], linewidth=2, marker='o')
            for j in range(data_cnn_MI.shape[-1]):
                ax.plot(SNR_list, data_cnn_ser[i,:,j], label=labels[1][j], linewidth=1, marker='o', color=f"C{9-j}")
            ax.plot(SNR_list, data_plabst_ser[i,:,0], label=labels[2][0], linewidth=1, marker='o', mfc='none', linestyle=":", color="gray")
            ax.plot(SNR_list, data_plabst_ser[i,:,1], label=labels[2][0], linewidth=1, marker='o', mfc='none', linestyle="-.", color="gray")
            ax.set_title(f"{Llink:.0f} km")
            SER_ax_setup(ax, (np.min(SNR_list), np.max(SNR_list)), ser_lims, 'SNR [dB]')
        for i in range(3): axs[1][i].set_xlabel('SNR [dB]')
        for i in range(2): axs[i][0].set_ylabel("SER")
        handles, lbs = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, lbs, loc='lower center', ncol=5, fontsize='large')
        fig.tight_layout(rect=[0, 0.1, 1, 1])
        if pdf is not None:
            pdf.savefig()
            plt.close()
        else:
            plt.show()
        

if __name__=="__main__":
    mod_formats = ["ASK2","ASK4","QAM4"]
    path = "/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/results_post_processing"
    save_fig = True
    fold_num = 6
    loss_funcs = ["TRAIN_MSE_U_MAG_PHASE",
                  "TRAIN_MSE_U_MAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SLDMAG_PHASE",
                  "TRAIN_MSE_U_SLDMAG_PHASE_PHASE_FIX",
                  "TRAIN_MSE_U_SYMBOLS",
                  "TRAIN_CE_U_SYMBOLS",
                  "BIG_CNN"]

    loss_funcs_fcn = ["TRAIN_CE", "TRAIN_MSE", "TRAIN_MSE_PHASE_FIX"]
    loss_funcs_selectors = [[-1,-2]]*2 + [[-1,-2,-3]]
    file_name = "results.txt"
    labels_cnn = [f"{loss_func[6:]}" for loss_func  in loss_funcs[:-1]]+loss_funcs[-1:]
    labels_fcn = [f"{loss_func[6:]}" for loss_func  in loss_funcs_fcn]
    labels_plabst = ["SIC 4", "SDD"]
    pdf = PdfPages(f"big_comparison{fold_num}.pdf") if save_fig else None


    for mod_format, loss_funcs_selector in zip(mod_formats, loss_funcs_selectors):
        paths_cnn = [f"{path}/{loss_funcs[loss_idx]}/{mod_format}/{file_name}" for loss_idx in loss_funcs_selector]
        lab_cnn = [labels_cnn[loss_idx] for loss_idx in loss_funcs_selector]
        path_plabst = f"/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/Plabst_results/{mod_format}.txt"
        paths_fcn = [f"/Users/diegofigueroa/Desktop/results{fold_num}/{loss_func}/{mod_format}_0/{file_name}" for loss_func in loss_funcs_fcn]
        plot_FCN_CNN_Plabst_comparison([paths_fcn, paths_cnn, path_plabst], mod_format, [labels_fcn, lab_cnn, labels_plabst], SER_plot=False, MI_plot=True, pdf=pdf)
    if save_fig:
        pdf.close()
    