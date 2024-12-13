import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D  # Import for custom legend entries

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def plot_complexity(mod_format, n_layers, n_str_design, complexities):
    fig, axs = plt.subplots(len(n_layers)+1, len(complexities), figsize=(15,9))
    fig.suptitle(f"Performance for different structures -- {mod_format}, 9 dB")
    for i, n_layer in enumerate(n_layers):
        for j, complexity in enumerate(complexities):
            ## read data
            file_path = f"/Users/diegofigueroa/Desktop/results_fix_comp_numch/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            path_plabst = f"/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/Plabst_results/{mod_format}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            L_link_list = data[:,0,0]
            data_plabst = (read_data(path_plabst))
            data_plabst_MI = data_plabst[:,[fc.MI_Plabst_column_ppr, fc.SDD_MI_Plabst_column_ppr]]
            idx = np.nonzero(np.isin(data_plabst[:,fc.L_link_Plabst_column_ppr],L_link_list) & (data_plabst[:,fc.SNR_Plabst_column_ppr] == 9))
            ref_MI = data_plabst_MI[idx]
            ## plot
            ax: Axes
            ax = axs[i][j]
            for k, L_link in enumerate(L_link_list):
                ax.set_title(f"L={n_layer} -- C={complexity}")
                # ax.plot(data[i,:,-2],':', alpha=0.7, c=f"C{i}")
                ax.plot(data[k,:,-2],'o', c=f"C{k}")
                ax.axhline(ref_MI[k,0], c=f"C{k}", label=f"{L_link:2>.0f} km")
            ax.set_xlabel("Structure number")
            ax.set_ylabel("Rate [bpcu]")
            ax.set_xlim([0,data.shape[1]])
            ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-1)))
            ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-2)), minor=True)
            ax.grid()
            ax.grid(which='minor', linestyle=":")
    
    legend_handles = []
    for i, L_link in enumerate(L_link_list):
        legend_handles.append(Line2D([0], [0], linestyle='', marker='o', color=f"C{i}", label=f"{L_link:2>.0f} km"))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize='large')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

def plot_performance_complexity(mod_format, n_layers, n_str_design, complexities):
    fig = plt.figure(figsize=(15,9))
    fig.suptitle(f"rate/complexity for different structures -- {mod_format}, 9 dB")
    outer_grid = gridspec.GridSpec(len(n_layers), len(complexities), wspace=0.2, hspace=0.3)
    for i, n_layer in enumerate(n_layers):
        for j, complexity in enumerate(complexities):
            ## read data
            file_path = f"/Users/diegofigueroa/Desktop/results_fix_comp_numch/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            L_link_list = data[:,0,0]
            data_MI_vs_comp = data[:,:,-2]/data[:,:,-1]
            ## plot
            inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[len(complexities)*i+j], hspace=0)
            ax1 = plt.Subplot(fig, inner_grid[0])
            ax2 = plt.Subplot(fig, inner_grid[1], sharex=ax1)
            ax3 = plt.Subplot(fig, inner_grid[2], sharex=ax1)

            for k, ax in enumerate([ax1,ax2,ax3]):
                # ax.set_title(f"L={n_layer} -- C={complexity}")
                ax.plot(data_MI_vs_comp[k,:],'o:', c=f"C{k}")
                ax.set_xlim([0,data.shape[1]])
                ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-1)))
                ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-2)), minor=True)
                ax.set_ylim([0,data_MI_vs_comp[k,:].max()*1.1])
                ax.set_yticks([0,data_MI_vs_comp[k,:].max()])
                ax.set_yticklabels(["0", "max"])
                ax.grid()
                ax.grid(which='minor', linestyle=":")
            
            ax1.set_title(f"L={n_layer} -- C={complexity}")
            ax3.set_xlabel("Structure number")
            ax2.set_ylabel("Rate/complexity u.a.")
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            fig.add_subplot(ax3)
    
    legend_handles = []
    for i, L_link in enumerate(L_link_list):
        legend_handles.append(Line2D([0], [0], linestyle='--', marker='o', color=f"C{i}", label=f"{L_link:2>.0f} km"))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize='large')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig
    

mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
n_layers = [2]
complexities = [200,500,1000]
L_link_list = [0,12,30]
n_str_design = 4
pdf = PdfPages(f"several_structures_performance_numch.pdf")
for mod_format in mod_formats:
    plot_complexity(mod_format, n_layers, n_str_design, complexities)
    pdf.savefig()
    plt.close()
pdf.close()
pdf = PdfPages(f"several_structures_IM_complexity_numch.pdf")
for mod_format in mod_formats:
    plot_performance_complexity(mod_format, n_layers, n_str_design, complexities)
    pdf.savefig()
    plt.close()
pdf.close()
