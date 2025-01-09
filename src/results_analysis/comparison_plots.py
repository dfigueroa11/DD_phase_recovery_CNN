import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from cycler import cycler

from ploting_tools import plot_data

cc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

#################################################################
# plot comparing normalization vs no normalization in the tvrnn #
#################################################################
save_fig = True
l_link_list = [0, 6, 12, 18, 24, 30]
mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
y_data_selection = [["MI","SDD_MI"]]*2
x_data_selection = ["SNR"]*2
slice_selection = [["L_link_km"]]*2
labels = [[f"norm SIC 4", f"norm SDD"], [f"no norm SIC 4", f"no norm SDD"]]
mpl.rcParams['axes.prop_cycle'] = cycler(color=[cc[0], cc[0], cc[1], cc[1]],
                                         ls=['-','--']*2,
                                         marker=['o',]*4)

pdf = PdfPages(f"tvrnn_norm_vs_no_norm.pdf") if save_fig else None
for mod_format in mod_formats:
    file_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt",
                  f"/Users/diegofigueroa/Desktop/tvrnn_no_norm_results/{mod_format}.txt"]
    ax: Axes
    fig, axs = plt.subplots(2,3, figsize=(15,9))
    fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
    for ax, l_link in zip(axs.flatten(), l_link_list):
        slice_values = [l_link]*2
        plot_data(ax,file_paths, x_data_selection, y_data_selection, labels, slice_selection, slice_values)
        ax.grid()
        ax.set_title(f"link length {l_link} km")
        ax.set_xlim(-5,11)
    for i in range(3): axs[1][i].set_xlabel('SNR [dB]')
    for i in range(2): axs[i][0].set_ylabel("Rate [bpcu]")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    if pdf is not None:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

if save_fig:
    pdf.close()
#################################################################