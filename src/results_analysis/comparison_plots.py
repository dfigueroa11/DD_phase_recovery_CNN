from itertools import product
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
# save_fig = True
# l_link_list = [0, 6, 12, 18, 24, 30]
# mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
# y_data_selection = [["MI","SDD_MI"]]*2
# x_data_selection = ["SNR"]*2
# slice_selection = [["L_link_km"]]*2
# labels = [[f"norm SIC 4", f"norm SDD"], [f"no norm SIC 4", f"no norm SDD"]]
# mpl.rcParams['axes.prop_cycle'] = cycler(color=[cc[0], cc[0], cc[1], cc[1]],
#                                          ls=['-','--']*2,
#                                          marker=['o',]*4)

# pdf = PdfPages(f"tvrnn_norm_vs_no_norm.pdf") if save_fig else None
# for mod_format in mod_formats:
#     file_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt",
#                   f"/Users/diegofigueroa/Desktop/tvrnn_no_norm_results/{mod_format}.txt"]
#     ax: Axes
#     fig, axs = plt.subplots(2,3, figsize=(15,9))
#     fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
#     for ax, l_link in zip(axs.flatten(), l_link_list):
#         slice_values = [l_link]*2
#         plot_data(ax,file_paths, x_data_selection, y_data_selection, labels, slice_selection, slice_values)
#         ax.grid()
#         ax.set_title(f"link length {l_link} km")
#         ax.set_xlim(-5,11)
#     for i in range(3): axs[1][i].set_xlabel('SNR [dB]')
#     for i in range(2): axs[i][0].set_ylabel("Rate [bpcu]")
#     handles, labels = axs[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
#     fig.tight_layout(rect=[0, 0.1, 1, 1])
#     if pdf is not None:
#         pdf.savefig()
#         plt.close()
#     else:
#         plt.show()

# if save_fig:
#     pdf.close()
#################################################################


#########################################################################
# plot comparing normalization vs no normalization in the tvrnn and cnn #
#########################################################################
# save_fig = True
# l_link_list = [0, 6, 12, 18, 24, 30]
# mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
# orders = [2,2,4,4,4]
# cnn_old_x = ["SNR_dB",]
# cnn_old_y = [["max_MI"]]*2
# cnn_old_ss = [["L_link_km"]]*2
# cnn_old_labels = ["CNN old",]

# cnn_x = ["SNR_dB"]*2
# cnn_y = [["MI"]]*2
# cnn_ss = [["L_link_km"]]*2
# cnn_labels = ["CNN norm","CNN no norm"]
# tvrnn_y = [["MI"]]*2+[["SDD_MI"]]*2
# tvrnn_x = ["SNR"]*4
# tvrnn_ss = [["L_link_km"]]*4
# tvrnn_labels = [f"norm SIC 4", f"no norm SIC 4", f"norm SDD", f"no norm SDD"]

# mpl.rcParams['axes.prop_cycle'] = cycler(color=[cc[0], cc[0], cc[1], cc[1], cc[2], cc[2],cc[3]],
#                                          ls=['-','--','-','--','-','--',':'],
#                                          marker=['o',]*7)

# pdf = PdfPages(f"tvrnn_cnn_norm_vs_no_norm.pdf") if save_fig else None
# for mod_format, order in zip(mod_formats, orders):
#     cnn_paths_old = [f"/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/results_post_processing/TRAIN_CE_U_SYMBOLS/{mod_format}/results.txt"]
#     cnn_paths = [f"/Users/diegofigueroa/Desktop/results_norm/TRAIN_CE_U_SYMBOLS/{mod_format}_0/results.txt",
#                 f"/Users/diegofigueroa/Desktop/results_no_norm/TRAIN_CE_U_SYMBOLS/{mod_format}_0/results.txt"]
#     tvrnn_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt",
#                 f"/Users/diegofigueroa/Desktop/tvrnn_no_norm_results/{mod_format}.txt"]*2

#     ax: Axes
#     fig, axs = plt.subplots(2,3, figsize=(15,9))
#     fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
#     for ax, l_link in zip(axs.flatten(), l_link_list):
#         plot_data(ax,cnn_paths+tvrnn_paths+cnn_paths_old, cnn_x+tvrnn_x+cnn_old_x, cnn_y+tvrnn_y+cnn_old_y, cnn_labels+tvrnn_labels+cnn_old_labels, cnn_ss+tvrnn_ss+cnn_old_ss, [[l_link]]*7)
#         ax.grid()
#         ax.set_title(f"link length {l_link} km")
#         ax.set_xlim(-5,11)
#         ax.set_ylim(0, np.log2(order)*1.1)
#     for i in range(3): axs[1][i].set_xlabel('SNR [dB]')
#     for i in range(2): axs[i][0].set_ylabel("Rate [bpcu]")
#     handles, labels = axs[0, 0].get_legend_handles_labels()
#     fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
#     fig.tight_layout(rect=[0, 0.1, 1, 1])
#     if pdf is not None:
#         pdf.savefig()
#         plt.close()
#     else:
#         plt.show()

# if save_fig:
#     pdf.close()
#########################################################################


#########################################################################
#                   plot the learning process for tvrnn                 #
#########################################################################

# SIC_stages = 4
# x_column = ["progress"]*SIC_stages
# y_column_1 = [["lr"]]*SIC_stages
# y_column_2 = [["MI"]]*SIC_stages
# labels = [f"stage {s}" for s in range(1,SIC_stages+1)]

# mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
# l_link_list = [0, 6, 12, 18, 24, 30]
# SNR_dB_list = SNR_dB_steps = [x for x in range(-5, 12, 2)]

# pdf = PdfPages(f"tvrnn_no_norm_progress.pdf")
# for mod_format in mod_formats:
#     for l_link in l_link_list:
#         for SNR_dB in SNR_dB_list:
#             # path = [f"/Users/diegofigueroa/Desktop/results2_norm/TRAIN_CE/{mod_format}_{s}/progress_S=4_s={s}_lr0p02_Llink{l_link}km_alpha0p0_{SNR_dB}dB.txt" for s in range(1,SIC_stages+1)]
#             # path = [f"/Users/diegofigueroa/Desktop/results2_no_norm/TRAIN_CE/{mod_format}_{s}/progress_S=4_s={s}_lr0p02_Llink{l_link}km_alpha0p0_{SNR_dB}dB.txt" for s in range(1,SIC_stages+1)]
#             axs: list[Axes]
#             fig, axs = plt.subplots(2,1, figsize=(15,9))
#             fig.suptitle(f"{mod_format} -- Link length: {l_link} km -- SNR: {SNR_dB} dB", fontsize=16)
#             plot_data(axs[0],path,x_column,y_column_1,labels)
#             plot_data(axs[1],path,x_column,y_column_2,labels)
#             axs[0].grid()
#             axs[0].set_xlim(0,1)
#             axs[0].set_ylabel("learning rate")
#             axs[0].set_yscale('log')

#             axs[1].grid()
#             axs[1].set_xlim(0,1)
#             axs[1].set_ylabel("rate [bpcu]")
#             axs[1].set_xlabel("progress")

#             handles, labels = axs[0].get_legend_handles_labels()
#             fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
#             fig.tight_layout(rect=[0.05, 0.05, 0.95, 1])
#             pdf.savefig()
#             plt.close()
# pdf.close()
#########################################################################




#########################################################################
#                   plot the learning process for cnn                   #
#########################################################################

# x_column = []*2
# y_column_1 = [["lr"]]*2
# y_column_2 = [["MI"]]*2
# labels = ["CNN norm","CNN no norm"]

# mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
# l_link_list = [0, 6, 12, 18, 24, 30]
# SNR_dB_list = SNR_dB_steps = [x for x in range(-5, 12, 2)]

# pdf = PdfPages(f"cnn_progress.pdf")
# for mod_format in mod_formats:
#     for l_link in l_link_list:
#         for SNR_dB in SNR_dB_list:
#             axs: list[Axes]
#             paths = [f"/Users/diegofigueroa/Desktop/results_norm/TRAIN_CE_U_SYMBOLS/{mod_format}_0/progress_lr0p004_Llink{l_link}km_alpha0p0_{SNR_dB}dB.txt",
#                      f"/Users/diegofigueroa/Desktop/results_no_norm/TRAIN_CE_U_SYMBOLS/{mod_format}_0/progress_lr0p004_Llink{l_link}km_alpha0p0_{SNR_dB}dB.txt"]
#             fig, axs = plt.subplots(2,1, figsize=(15,9))
#             fig.suptitle(f"{mod_format} -- Link length: {l_link} km -- SNR: {SNR_dB} dB", fontsize=16)
#             plot_data(axs[0],paths,x_column,y_column_1,labels)
#             plot_data(axs[1],paths,x_column,y_column_2,labels)
#             axs[0].grid()
#             axs[0].set_xlim(0,299)
#             axs[0].set_ylabel("learning rate")
#             axs[0].set_yscale('log')

#             axs[1].grid()
#             axs[1].set_xlim(0,299)
#             axs[1].set_ylabel("rate [bpcu]")
#             axs[1].set_xlabel("progress")

#             handles, labels = axs[0].get_legend_handles_labels()
#             fig.legend(handles, labels, loc='lower center', ncol=4, fontsize='large')
#             fig.tight_layout(rect=[0.05, 0.05, 0.95, 1])
#             pdf.savefig()
#             plt.close()
# pdf.close()



#########################################################################
#################### plot comparing tvrnn complexity ####################
#########################################################################
save_fig = False
l_link_list = [0, 12, 30]
L_y_list = [10, 20, 30]
hidden_states_size_list = [10, 20, 30]
Bi_dir_list = [True, False]
path_combinations = list(product(Bi_dir_list, L_y_list, hidden_states_size_list))
mod_formats = ["ASK2", "PAM2", "ASK4", "PAM4", "QAM4"]
orders = [2,2,4,4,4]

tvrnn_y = [["MI"]]*18
tvrnn_x = ["SNR_dB"]*18
tvrnn_ss = [["L_link_km"]]*18

tvrnn_labels = [f'L_y={pc[1]} hs={pc[2]}' for pc in path_combinations]
    

mpl.rcParams['axes.prop_cycle'] = cycler(color=cc[:9]*2,
                                         ls=['-',]*9+['--',]*9,
                                         marker=['o',]*18)

pdf = PdfPages(f"tvrnn_cnn_norm_vs_no_norm.pdf") if save_fig else None
for mod_format, order in zip(mod_formats, orders):
    tvrnn_paths = [f'/Users/diegofigueroa/Desktop/results/{mod_format}/results_S=1_s=1_L_y={pc[1]}_hs=[{pc[2]}]_Bi_dir={pc[0]}.txt' for pc in path_combinations]
    ax: Axes
    fig, axs = plt.subplots(1,3, figsize=(15,9))
    fig.suptitle(f"{mod_format} -- Rate", fontsize=16)
    for ax, l_link in zip(axs.flatten(), l_link_list):
        plot_data(ax,tvrnn_paths, tvrnn_x, tvrnn_y, tvrnn_labels, tvrnn_ss, [[l_link]]*18)
        ax.grid()
        ax.set_title(f"link length {l_link} km")
        ax.set_xlim(-5,11)
        ax.set_ylim(0, np.log2(order)*1.1)
        ax.set_xlabel('SNR [dB]')
    axs[0].set_ylabel("Rate [bpcu]")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles[:9], labels[:9], loc='lower center', ncol=3, fontsize='large')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    if pdf is not None:
        pdf.savefig()
        plt.close()
    else:
        plt.show()

if save_fig:
    pdf.close()
