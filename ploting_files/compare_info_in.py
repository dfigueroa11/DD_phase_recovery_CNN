import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.lines import Line2D
import re


folders = ["phase_diff_mag_in",
           "phase_diff"]


all_data = []
for folder in folders:
    path = f"/Users/diegofigueroa/Desktop/results_SNR_histogram/{folder}/ASK4"

    data = []
    with open(f"{path}/SER_results.txt", 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        # Extract all numerical values including decimals and scientific notation
        numbers = re.findall(r'[-+]?\d*\.\d+e[-+]?\d+|[-+]?\d*\.\d+|\d+', line)
        # Convert extracted numbers to float and append to data
        data.append([float(num) for num in numbers])

    data = np.array(data)
    lr_list = np.unique(data[:,0])
    Llink_list = np.unique(data[:,1])
    alpha_list = np.unique(data[:,2])
    SNR_list = np.unique(data[:,3])
    SER_data = np.reshape(data[:,-1],(np.size(lr_list), np.size(Llink_list), np.size(alpha_list), np.size(SNR_list)))
    all_data.append(SER_data)
all_data = np.array(all_data)


ax: matplotlib.axes.Axes
fig, axs = plt.subplots(2, 2, figsize=(15,9))
for j, (lr, ax) in enumerate(zip(lr_list, axs.flat)):
    ax.set_title(f"Lr={lr} -- alpha={alpha_list[0]}")
    for i, Llink in enumerate(Llink_list):
        ax.semilogy(SNR_list, all_data[0,j,i,0,:], linestyle='-', marker='o', color=f'C{i}', label=f"L_link={Llink}km")
        ax.semilogy(SNR_list, all_data[1,j,i,0,:], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend()
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.3, 0))
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_SNR_1.pdf')
plt.close()

fig, axs = plt.subplots(2,3, figsize=(15,9))
for j, (llink, ax) in enumerate(zip(Llink_list, axs.flat)):
    ax.set_title(f"L_link={llink}km -- alpha={alpha_list[0]}")
    for i, lr in enumerate(lr_list):
        ax.semilogy(SNR_list, all_data[0,i,j,0,:], linestyle='-', marker='o', color=f'C{i}', label=f"Lr={lr}")
        ax.semilogy(SNR_list, all_data[1,i,j,0,:], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend(loc='lower left')
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.38, 0))
    ax.set_xlabel("SNR [dB]")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_SNR_2.pdf')
plt.close()

fig, axs = plt.subplots(2,2, figsize=(15,9))
for j, (lr, ax) in enumerate(zip(lr_list, axs.flat)):
    ax.set_title(f"Lr={lr} -- alpha={alpha_list[0]}")
    for i, snr in enumerate(SNR_list):
        ax.semilogy(Llink_list, all_data[0,j,:,0,i], linestyle='-', marker='o', color=f'C{i}', label=f"SNR={snr} dB")
        ax.semilogy(Llink_list, all_data[1,j,:,0,i], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend()
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.29, 0))
    ax.set_xlabel("link length [km]")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_Llink_1.pdf')
plt.close()

fig, axs = plt.subplots(2,3, figsize=(15,9))
for j, (snr, ax) in enumerate(zip(SNR_list, axs.flat)):
    ax.set_title(f"SNR={snr} dB -- alpha={alpha_list[0]}")
    for i, lr in enumerate(lr_list):
        ax.semilogy(Llink_list, all_data[0,i,:,0,j], linestyle='-', marker='o', color=f'C{i}', label=f"Lr={lr}")
        ax.semilogy(Llink_list, all_data[1,i,:,0,j], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend(loc='lower right')
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.3, 0))
    ax.set_xlabel("link length [km]")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_Llink_2.pdf')
plt.close()

fig, axs = plt.subplots(2,3, figsize=(15,9))
for j, (snr, ax) in enumerate(zip(SNR_list, axs.flat)):
    ax.set_title(f"SNR={snr} dB -- alpha={alpha_list[0]}")
    for i, Llink in enumerate(Llink_list):
        ax.semilogy(lr_list, all_data[0,:,i,0,j], linestyle='-', marker='o', color=f'C{i}', label=f"L_link={Llink}km")
        ax.semilogy(lr_list, all_data[1,:,i,0,j], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend(loc='lower left')
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.47, 0))
    ax.set_xlabel("learning rate")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_lr_1.pdf')
plt.close()

fig, axs = plt.subplots(2,3, figsize=(15,9))
for j, (llink, ax) in enumerate(zip(Llink_list, axs.flat)):
    ax.set_title(f"L_link={llink}km -- alpha={alpha_list[0]}")
    for i, snr in enumerate(SNR_list):
        ax.semilogy(lr_list, all_data[0,:,j,0,i], linestyle='-', marker='o', color=f'C{i}', label=f"SNR={snr} dB")
        ax.semilogy(lr_list, all_data[1,:,j,0,i], linestyle='--', marker='o', color=f'C{i}')
    main_legend = ax.legend(loc='lower left')
    ax.add_artist(main_legend)
    line1 = Line2D([1],[1], linestyle='-',color='0.3')
    line2 = Line2D([1],[1], linestyle='--',color='0.3')
    ax.legend([line1, line2], ['info in', 'no info' ], loc='lower left', bbox_to_anchor=(0.44, 0))
    ax.set_xlabel("learning rate")
    ax.set_ylabel("SER")
    ax.set_ylim((1e-5,1))
    ax.grid(True)
fig.savefig(f'sweep_lr_2.pdf')
plt.close()


