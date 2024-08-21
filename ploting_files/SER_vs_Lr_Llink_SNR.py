import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
import re


folders = ["DDQAM8_phase_diff_mag_in",
           "DDQAM4_phase_diff",
           "PAM4_odd_samp",
           "ASK4_odd_samp",
           "QAM4_phase_diff",
           "ASK4_phase",
           "ASK4_odd_samp_phase_in",
           "QAM4_phase",
           "ASK4_phase_diff_mag_in",
           "PAM2_odd_samp",
           "ASK4_phase_diff",
           "ASK2_phase",
           "DDQAM4_phase",
           "ASK2_phase_diff"]


for folder in folders:
    path = f"/Users/diegofigueroa/Desktop/results_best/{folder}"

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

    ax: matplotlib.axes.Axes
    fig, axs = plt.subplots(2, 2, figsize=(15,9))
    for j, (lr, ax) in enumerate(zip(lr_list, axs.flat)):
        ax.set_title(f"Lr={lr} -- alpha={alpha_list[0]}")
        for i, Llink in enumerate(Llink_list):
            ax.semilogy(SNR_list, SER_data[j,i,0,:], marker='o', label=f"L_link={Llink}km")
        ax.legend()
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_SNR_1.pdf')
    plt.close()
    
    fig, axs = plt.subplots(2,3, figsize=(15,9))
    for j, (llink, ax) in enumerate(zip(Llink_list, axs.flat)):
        ax.set_title(f"L_link={llink}km -- alpha={alpha_list[0]}")
        for i, lr in enumerate(lr_list):
            ax.semilogy(SNR_list, SER_data[i,j,0,:], marker='o', label=f"Lr={lr}")
        ax.legend()
        ax.set_xlabel("SNR [dB]")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_SNR_2.pdf')
    plt.close()
    
    fig, axs = plt.subplots(2,2, figsize=(15,9))
    for j, (lr, ax) in enumerate(zip(lr_list, axs.flat)):
        ax.set_title(f"Lr={lr} -- alpha={alpha_list[0]}")
        for i, snr in enumerate(SNR_list):
            ax.semilogy(Llink_list, SER_data[j,:,0,i], marker='o', label=f"SNR={snr} dB")
        ax.legend()
        ax.set_xlabel("link length [km]")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_Llink_1.pdf')
    plt.close()
    
    fig, axs = plt.subplots(2,3, figsize=(15,9))
    for j, (snr, ax) in enumerate(zip(SNR_list, axs.flat)):
        ax.set_title(f"SNR={snr} dB -- alpha={alpha_list[0]}")
        for i, lr in enumerate(lr_list):
            ax.semilogy(Llink_list, SER_data[i,:,0,j], marker='o', label=f"Lr={lr}")
        ax.legend()
        ax.set_xlabel("link length [km]")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_Llink_2.pdf')
    plt.close()
    
    fig, axs = plt.subplots(2,3, figsize=(15,9))
    for j, (snr, ax) in enumerate(zip(SNR_list, axs.flat)):
        ax.set_title(f"SNR={snr} dB -- alpha={alpha_list[0]}")
        for i, Llink in enumerate(Llink_list):
            ax.semilogy(lr_list, SER_data[:,i,0,j], marker='o', label=f"L_link={Llink}km")
        ax.legend()
        ax.set_xlabel("learning rate")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_lr_1.pdf')
    plt.close()
    
    fig, axs = plt.subplots(2,3, figsize=(15,9))
    for j, (llink, ax) in enumerate(zip(Llink_list, axs.flat)):
        ax.set_title(f"L_link={llink}km -- alpha={alpha_list[0]}")
        for i, snr in enumerate(SNR_list):
            ax.semilogy(lr_list, SER_data[:,j,0,i], marker='o', label=f"SNR={snr} dB")
        ax.legend()
        ax.set_xlabel("learning rate")
        ax.set_ylabel("SER")
        ax.set_ylim((1e-5,1))
        ax.grid(True)
    fig.savefig(f'{path}/sweep_lr_2.pdf')
    plt.close()
    

