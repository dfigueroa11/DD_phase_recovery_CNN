import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.axes as axes


import help_functions as hlp

def print_progress(y_ideal, y_hat, batch_size, progress, loss, multi_mag, multi_phase):
    if multi_mag and multi_phase:
        _, mag_ER = hlp.decode_and_ER(y_ideal[:,0,:], y_hat[:,0,:])
        _, phase_ER = hlp.decode_and_ER(y_ideal[:,1,:], y_hat[:,1,:])
        _, SER = hlp.decode_and_ER_mag_phase(y_ideal, y_hat)
        print(f"\tBatch size {batch_size:_}\tprogress {progress:>6.1%}\tloss: {loss:.3e}\tmag ER: {mag_ER:.3e}\tphase ER: {phase_ER:.3e}\tSER: {SER:.3e}",end='\r')
    else:
        _, SER = hlp.decode_and_ER(y_ideal, y_hat)
        print(f"\tBatch size {batch_size:_}\tprogress {progress:>6.1%}\tloss: {loss:.3e}\tSER: {SER:.3e}",end='\r')

def print_save_summary(y_ideal, y_hat, multi_mag, multi_phase, lr, L_link, alpha, SNR_dB, path):
    if multi_mag and multi_phase:
        alphabet_mag, mag_ER = hlp.decode_and_ER(y_ideal[:,0,:], y_hat[:,0,:])
        alphabet_phase, phase_ER = hlp.decode_and_ER(y_ideal[:,1,:], y_hat[:,1,:])
        alphabet, SER = hlp.decode_and_ER_mag_phase(y_ideal, y_hat)
        print(f"\tmag ER: {mag_ER:.3e}\tphase ER: {phase_ER:.3e}\tSER: {SER:.3e}")
        alphabets = [alphabet_mag, alphabet_phase, alphabet]
        SERs = [mag_ER, phase_ER, SER]
    else:
        alphabet, SER = hlp.decode_and_ER(y_ideal, y_hat)
        print(f"\tSER: {SER:.3e}")
        alphabets = [alphabet]
        SERs = [SER]
    
    with open(path, 'a') as file:
        if multi_mag and multi_phase:    
            file.write(f"lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB}dB --> mag ER:{mag_ER:.10e}, phase ER:{phase_ER:.10e}, SER: {SER:.10e}")
        else:
            file.write(f"lr={lr}, L_link={L_link*1e-3:.0f}km, alpha={alpha}, SNR={SNR_dB}dB --> SER:{SER:.10e}\n")
    return alphabets, SERs

def save_fig_summary(y, y_hat, multi_mag, multi_phase, alphabets, folder_path, lr, L_link, alpha, SNR_dB,):
        if multi_mag and multi_phase:
            fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15,9))
            plot_constellation(ax1, y_hat, alphabets[2])
            plot_histogram(ax2, y[:,:,1::2].flatten(), y_hat[:,0,:].flatten(), alphabets[0], "Magnitude")
            plot_histogram(ax3, y[:,:,0::2].flatten(), y_hat[:,1,:].flatten(), alphabets[1], "Phase")
            lr_str = f"{lr:}".replace('.', 'p')
            alpha_str = f"{alpha:.1f}".replace('.', 'p')
            fig.savefig(f"{folder_path}/lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB.png")
            plt.close()
            return
        if multi_mag:
            fig, ax1 = plt.subplots(figsize=(5,9))
            plot_histogram(ax1, y[:,:,1::2].flatten(), y_hat[:,0,:].flatten(), alphabets[0], "Magnitude")
            lr_str = f"{lr:}".replace('.', 'p')
            alpha_str = f"{alpha:.1f}".replace('.', 'p')
            fig.savefig(f"{folder_path}/lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB.png")
            plt.close()
            return
        if multi_phase:
            fig, ax1 = plt.subplots(figsize=(5,9))
            plot_histogram(ax1, y[:,:,0::2].flatten(), y_hat[:,0,:].flatten(), alphabets[0], "Phase")
            lr_str = f"{lr:}".replace('.', 'p')
            alpha_str = f"{alpha:.1f}".replace('.', 'p')
            fig.savefig(f"{folder_path}/lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB.png")
            plt.close()
            return

def plot_constellation(ax, y_hat, alphabet):
    ax.set_title("Constellation diagram")
    y_hat_comp = hlp.mag_phase_2_complex(y_hat)
    ax.scatter(np.real(y_hat_comp), np.imag(y_hat_comp), c='b', alpha=0.1, label='CNN out')
    ax.scatter(np.real(alphabet), np.imag(alphabet), c='r', label='ideal')
    ax.legend(loc='upper right')
    ax.grid()

def plot_histogram(ax, y, y_hat, alphabet, name):
    ax.set_title(name)
    for val in alphabet:
        line = ax.axvline(x=val, color='red', linestyle='--')
    _, _, hist1 = ax.hist(y, 200, alpha=0.5, density=True)
    _, _, hist2 = ax.hist(y_hat, 200, alpha=0.5, density=True)
    ax.legend([line, hist1[0], hist2[0]],['ideal', 'DD out', 'CNN out'], loc='upper right')
    ax.grid()
