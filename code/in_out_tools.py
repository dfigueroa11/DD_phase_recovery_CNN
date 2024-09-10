import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import argparse


import help_functions as hlp

def create_folder(path,n_copy):
    '''Creates the folder specified by path, if it already exists append a number to the path and creates the folder'''
    try:
        real_path = f"{path}_{n_copy}"
        os.makedirs(real_path)
        return real_path
    except Exception as e:
        n_copy += 1
        print(f"directory '{path}' already exist, try to create '{path}_{n_copy}'")
        return create_folder(path,n_copy)

def init_progress_file(path, multi_mag, multi_phase):
    ''' creates the file to save the progress, and writes the first row with the variable names

    Arguments:
    path:           path of the file to save the results
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    '''
    with open(path, 'a') as file:
        if multi_mag and multi_phase:
            file.write(f"Batch_size,progress,lr,loss,mag_ER,phase_ER,SER,MI\n")
        else:
            file.write(f"Batch_size,progress,lr,loss,SER,MI\n")
    
def save_progress(path, multi_mag, multi_phase, batch_size, progress, lr, loss, SERs, MI):
    ''' save the progress

    Arguments:
    path:           path of the file to save the results
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    batch_size:     int
    progress:       progress of the current epoch (float)
    lr:             learning rate
    loss:           float
    SERs:           [mag ER, phase, ER, SER] or [SER] depending on multi_mag, multi_phase
    MI:             Mutual information float
    '''
    with open(path, 'a') as file:
        if multi_mag and multi_phase:
            file.write(f"{batch_size},{progress:.5},{lr[0]:.5e},{loss:.5e},{SERs[0]:.5e},{SERs[1]:.5e},{SERs[2]:.5e},{MI:.5e}\n")
        else:
            file.write(f"{batch_size},{progress:.5},{lr[0]:.5e},{loss:.5e},{SERs[0]:.5e},{MI:.5e}\n")
    return 

def print_progress(multi_mag, multi_phase, batch_size, progress, lr, loss, SERs, MI):
    '''Print the training progress

    Arguments:
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    batch_size:     int
    progress:       progress of the current epoch (float)
    lr:             learning rate
    loss:           float
    SERs:           [mag ER, phase, ER, SER] or [SER] depending on multi_mag, multi_phase
    MI:             Mutual information float
    '''
    if multi_mag and multi_phase:
        print(f"\tBatch size: {batch_size:>4}  progress: {progress:>6.1%}   lr: {lr[0]:>8.2e}   loss: {loss:>9.3e}   "+
              f"mag ER: {SERs[0]:>9.3e}   phase ER: {SERs[1]:>9.3e}   SER: {SERs[2]:>9.3e}   MI: {MI:>6.3f}", end='\r')
    else:
        print(f"\tBatch size: {batch_size:>4}  progress: {progress:>6.1%}   lr: {lr[0]:>8.2e}   loss: {loss:>9.3e}   SER: {SERs[0]:>9.3e}   MI: {MI:>6.3f}", end='\r')

def init_summary_file(path):
    ''' creates the file to save the results, and writes the first row with the variable names

    Arguments:
    path:           path of the file to save the results
    '''
    with open(path, 'a') as file:
        file.write("lr,L_link_km,alpha,SNR_dB,(mag_ER,phase_ER),SER,MI\n")
        
def print_save_summary(path, multi_mag, multi_phase, lr, L_link, alpha, SNR_dB, SERs, MI):
    ''' Print and saves the summary of the training process

    Arguments:
    path:           path of the file to save the results
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    lr:             learning rate
    L_link:         length of the SMF in meters (float) use if the channel presents CD
    alpha:          roll off factor (float between 0 and 1)
    SNR_dB:         SNR in dB used for the simulation (float)
    SERs:           [mag ER, phase, ER, SER] or [SER] depending on multi_mag, multi_phase
    MI:             Mutual information float
    '''
    with open(path, 'a') as file:
        if multi_mag and multi_phase:
            file.write(f"{lr},{L_link*1e-3:.0f},{alpha},{SNR_dB},{SERs[0]:.10e},{SERs[1]:.10e},{SERs[2]:.10e},{MI:.10e}\n")
            print(f"\tmag ER: {SERs[0]:>9.3e}   phase ER: {SERs[1]:>9.3e}   SER: {SERs[2]:>9.3e}   MI: {MI:>6.3f}")
        else:
            file.write(f"{lr},{L_link*1e-3:.0f},{alpha},{SNR_dB},{SERs[0]:.10e},{MI:.10e}\n")
            print(f"\tSER: {SERs[0]:>9.3e}   MI: {MI:>6.3f}")
    
def save_fig_summary(y, y_hat, multi_mag, multi_phase, alphabets, folder_path, lr, L_link, alpha, SNR_dB):
    '''Save the figure with the resume
    
    Arguments:
    y_ideal:        Tensor containing the ideal magnitudes and phase differences (shape (batch_size, 2/1, N_sym) depending on multi_mag, multi_phase)
    y_hat:          output of the CNN (same shape as y_ideal)
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    alphabets:      [mag alphabet, phase alphabet, symbol alphabet] or [alphabet] depending on multi_mag, multi_phase
    folder_path:    path of the folder to save the image
    lr:             learning rate
    L_link:         length of the SMF in meters (float) use if the channel presents CD
    alpha:          roll off factor (float between 0 and 1)
    SNR_dB:         SNR in dB used for the simulation (float)
    '''
    if multi_mag and multi_phase:
        fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(15,9))
        plot_histogram(ax2, y[:,:,1::2].flatten(), y_hat[:,0,:].flatten(), alphabets[0], "Magnitude")
        plot_histogram(ax3, y[:,:,0::2].flatten(), y_hat[:,1,:].flatten(), alphabets[1], "Phase")
        plot_constellation(ax1, y_hat, alphabets[2])
    else :
        fig, ax1 = plt.subplots(figsize=(5,9))
        name = "Magnitude" if multi_mag else "phase"
        plot_histogram(ax1, y[:,:,1::2].flatten(), y_hat[:,0,:].flatten(), alphabets[0], name)

    fig.savefig(f"{folder_path}/{make_file_name(lr, L_link, alpha, SNR_dB)}.png")
    plt.close()

def plot_constellation(ax, y_hat, alphabet):
    '''Plot the constellation diagram
    
    Arguments:
    ax:         Matplotlib.axes.Axes to do the plot
    y_hat:      output of the CNN (same shape as y_ideal)
    alphabet:   contains the ideal output for reference
    '''
    ax.set_title("Constellation diagram")
    y_hat_comp = hlp.mag_phase_2_complex(y_hat)
    ax.scatter(np.real(y_hat_comp), np.imag(y_hat_comp), c='b', alpha=0.1, label='CNN out')
    ax.scatter(np.real(alphabet), np.imag(alphabet), c='r', label='ideal')
    ax.legend(loc='upper right')
    ax.grid()

def plot_histogram(ax, y, y_hat, alphabet, name):
    '''Plot the constellation diagram
    
    Arguments:
    ax:         Matplotlib.axes.Axes to do the plot
    y:          output of the DD channel
    y_hat:      output of the CNN (same shape as y_ideal)
    alphabet:   contains the ideal output for reference
    name:       name for the plot, (Magnitude or Phase)
    '''
    ax.set_title(name)
    for val in alphabet:
        line = ax.axvline(x=val, color='red', linestyle='--')
    _, _, hist1 = ax.hist(y, 200, alpha=0.5, density=True)
    _, _, hist2 = ax.hist(y_hat, 200, alpha=0.5, density=True)
    ax.legend([line, hist1[0], hist2[0]],['ideal', 'DD out', 'CNN out'], loc='upper right')
    ax.grid()

def process_args():
    formatter = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=60)
    parser = argparse.ArgumentParser(
        description="CNN equalizer for Direct Detection channels with phase recovery",
        formatter_class=formatter)
    parser.add_argument(
        "--mod_format",
        "-m",
        type=str,
        help="modulation format",
        choices=["ASK", "PAM", "DDQAM", "QAM"],
        default="ASK")
    parser.add_argument(
        "--order",
        "-o",
        type=int,
        help="modulation format order",
        default=4)
    return parser.parse_args()

def make_file_name(lr, L_link, alpha, SNR_dB):
    lr_str = f"{lr:}".replace('.', 'p')
    alpha_str = f"{alpha:.1f}".replace('.', 'p')
    return f"lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB"
