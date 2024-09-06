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
        real_path = f"{path}_{n_copy}" if n_copy > 0 else path
        os.makedirs(real_path)
        return real_path
    except Exception as e:
        n_copy += 1
        print(f"directory '{path}' already exist, try to create '{path}_{n_copy}'")
        return create_folder(path,n_copy)

def print_progress(y_ideal, y_hat, batch_size, progress, loss, multi_mag, multi_phase):
    '''Print the training progress

    Arguments:
    y_ideal:        Tensor containing the ideal magnitudes and phase differences (shape (batch_size, 2/1, N_sym) depending on multi_mag, multi_phase)
    y_hat:          output of the CNN (same shape as y_ideal)
    batch_size:     int
    progress:       progress of the current epoch (float)
    loss:           float
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    '''
    if multi_mag and multi_phase:
        _, mag_ER = hlp.decode_and_ER(y_ideal[:,0,:], y_hat[:,0,:])
        _, phase_ER = hlp.decode_and_ER(y_ideal[:,1,:], y_hat[:,1,:])
        _, SER = hlp.decode_and_ER_mag_phase(y_ideal, y_hat)
        SERs = [mag_ER, phase_ER, SER]
        print(f"\tBatch size {batch_size:_}\tprogress {progress:>6.1%}\tloss: {loss:.3e}\tmag ER: {mag_ER:.3e}\tphase ER: {phase_ER:.3e}\tSER: {SER:.3e}",end='\r')
    else:
        _, SER = hlp.decode_and_ER(y_ideal, y_hat)
        print(f"\tBatch size {batch_size:_}\tprogress {progress:>6.1%}\tloss: {loss:.3e}\tSER: {SER:.3e}",end='\r')
        SERs = [SER]
    return SERs

def init_summary_file(path, multi_mag, multi_phase):
    ''' creates the file to save the results, and writes the first row with the variable names

    Arguments:
    path:           path of the file to save the results
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    '''
    with open(path, 'a') as file:
        if multi_mag and multi_phase:
            file.write("lr,L_link_km,alpha,SNR_dB,mag_ER,phase_ER,SER")
        else:
            file.write("lr,L_link_km,alpha,SNR_dB,SER")

def print_save_summary(y_ideal, y_hat, multi_mag, multi_phase, lr, L_link, alpha, SNR_dB, path):
    ''' Print and saves the summary of the training process

    Arguments:
    y_ideal:        Tensor containing the ideal magnitudes and phase differences (shape (batch_size, 2/1, N_sym) depending on multi_mag, multi_phase)
    y_hat:          output of the CNN (same shape as y_ideal)
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    lr:             learning rate
    L_link:         length of the SMF in meters (float) use if the channel presents CD
    alpha:          roll off factor (float between 0 and 1)
    SNR_dB:         SNR in dB used for the simulation (float)
    path:           path of the file to save the results

    Returns:
    alphabets:      [mag alphabet, phase alphabet, symbol alphabet] or [alphabet] depending on multi_mag, multi_phase
    SER:            [mag ER, phase, ER, SER] or [SER] depending on multi_mag, multi_phase
    '''
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
            file.write(f"{lr},{L_link*1e-3:.0f},{alpha},{SNR_dB},{mag_ER:.10e},{phase_ER:.10e},{SER:.10e}")
        else:
            file.write(f"{lr},{L_link*1e-3:.0f},{alpha},{SNR_dB},{SER:.10e}\n")
    return alphabets, SERs

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

    lr_str = f"{lr:}".replace('.', 'p')
    alpha_str = f"{alpha:.1f}".replace('.', 'p')
    fig.savefig(f"{folder_path}/lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB.png")
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
