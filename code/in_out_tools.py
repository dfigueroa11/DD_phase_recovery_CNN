import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib import axes 
from matplotlib import gridspec
from matplotlib.lines import Line2D
import argparse


import help_functions as hlp
from DD_system import DD_system
import CNN_equalizer
from performance_metrics import get_alphabets
import data_conversion_tools

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

def init_progress_file(path):
    ''' creates the file to save the progress, and writes the first row with the variable names

    Arguments:
    path:           path of the file to save the results
    multi_mag:      whether the constellation have multiple magnitudes or not
    multi_phase:    whether the constellation have multiple phases or not
    '''
    with open(path, 'a') as file:
        file.write(f"Batch_size,progress,lr,loss,mag_ER,phase_ER,SER,MI\n")

def save_progress(path, batch_size, progress, lr, loss, SERs, MI):
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
        file.write(f"{batch_size},{progress:.5},{lr[0]:.5e},{loss:.5e},{SERs[0]:.5e},{SERs[1]:.5e},{SERs[2]:.5e},{MI:.5e}\n")

def print_progress(batch_size, progress, lr, loss, SERs, MI):
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
    print(f"   Batch size:{batch_size:>4}  prog:{progress:>6.1%}  lr:{lr[0]:>8.2e}  loss:{loss:>9.3e}  "+
          f"mag ER:{SERs[0]:>8.2e}  ph ER:{SERs[1]:>8.2e}  SER:{SERs[2]:>8.2e}  MI:{MI:>4.2f}", end='\r')

def init_summary_file(path):
    ''' creates the file to save the results, and writes the first row with the variable names

    Arguments:
    path:           path of the file to save the results
    '''
    with open(path, 'a') as file:
        file.write("lr,L_link_km,alpha,SNR_dB,mag_ER,phase_ER,SER,MI\n")
        
def print_save_summary(path, lr, L_link, alpha, SNR_dB, SERs, MI):
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
        file.write(f"{lr},{L_link*1e-3:.0f},{alpha},{SNR_dB},{SERs[0]:.10e},{SERs[1]:.10e},{SERs[2]:.10e},{MI:.10e}\n")
        print(f"\tmag ER: {SERs[0]:>9.3e}   phase ER: {SERs[1]:>9.3e}   SER: {SERs[2]:>9.3e}   MI: {MI:>6.3f}")

def save_fig_summary(u, y, u_hat, cnn_out, dd_system: DD_system, train_type, folder_path, lr, L_link, alpha, SNR_dB):
    alphabets = get_alphabets(dd_system, SNR_dB)
    if dd_system.multi_mag_const and dd_system.multi_phase_const:
        fig = plt.figure(figsize=(15,9))
        gs = gridspec.GridSpec(2,3)
        ax1 = fig.add_subplot(gs[:,0])
        ax2 = fig.add_subplot(gs[0,1])
        ax3 = fig.add_subplot(gs[1,1])
        ax4 = fig.add_subplot(gs[0,2])
        ax5 = fig.add_subplot(gs[1,2])
        plot_constellation(ax1, u, u_hat.flatten(), alphabets[2])
        plot_histogram(ax2, ax3, y[:,:,1::2].flatten(), torch.abs(u_hat).flatten(), torch.abs(u), alphabets[0], ["odd sample","Magnitude"])
        plot_histogram(ax4, ax5, y[:,:,0::2].flatten(), torch.angle(u_hat).flatten(), torch.angle(u), alphabets[1], ["even sample","Phase"])
        fig.savefig(f"{folder_path}/{make_file_name(lr, L_link, alpha, SNR_dB)}.png")
        plt.close()
        return
    if dd_system.multi_mag_const:
        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(5,9))
        plot_histogram(ax1, ax2, y[:,:,1::2].flatten(), u_hat.flatten(), u, alphabets[0], ["odd sample","Magnitude"])
        fig.savefig(f"{folder_path}/{make_file_name(lr, L_link, alpha, SNR_dB)}.png")
        plt.close()
        return
    if dd_system.multi_phase_const:
        fig, (ax1,ax2) = plt.subplots(2, 1, figsize=(5,9))
        plot_histogram(ax1, ax2, y[:,:,0::2].flatten(), torch.angle(u_hat).flatten(), torch.angle(u), alphabets[1], ["even sample","Phase"])
        fig.savefig(f"{folder_path}/{make_file_name(lr, L_link, alpha, SNR_dB)}.png")
        plt.close()
        return

def plot_constellation(ax, u, u_hat, alphabet):
    '''Plot the constellation diagram
    
    Arguments:
    ax:         Matplotlib.axes.Axes to do the plot
    y_hat:      output of the CNN (same shape as y_ideal)
    alphabet:   contains the ideal output for reference
    '''
    ax.set_title("Constellation diagram")
    legend_elements = []
    for i, sym in enumerate(alphabet):
        idx = np.flatnonzero(np.isclose(u,sym))
        ax.scatter(np.real(u_hat[idx]), np.imag(u_hat[idx]), alpha=0.01)
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'CNN out given u={np.real(sym):.1f}+{np.imag(sym):.1f}j',
                                      markerfacecolor=f'C{i}', markersize=10))
    ax.scatter(np.real(alphabet), np.imag(alphabet), c='k')
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label='ideal', markerfacecolor='k', markersize=10))
    ax.legend(handles=legend_elements, loc='upper right')
    ax.grid()

def plot_histogram(ax1, ax2, y, u_hat, u, alphabet, names):
    '''Plot the constellation diagram
    
    Arguments:
    ax:         Matplotlib.axes.Axes to do the plot
    y:          output of the DD channel
    y_hat:      output of the CNN (same shape as y_ideal)
    alphabet:   contains the ideal output for reference
    name:       name for the plot, (Magnitude or Phase)
    '''
    ax1.set_title(names[0])
    ax1.hist(y, 200, alpha=0.5, density=True)
    ax1.grid()
    ax2.set_title(names[1])
    elements = []
    legends = []
    for val in alphabet:
        line = ax2.axvline(x=val, color='red', linestyle='--')
        idx = np.flatnonzero(np.isclose(u, val, rtol=1e-3))
        _, _, hist = ax2.hist(u_hat[idx], 200, alpha=0.5, density=True)
        elements.append(hist[0])
        legends.append(f"CNN_out given u={val:.2f}")
    elements.append(line)
    legends.append("ideal")
    ax2.legend(elements,legends, loc='upper right')
    ax2.grid()

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
        default="PAM")
    parser.add_argument(
        "--order",
        "-o",
        type=int,
        help="modulation format order",
        default=2)
    parser.add_argument(
        "--loss_func",
        "-l",
        type=int,
        help="selection of loss function: 0:TRAIN_MSE_U_SYMBOLS, 1:TRAIN_MSE_U_MAG_PHASE, 2:TRAIN_MSE_U_SLDMAG_PHASE, 3:TRAIN_CE_U_SYMBOLS",
        choices=[0, 1, 2, 3],
        default=1)
    return parser.parse_args()
 
def make_file_name(lr, L_link, alpha, SNR_dB):
    lr_str = f"{lr:}".replace('.', 'p')
    alpha_str = f"{alpha:.1f}".replace('.', 'p')
    return f"lr{lr_str}_Llink{L_link*1e-3:.0f}km_alpha{alpha_str}_{SNR_dB}dB"
