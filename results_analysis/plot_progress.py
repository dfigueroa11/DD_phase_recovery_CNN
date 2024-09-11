import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

def read_progress_file(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)

def plot_progress(ax:Axes, data, data_label=None, reference_lines=None, reference_labels=None):
    ax.plot(data, label=data_label)
    if reference_lines is not None and reference_labels is not None:
        for i, (ref_line, ref_label) in enumerate(zip(reference_lines, reference_labels)):
            ax.axvline(ref_line, label=ref_label, linestyle='--', color=f"C{9-i}")
    ax.set_xlabel("progress")
    ax.grid()

def plot_all_progress(progress_data, title):
    batch_size_change = np.nonzero(np.diff(np.roll(progress_data[:,0],1)))[0]
    batch_size_change_labels = [f"batch size: {progress_data[bsc,0]:.0f}" for bsc in batch_size_change]
    fig, axs = plt.subplots(2,2, figsize=(10,6))
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(pad=2, rect=(0.04,0,0.96,1.05))
    plot_progress(axs[0,0], progress_data[:,2], reference_lines=batch_size_change, reference_labels=batch_size_change_labels)
    axs[0,0].set_ylabel("Learning rate")
    axs[0,0].legend()
    plot_progress(axs[0,1], progress_data[:,3], reference_lines=batch_size_change, reference_labels=batch_size_change_labels)
    axs[0,1].set_ylabel("loss")
    axs[0,1].legend()
    labels = ["mag ER","phase ER", "SER"] if progress_data.shape[-1]==8 else None
    plot_progress(axs[1,0], progress_data[:,4:-1], data_label=labels, reference_lines=batch_size_change, reference_labels=batch_size_change_labels)
    axs[1,0].set_ylabel("SER")
    axs[1,0].set_yscale("log")
    axs[1,0].legend()
    plot_progress(axs[1,1], progress_data[:,-1], reference_lines=batch_size_change, reference_labels=batch_size_change_labels)
    axs[1,1].set_ylabel("Mutual information [bpcu]")
    axs[1,1].legend()
    plt.show()



if __name__=="__main__":
    L_link_steps = np.arange(0,35,6)
    SNR_dB_steps = np.arange(-5, 12, 2)
    mod_format = "PAM2"
    for L_link in L_link_steps:
        for SNR in SNR_dB_steps:
            path = f"/Users/diegofigueroa/Desktop/{mod_format}_sym_0/progress_lr0p004_Llink{L_link}km_alpha0p0_{SNR}dB.txt"
            progress_data = read_progress_file(path)
            plot_all_progress(progress_data, f"{mod_format} -- Link length: {L_link} km -- SNR: {SNR} dB")
        