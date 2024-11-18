import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D  # Import for custom legend entries

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def plot_complexity_sweep_mod_format(data, mod_format, n_layers, L_link_list):
    fig, axs = plt.subplots(len(n_layers), len(L_link_list), figsize=(15,9))
    fig.suptitle(f"rate vs complexity for {mod_format} at 9 dB")
    ax: Axes
    ylims = (0, np.ceil(data[:,:,:,:,-3].max())*1.1)
    for i, n_layer in enumerate(n_layers):
        axs[i,0].set_ylabel("rate [bpcu]")
        for j, L_link in enumerate(L_link_list):
            ax = axs[i, j]
            ax.set_title(f"{n_layer} layers -- {L_link} km")
            for k in range(data.shape[2]):
                ax.plot(data[i,j,k,:,-2],data[i,j,k,:,-3], 'o')
            ax.grid(which="both")
            ax.set_ylim(ylims)
            ax.set_xscale('log')
            axs[len(n_layers)-1,j].set_xlabel("complexity in #mult")
    
    legend_handles = []
    for i in range(data.shape[2]):
        legend_handles.append(Line2D([0], [0], linestyle='', marker='o', color=f"C{i}", label=f"structure {i}"))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize='large')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return fig

mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
n_layers = [2,3]
test_num = 3
path = f"/Users/diegofigueroa/Desktop/results_comp_sweep{test_num}"
data = []
for n_layer in n_layers:
    for mod_format in mod_formats:
        data.append(read_data(f"{path}/{mod_format}/results_L={n_layer}.txt"))
data = np.array(data)
L_link_list = np.unique(data[0,:,0])
n_structure_geom = np.unique(data[0,:,-1])
data = data.reshape((len(n_layers), len(mod_formats), len(L_link_list), len(n_structure_geom),-1,data.shape[-1]))
pdf = PdfPages(f"complexity_sweep{test_num}.pdf")
for i, mod_format in enumerate(mod_formats):
    plot_complexity_sweep_mod_format(data[:,i], mod_format, n_layers, L_link_list)
    pdf.savefig()
    plt.close()
pdf.close()
