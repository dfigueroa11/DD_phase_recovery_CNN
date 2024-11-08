import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D  # Import for custom legend entries
from functools import reduce
import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

def plot_complexity(mod_format, n_layers, n_str_design, complexities):
    fig, axs = plt.subplots(len(n_layers), len(complexities), figsize=(15,9))
    fig.suptitle(f"Performance for different structures -- {mod_format}, 9 dB")
    for i, n_layer in enumerate(n_layers):
        for j, complexity in enumerate(complexities):
            ## read data
            file_path = f"/Users/diegofigueroa/Desktop/results/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            path_plabst = f"/Users/diegofigueroa/Desktop/KIT/HiWi/results_cnn_vs_Plabst_2/txt/Plabst_results/{mod_format}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            L_link_list = data[:,0,0]
            data_plabst = (read_data(path_plabst))
            data_plabst_MI = data_plabst[:,[fc.MI_Plabst_column_ppr, fc.SDD_MI_Plabst_column_ppr]]
            idx = np.nonzero(np.isin(data_plabst[:,fc.L_link_Plabst_column_ppr],L_link_list) & (data_plabst[:,fc.SNR_Plabst_column_ppr] == 9))
            ref_MI = data_plabst_MI[idx]
            ## plot
            ax: Axes
            ax = axs[i][j]
            for k, L_link in enumerate(L_link_list):
                ax.set_title(f"L={n_layer} -- C={complexity}")
                # ax.plot(data[i,:,-2],':', alpha=0.7, c=f"C{i}")
                ax.plot(data[k,:,-2],'o', c=f"C{k}")
                ax.axhline(ref_MI[k,0], c=f"C{k}", label=f"{L_link:2>.0f} km")
            ax.set_xlabel("Structure number")
            ax.set_ylabel("Rate [bpcu]")
            ax.legend()
            ax.set_xlim([0,data.shape[1]])
            ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-1)))
            ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-2)), minor=True)
            ax.grid()
            ax.grid(which='minor', linestyle=":")
    plt.show()

def plot_performance_complexity(mod_format, n_layers, n_str_design, complexities):
    fig = plt.figure(figsize=(15,9))
    fig.suptitle(f"rate/complexity for different structures -- {mod_format}, 9 dB")
    outer_grid = gridspec.GridSpec(len(n_layers), len(complexities), wspace=0.2, hspace=0.3)
    for i, n_layer in enumerate(n_layers):
        for j, complexity in enumerate(complexities):
            ## read data
            file_path = f"/Users/diegofigueroa/Desktop/results/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            L_link_list = data[:,0,0]
            data_MI_vs_comp = data[:,:,-2]/data[:,:,-1]
            ## plot
            inner_grid = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=outer_grid[len(complexities)*i+j], hspace=0)
            ax1 = plt.Subplot(fig, inner_grid[0])
            ax2 = plt.Subplot(fig, inner_grid[1], sharex=ax1)
            ax3 = plt.Subplot(fig, inner_grid[2], sharex=ax1)

            for k, ax in enumerate([ax1,ax2,ax3]):
                # ax.set_title(f"L={n_layer} -- C={complexity}")
                ax.plot(data_MI_vs_comp[k,:],'o:', c=f"C{k}")
                ax.set_xlim([0,data.shape[1]])
                ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-1)))
                ax.set_xticks(np.arange(0,data.shape[1]+1,n_str_design**(n_layer-2)), minor=True)
                ax.set_ylim([0,data_MI_vs_comp[k,:].max()*1.1])
                ax.set_yticks([0,data_MI_vs_comp[k,:].max()])
                ax.set_yticklabels(["0", "max"])
                ax.grid()
                ax.grid(which='minor', linestyle=":")
            
            ax1.set_title(f"L={n_layer} -- C={complexity}")
            ax3.set_xlabel("Structure number")
            ax2.set_ylabel("Rate/complexity u.a.")
            plt.setp(ax1.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)
            fig.add_subplot(ax1)
            fig.add_subplot(ax2)
            fig.add_subplot(ax3)
    
    legend_handles = []
    for i, L_link in enumerate(L_link_list):
        legend_handles.append(Line2D([0], [0], color=f"C{i}", label=f"{L_link:2>.0f} km"))
    fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize='large')
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    return fig
    

def find_idx_common_top(data, num_top=1):
    threshold = 0.5
    n = 1
    idx = find_idx_in_top_x(data, threshold)
    while len(idx) != num_top:
        threshold += 0.5**n if len(idx)>num_top else -0.5**n
        idx = find_idx_in_top_x(data, threshold)
        n += 1
    return idx, (data[:,idx]/data.max(axis=1, keepdims=True)).max(axis=0), (data[:,idx]/data.max(axis=1, keepdims=True)).min(axis=0)

def find_idx_in_top_x(data: np.ndarray, threshold):
    idx_sets = []
    for i in range(data.shape[0]):
        idx_sets.append(np.where(data[i,:] >= data[i,:].max()*threshold)[0])
    return reduce(np.intersect1d, idx_sets)

def print_structure(str_list,k,i,j,idx):
    print(str_list[k][i][j][idx])

def gen_structure_geometry(structures):
    strides = structures[0,3,:]
    groups = structures[0,4,:]
    CNN_ch_in = int(structures[0,0,0])
    CNN_ch_out = int(structures[0,1,-1])
    ch_out_ker_sz_ratios = np.mean(structures[:,1,:-1]/structures[:,2,:-1], axis=0)
    complexity_profile = structures[:,1,:]*structures[:,2,:]*structures[:,0,:]*np.prod(strides)/(np.cumprod(strides)*groups)
    complexity_profile = complexity_profile/np.sum(complexity_profile, axis=-1, keepdims=True)
    complexity_profile.mean(axis=0)/complexity_profile.mean(axis=0).sum()
    return complexity_profile, ch_out_ker_sz_ratios, CNN_ch_in, CNN_ch_out, strides, groups

def save_structure_geometry(structure_geometry):
    pass


mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
n_layers = [2,3]
complexities = [200,500,1000]
L_link_list = [0,12,30]
n_str_design = 4
# for mod_format in mod_formats:
#     plot_complexity(mod_format, n_layers, n_str_design, complexities)
# pdf = PdfPages(f"big_comparison.pdf")
# for mod_format in mod_formats:
#     plot_performance_complexity(mod_format, n_layers, n_str_design, complexities)
#     pdf.savefig()
# pdf.close()

MI_vs_comp_list = []
for j, n_layer in enumerate(n_layers):
    MI_vs_comp_layer_n = []
    for i, mod_format in enumerate(mod_formats):
        for k, complexity in enumerate(complexities):
            ## read data
            file_path = f"/Users/diegofigueroa/Desktop/results/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            MI_vs_comp_layer_n.append(data[:,:,-2]/data[:,:,-1])
    MI_vs_comp_list.append(np.reshape(np.array(MI_vs_comp_layer_n),(len(mod_formats),len(complexities),len(L_link_list),-1)))

struc_idx = []
for k, n_layer in enumerate(n_layers):
    idx_aux = []
    for i, mod_format in enumerate(mod_formats):
        for j, L_link in enumerate(L_link_list):
            idx_s, th_max_s, th_min_s = find_idx_common_top(MI_vs_comp_list[k][i,:,j,:],1)
            idx_aux.append(idx_s)
            print(f"{n_layer} layers, {mod_format}, {L_link} km:")
            for idx, th_max, th_min in zip(idx_s, th_max_s, th_min_s):
                print(f"\tstructure {idx:>3}, relative performance between {th_min:.2f} and {th_max:.2f}")
    struc_idx.append(np.array(idx_aux).reshape(len(mod_formats),len(L_link_list),-1))

structures = []
for j, n_layer in enumerate(n_layers):
    structures_layer = []
    for i, mod_format in enumerate(mod_formats):
        for k, complexity in enumerate(complexities):
            file_path = f"/Users/diegofigueroa/Desktop/results/{mod_format}/structures_L={n_layer}_C={complexity}.npy"
            structures_layer.append(np.load(file_path))
    structures.append(np.array(structures_layer).reshape(len(mod_formats),len(complexities),-1,5,n_layer))

for j, n_layer in enumerate(n_layers):
    for i, mod_format in enumerate(mod_formats):
        for k, L_link in enumerate(L_link_list):
            structure_geometry = gen_structure_geometry(structures[j][i,:,struc_idx[j][i,k,0],:,:])
            save_structure_geometry(structure_geometry)

