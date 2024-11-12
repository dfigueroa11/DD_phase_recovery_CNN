import numpy as np
from functools import reduce
import pickle

import file_constants as fc

def read_data(path_file):
    return np.loadtxt(path_file, delimiter=",", skiprows=1)

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
    prod_layer_ch_out_ker_sz = structures[:,1,:-1]*structures[:,2,:-1]
    exp = np.mean(np.log(structures[:,1,:-1])/np.log(prod_layer_ch_out_ker_sz), axis=0)
    complexity_profile = structures[:,1,:]*structures[:,2,:]*structures[:,0,:]*np.prod(strides)/(np.cumprod(strides)*groups)
    complexity_profile = complexity_profile/np.sum(complexity_profile, axis=-1, keepdims=True)
    complexity_profile = complexity_profile.mean(axis=0)/complexity_profile.mean(axis=0).sum()
    return complexity_profile, exp, CNN_ch_in, CNN_ch_out, strides, groups

def save_structure_geometry(path, structure_geometry, n_layer, mod_format, L_link, n):
    path = f"{path}/L_{n_layer}_M_{mod_format}_Ll_{L_link}km_{n}.pkl"
    data = {"complexity_profile": structure_geometry[0],
            "exp_chs": structure_geometry[1],
            "CNN_ch_in": structure_geometry[2],
            "CNN_ch_out": structure_geometry[3],
            "strides": structure_geometry[4],
            "groups": structure_geometry[5]}
    with open(path, "wb") as file:
        pickle.dump(data, file)

def gen_structure_geometry2(structures):
    strides = structures[0,3,:]
    groups = structures[0,4,:]
    CNN_ch_in = int(structures[0,0,0])
    CNN_ch_out = int(structures[0,1,-1])
    ker_sz_s = structures[:,2,:-1].mean(axis=0)
    complexity_profile = structures[:,1,:]*structures[:,2,:]*structures[:,0,:]*np.prod(strides)/(np.cumprod(strides)*groups)
    complexity_profile = complexity_profile/np.sum(complexity_profile, axis=-1, keepdims=True)
    complexity_profile = complexity_profile.mean(axis=0)/complexity_profile.mean(axis=0).sum()
    return complexity_profile, ker_sz_s, CNN_ch_in, CNN_ch_out, strides, groups

def save_structure_geometry2(path, structure_geometry, n_layer, mod_format, L_link, n):
    path = f"{path}/L_{n_layer}_M_{mod_format}_Ll_{L_link}km_{n}.pkl"
    data = {"complexity_profile": structure_geometry[0],
            "ker_sz_s": structure_geometry[1],
            "CNN_ch_in": structure_geometry[2],
            "CNN_ch_out": structure_geometry[3],
            "strides": structure_geometry[4],
            "groups": structure_geometry[5]}
    with open(path, "wb") as file:
        pickle.dump(data, file)

mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
n_layers = [2,3]
complexities = [200,500,1000]
L_link_list = [0,12,30]

MI_vs_comp_list = []
for n_layer in n_layers:
    MI_vs_comp_layer_n = []
    for mod_format in mod_formats:
        for complexity in complexities:
            file_path = f"/Users/diegofigueroa/Desktop/results_fix_comp/{mod_format}/results_L={n_layer}_C={complexity}.txt"
            data = read_data(file_path).reshape(3,-1,6)
            MI_vs_comp_layer_n.append(data[:,:,-2]/data[:,:,-1])
    MI_vs_comp_list.append(np.reshape(np.array(MI_vs_comp_layer_n),(len(mod_formats),len(complexities),len(L_link_list),-1)))

num_best_structures = 3
struc_idx = []
for k, n_layer in enumerate(n_layers):
    idx_aux = []
    for i, mod_format in enumerate(mod_formats):
        for j, L_link in enumerate(L_link_list):
            idx_s, th_max_s, th_min_s = find_idx_common_top(MI_vs_comp_list[k][i,:,j,:], num_best_structures)
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
            file_path = f"/Users/diegofigueroa/Desktop/results_fix_comp/{mod_format}/structures_L={n_layer}_C={complexity}.npy"
            structures_layer.append(np.load(file_path))
    structures.append(np.array(structures_layer).reshape(len(mod_formats),len(complexities),-1,5,n_layer))

for j, n_layer in enumerate(n_layers):
    for i, mod_format in enumerate(mod_formats):
        for k, L_link in enumerate(L_link_list):
            for n in range(struc_idx[j].shape[-1]):
                structure_geometry = gen_structure_geometry2(structures[j][i,:,struc_idx[j][i,k,n],:,:])
                save_structure_geometry2(f"/Users/diegofigueroa/Desktop/CNN_geometries2", structure_geometry, n_layer, mod_format, L_link, n)

