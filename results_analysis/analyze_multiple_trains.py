import numpy as np
import re
import os
import shutil

def create_results_folder(path,n_copy):
    try:
        real_path = f"{path}_{n_copy}" if n_copy > 0 else path
        os.makedirs(real_path)
        return real_path
    except Exception as e:
        n_copy += 1
        print(f"directory '{path}' already exist, try to create '{path}_{n_copy}'")
        return create_results_folder(path,n_copy)

def read_all_data(num_folders):
    all_data = []
    for i in range(num_folders):
        all_data.append(np.loadtxt(f'{path}/{folder}_{i}/SER_results.txt', delimiter=",", skiprows=1))
    return np.array(all_data)

def find_replace_non_convergence(all_data, threshold):
    for j in range(4, all_data.shape[-1]):
        all_data[:,:,j] = np.where(all_data[:,:,-1]<threshold, np.nan, all_data[:,:,j])
    return all_data

def pick_max_min_mean_data(all_data):
    min_data, min_idx = np.nanmin(all_data, axis=0), np.nanargmin(all_data,axis=0)
    max_data, max_idx = np.nanmax(all_data, axis=0), np.nanargmax(all_data,axis=0)
    mean_data = np.nanmean(all_data, axis=0)
    return np.stack(np.stack([min_data, mean_data, max_data])), np.stack([min_idx, max_idx])

def find_convergence_rate(all_data):
    return 1-np.mean(np.isnan(all_data[:,:,-1]),axis=0)

def write_ser_txt(data, conv_rate, path_source, path_destination):
    with open(f'{path_source}/SER_results.txt', 'r') as source_file:
        first_line = source_file.readline()
    with open(f"{path_destination}/SER_results.txt", 'w') as file:
        file.write(first_line+",conv_rate")
        for i in range(data.shape[1]):
            line = f"{data[0,i,0]},{data[0,i,1]},{data[0,i,2]},{data[0,i,3]}"
            for j in range(4, data.shape[-1]):
                line += f",[{data[0,i,j]:.10e},{data[1,i,j]:.10e},{data[2,i,j]:.10e}]"
            line += f",{conv_rate[i]}"
            file.write(f"{line}\n")

def save_best_images(data, idx, result_path, folder):
    for i, data_line in zip(idx, data):
        lr_str = f"{data_line[0]:}".replace('.', 'p')
        alpha_str = f"{data_line[2]:.1f}".replace('.', 'p')
        source_path = f"{path}/{folder}_{i}/lr{lr_str}_Llink{data_line[1]:.0f}km_alpha{alpha_str}_{data_line[3]:.0f}dB.png"
        dest_path = f"{result_path}/"
        try:
            shutil.copy(source_path, dest_path)
        except FileNotFoundError:
            pass


path = "/Users/diegofigueroa/Desktop/results"
path_best = f"{path}_best"

folders = ["PAM2_sym"]

all_data = []

for folder in folders:
    all_data = read_all_data(num_folders=5)
    all_data = find_replace_non_convergence(all_data, threshold=1e-4)
    data, idx = pick_max_min_mean_data(all_data)
    conv_rate = find_convergence_rate(all_data)
    result_path = create_results_folder(f"{path_best}/{folder}",0)
    write_ser_txt(data, conv_rate, f"{path}/{folder}_{0}", result_path)
    # # save_best_images(best_ser_data, best_idx, result_path, folder)
