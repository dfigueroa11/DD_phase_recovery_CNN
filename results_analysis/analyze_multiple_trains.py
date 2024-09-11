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

def pick_best_ser_data(num_folders):
    all_data = read_all_data(num_folders)
    return np.argmin(all_data[:,:,-1], axis=0), np.min(all_data, axis=0)

def pick_max_min_mean_ser_data(num_folders):
    all_data = read_all_data(num_folders)
    return np.argmin(all_data[:,:,-1], axis=0), np.min(all_data, axis=0)

def write_ser_txt(data, path):
    with open(f"{path}/SER_results.txt", 'w') as file:
        for data_line in data:
            file.write(f"lr={data_line[0]}, L_link={data_line[1]:.0f}km, alpha={data_line[2]}, SNR={data_line[3]}dB --> SER:{data_line[4]:.10e}\n")

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


path = "/Users/diegofigueroa/Desktop/"
path_best = f"{path}_best"

folders = ["PAM2_sym"]

all_data = []

for folder in folders:
    best_idx, best_ser_data = pick_best_ser_data(5)
    result_path = create_results_folder(f"{path_best}/{folder}",0)
    write_ser_txt(best_ser_data, result_path)
    save_best_images(best_ser_data, best_idx, result_path, folder)
