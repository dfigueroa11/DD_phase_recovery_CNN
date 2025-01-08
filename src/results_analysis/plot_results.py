import numpy as np

def read_file_data(file_path: str, delimiter: str=','):
    with open(file_path, 'r') as file:
        headers = file.readline().lstrip('# ').strip().split(delimiter)
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=1)
    return headers, data

def read_data(file_paths: str, delimiter: str=','):
    headers = []
    data = []
    for file_path in file_paths:
        h, d = read_file_data(file_path, delimiter)
        headers.append(h)
        data.append(d)
    return headers, data

def select_columns_data(headers, data: np.ndarray, column_selection):
    idx = [headers.index(cs) for cs in column_selection]
    return data[:,idx]

def select_data(headers, data, x_data_selection, y_data_selection, slice_selection):
    x_data = []
    y_data = []
    slice_data = []
    for h, d, x, y, s in zip(headers, data, x_data_selection, y_data_selection, slice_selection):
        x_data.append(select_columns_data(h, d, x))
        y_data.append(select_columns_data(h, d, y))
        slice_data.append(select_columns_data(h, d, s))
    return x_data, y_data, slice_data

def take_x_y_slice(x_data, y_data, slice_data, slice_values):
    x_data_out = []
    y_data_out = []
    for x, y, s, s_val in zip(x_data, y_data, slice_data, slice_values):
        x_data_out.append(x[np.all(s == s_val,-1)])
        y_data_out.append(y[np.all(s == s_val,-1)])
    return x_data_out, y_data_out

def plot_data(file_paths, x_data_selection, y_data_selection, slice_selection, slice_values):
    headers, data = read_data(file_paths)
    x_data, y_data, slice_data = select_data(headers, data, x_data_selection, y_data_selection, slice_selection)
    x_data, y_data = take_x_y_slice(x_data, y_data, slice_data, slice_values)

    return 0 




mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
file_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt" for mod_format in mod_formats]
x_data_selection = [["MI", "SER"] for mod_format in mod_formats]
y_data_selection = [["L_link_km"] for mod_format in mod_formats]
slice_selection = [["SNR","L_link_km"] for mod_format in mod_formats]
slice_values = [[1,6] for mod_format in mod_formats]


plot_data(file_paths, x_data_selection, y_data_selection, slice_selection, slice_values)