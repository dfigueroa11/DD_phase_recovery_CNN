import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler

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

def plot_data(ax: Axes, file_paths: list[str], x_data_selection: list[list[str]], y_data_selection: list[list[str]],
              slice_selection: list[list[str]], slice_values: list[list[float]], labels_list: list[list[str]]):
    headers, data = read_data(file_paths)
    x_data, y_data, slice_data = select_data(headers, data, x_data_selection, y_data_selection, slice_selection)
    x_data, y_data = take_x_y_slice(x_data, y_data, slice_data, slice_values)
    for x, y, lbl in zip(x_data, y_data, labels_list):
        ax.plot(x, y, label=lbl)




mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
file_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt" for mod_format in mod_formats]
y_data_selection = [["MI", "SER"] for mod_format in mod_formats]
x_data_selection = [["SNR"] for mod_format in mod_formats]
slice_selection = [["L_link_km"] for mod_format in mod_formats]
slice_values = [6 for mod_format in mod_formats]
labels_list = [[f"{mod_format} MI", f"{mod_format} SER"] for mod_format in mod_formats]

# use the cycler to set the properties of each graph
color_cycler = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
mpl.rcParams['axes.prop_cycle'] = cycler(color=color_cycler, ls=['--',]*len(color_cycler))

fig, ax = plt.subplots()
plot_data(ax,file_paths, x_data_selection, y_data_selection, slice_selection, slice_values, labels_list)
ax.legend()
plt.show()