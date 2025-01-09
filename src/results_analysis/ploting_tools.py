import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler

def read_file_data(file_path: str, delimiter: str=','):
    '''
    Retruns:
    headers:    List with the headers of the file
    data:       nd.array with the data of the file
    '''
    with open(file_path, 'r') as file:
        headers = file.readline().lstrip('# ').strip().split(delimiter)
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=1)
    return headers, data

def read_data(file_paths: list[str], delimiter: str=','):
    '''
    Retruns:
    headers:    List of lists where each inner list contains the headers of each file
    data:       list of nd.array with the data of each file
    '''
    headers = []
    data = []
    for file_path in file_paths:
        h, d = read_file_data(file_path, delimiter)
        headers.append(h)
        data.append(d)
    return headers, data

def select_columns_data(headers: list[str], data: np.ndarray, column_selection: list[str]):
    '''Selects data columns based on the headers and the specified column selection. Only the columns whose headers match entries in the column selection list are included

    Returns: nd.array containing the selected data
    '''
    idx = [headers.index(cs) for cs in column_selection]
    return data[:,idx]

def select_data(headers: list[list[str]], data: list[np.ndarray], x_data_selection: list[str], y_data_selection: list[list[str]], slice_selection: list[list[str]]):
    ''' From each array in data selects the x, y and slice columns acording the the headers and x_data_selection, y_data_selection, slice_selection

    Returns:
    x_data, y_data, slice_data:     three list of nd.array for x, y and slice data, if slice selection is None slice_data is equal to y_data
    '''
    x_data = []
    y_data = []
    slice_data = []
    if slice_selection is None: slice_selection = y_data_selection
    for h, d, x, y, s in zip(headers, data, x_data_selection, y_data_selection, slice_selection):
        x_data.append(select_columns_data(h, d, [x,]))
        y_data.append(select_columns_data(h, d, y))
        slice_data.append(select_columns_data(h, d, s))
    return x_data, y_data, slice_data

def take_x_y_slice(x_data: list[np.ndarray], y_data: list[np.ndarray], slice_data: list[np.ndarray], slice_values: list[list[float]]):
    ''' Take a slice of the data based on specified slice conditions, where slice_data should match slice_values (e.g., take data where s1 = c1, s2 = c2, etc.)
    
    Retutns:
    x_data_out, y_data_out:     two list of nd.array for x and y where the slice conditions are met
    '''
    x_data_out = []
    y_data_out = []
    for x, y, s, s_val in zip(x_data, y_data, slice_data, slice_values):
        x_data_out.append(x[np.all(s == s_val,-1)])
        y_data_out.append(y[np.all(s == s_val,-1)])
    return x_data_out, y_data_out

def plot_data(ax: Axes, file_paths: list[str], x_data_selection: list[str], y_data_selection: list[list[str]],
              labels: list[list[str]], slice_selection: list[list[str]]=None, slice_values: list[list[float]]=None):
    ''' Plot multiple graphs from different files on the same axes.
    For multidimensional data, a slice of the data is plotted based on specified slice conditions, where the selected columns match the specified values (e.g., plot points where x1 = c1, x2 = c2, etc.)
    
    Arguments:
    ax:                 Axes object to do the plots
    file_paths:         a list containing all the file paths to use
    x_data_selection:   A list specifying the column containing the x-axis data for each corresponding file. The column identifiers should match the headers provided in each file
    y_data_selection:   A list of lists, where each inner list specifies the column(s) containing the y-axis data for each corresponding file. The column identifiers should match the headers provided in each file
    labels:             A list of lists, where each inner list specifies the label(s) for the corresponding graph(s) of each file
    slice_selection:    A list of lists, where each inner list specifies the column(s) containing the data to do the slice for each corresponding file. The column identifiers should match the headers provided in each file
    slice_values:       A list of lists, where each inner list specifies the value(s) to take the slice for each corresponding file
    '''
    headers, data = read_data(file_paths)
    x_data, y_data, slice_data = select_data(headers, data, x_data_selection, y_data_selection, slice_selection)
    if slice_values is not None:
        x_data, y_data = take_x_y_slice(x_data, y_data, slice_data, slice_values)
    for x, y, lbl in zip(x_data, y_data, labels):
        ax.plot(x, y, label=lbl)
