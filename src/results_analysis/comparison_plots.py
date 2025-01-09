import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler

from ploting_tools import plot_data


l_link = 6
mod_formats = ["ASK2", "ASK4", "PAM2", "PAM4", "QAM4"]
file_paths = [f"/Users/diegofigueroa/Desktop/tvrnn_norm_results/{mod_format}.txt" for mod_format in mod_formats]
y_data_selection = [["MI","SDD_MI"] for mod_format in mod_formats]
x_data_selection = ["SNR" for mod_format in mod_formats]
slice_selection = [["L_link_km"] for mod_format in mod_formats]
slice_values = [l_link for mod_format in mod_formats]
labels = [[f"{mod_format} MI", f"{mod_format} SDD MI"] for mod_format in mod_formats]

# use the cycler to set the properties of each graph
cc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
color_cycler = []
for i in range(len(mod_formats)):
    color_cycler += [cc[i], cc[i]] 
mpl.rcParams['axes.prop_cycle'] = cycler(color=color_cycler,
                                         ls=['-','--']*len(mod_formats),
                                         marker=['o',]*len(color_cycler))

ax: Axes
fig, ax = plt.subplots()
plot_data(ax,file_paths, x_data_selection, y_data_selection, labels, slice_selection, slice_values)
ax.legend()
ax.grid()
ax.set_xlim(-5,11)
plt.show()