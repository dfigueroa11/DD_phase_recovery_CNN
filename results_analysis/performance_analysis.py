import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes

def read_data(path, folder, file_name):
    return np.loadtxt(f"{path}/{folder}/{file_name}", delimiter=",", skiprows=1)










if __name__=="__main__":
    path = "/Users/diegofigueroa/Desktop/results_post_processing"
    folder = "ASK2_sym"
    file_name = "SER_results.txt"
    data = np.delete(read_data(path, folder, file_name), (0,2), axis=1)
    x=1


