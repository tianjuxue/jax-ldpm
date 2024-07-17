import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os 

crt_dir = os.path.dirname(__file__)
output_dir = os.path.join(crt_dir, 'output')
numpy_dir = os.path.join(output_dir, 'numpy')

"""
free case (float32): 9686.0 s
fixed case (float32): 19346.7 s

"""

def plot_data():
    name = 'free_float32'
    # np.array([ts_save, bc_z_vals, P_top, R_support, W_ext, W_int, E_kin, sigma, eps]).T
    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    plt.figure(figsize=(12, 9))
    # plt.plot(data[:, 0], data[:, -5] - data[:, -4] - data[:, -3], linestyle='-', marker='s', markersize=1, linewidth=1, color='red')
    plt.plot(data[:, -1], data[:, -2], linestyle='-', marker='s', markersize=1, linewidth=1, color='red')
    # plt.xlabel("Strain", fontsize=20)
    # plt.ylabel("Stress [Pa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


if __name__ == '__main__':
    plot_data()