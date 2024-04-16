import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os 

crt_dir = os.path.dirname(__file__)
output_dir = os.path.join(crt_dir, 'output')
numpy_dir = os.path.join(output_dir, 'numpy')


"""
5mm_float32: 2831 s
5mm_float64: 15518 s
15mm_float32: 1463 s
15mm_float64: 7751 s
"""

def plot_data():
    name = '5mm_float64'
    # np.array([ts_save, bc_z_vals, P_top, R_support, CMOD, v1, v2, W_ext, W_int, E_kin]).T
    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    plt.figure(figsize=(12, 9))
    plt.plot(data[:, 0], data[:, -3] - data[:, -2] - data[:, -1], linestyle='-', marker='s', markersize=1, linewidth=1, color='red')
    # plt.plot(data[:, 0], data[:, 2], linestyle='-', marker='s', markersize=1, linewidth=1, color='red')
    # plt.xlabel("Strain", fontsize=20)
    # plt.ylabel("Stress [Pa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


if __name__ == '__main__':
    plot_data()