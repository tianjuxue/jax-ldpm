import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os 

crt_dir = os.path.dirname(__file__)
output_dir = os.path.join(crt_dir, 'output')
numpy_dir = os.path.join(output_dir, 'numpy')


"""
case1:
float32: 2095.06s
float64: 4836.48s

case2:
float32: 1551.14s
float64: 3055.80s

"""

def plot_data():
    name = 'jax_20818274'
    # np.array([ts_save, bc_z_vals, P_top, R_support, W_ext, W_int, E_kin, sigma, eps]).T
    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    # plt.figure(figsize=(12, 9))
    plt.plot(data[:, 0], data[:, 6], linestyle='-', markersize=1, linewidth=1, color='m', label='R_T')
    plt.xlabel("Time [s]", fontsize=20)
    # plt.ylabel('Energy Balance Error [%]', fontsize=20)
    plt.ylabel('External Work [mJ]', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20)   
    # plt.legend(fontsize=20, frameon=False)   
    plt.title('Energy Balance Error', fontsize=20)
    # plt.ylim(0, 0.5)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_data()