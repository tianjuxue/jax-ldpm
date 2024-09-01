import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os 
import matplotlib.pyplot as plt 
import pandas as pd 

crt_dir = os.path.dirname(__file__)
output_dir = os.path.join(crt_dir, 'output')
input_dir = os.path.join(crt_dir, 'input')
numpy_dir = os.path.join(output_dir, 'numpy')
xlsx_dir = os.path.join(input_dir, 'xlsx')


"""
For LDPM_mesh_8-16:

5mm_float32: 2831 s
5mm_float64: 15518 s
15mm_float32: 1463 s
15mm_float64: 7751 s


For LDPM_mesh_6-12:

15mm_float32: 4580 s
"""

def plot_data():
    name = '6-12_15mm_float32'
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


def compare_with_chrono_and_abaqus():
    name = '6-12_15mm_float32'
    # np.array([ts_save, bc_z_vals, P_top, R_support, CMOD, v1, v2, W_ext, W_int, E_kin]).T
    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    plt.figure(figsize=(12, 9))
    plt.plot(data[:, 0], data[:, 2], linestyle='-', marker='s', markersize=1, linewidth=1, color='orange', label='jax')

    chrono_data = pd.read_excel(os.path.join(xlsx_dir, '6-12_chrono_abaqus.xlsx'), sheet_name='dt0.00001')
    abaqus_data = pd.read_excel(os.path.join(xlsx_dir, '6-12_chrono_abaqus.xlsx'), sheet_name='ABAQUS')
    c_time = chrono_data['time'].to_numpy()
    c_force = chrono_data['F'].to_numpy()
    a_time = abaqus_data['time'].to_numpy()
    a_force = abaqus_data['F'].to_numpy()

    plt.plot(c_time, -c_force, linestyle='-', marker='s', markersize=1, linewidth=1, color='red', label='chrono')
    plt.plot(a_time, -a_force, linestyle='-', marker='s', markersize=1, linewidth=1, color='blue', label='abaqus')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Force", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


if __name__ == '__main__':
    # plot_data()
    compare_with_chrono_and_abaqus()
