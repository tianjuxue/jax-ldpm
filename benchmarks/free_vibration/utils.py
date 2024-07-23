import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os 

crt_dir = os.path.dirname(__file__)
output_dir = os.path.join(crt_dir, 'output')
numpy_dir = os.path.join(output_dir, 'numpy')

"""
float32: 3490.77s
float64: None

"""

def plot_data():
    name = 'jax_11029420'
    # np.array([ts_save, R_support, W_int, W_ext, E_kin,\
              # ux_0, uy_0, uz_0, rx_0, ry_0, rz_0,\
              # ux_1, uy_1, uz_1, rx_1, ry_1, rz_1,\
              # ux_2, uy_2, uz_2, rx_2, ry_2, rz_2,\
              # ux_3, uy_3, uz_3, rx_3, ry_3, rz_3,\
              # ux_4, uy_4, uz_4, rx_4, ry_4, rz_4,\
              # ux_5, uy_5, uz_5, rx_5, ry_5, rz_5,\
              # ux_6, uy_6, uz_6, rx_6, ry_6, rz_6,\
              # ux_7, uy_7, uz_7, rx_7, ry_7, rz_7,\
              # ux_8, uy_8, uz_8, rx_8, ry_8, rz_8,\
              # ux_9, uy_9, uz_9, rx_9, ry_9, rz_9,\
              # ux_10, uy_10, uz_10, rx_10, ry_10, rz_10]).T

    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    plt.figure()
    plt.plot(data[:, 0], data[:, 4], linestyle='-', markersize=1, linewidth=1, color='m', label='R_T')
    # plt.plot(data[:, 0], abs(data[:, 3] - data[:, 2] - data[:, 4]), linestyle='-', markersize=1, linewidth=1, color='m', label='R_T')
    plt.xlabel("Time [s]", fontsize=10)
    plt.ylabel('Ek [mJ]', fontsize=10)
    # plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20, frameon=False)   
    plt.title('Kinetic Energy', fontsize=12)
    plt.grid(True)
    plt.show()

def plot_fft():
    name = 'jax_11029420'
    data = onp.load(os.path.join(numpy_dir, f'{name}.npy'))
    ts = 0.00002
    N = len(data)
    fft_result = onp.fft.fft(data[:, 59])
    freq = onp.fft.fftfreq(N, ts)[:N//2]

    
    plt.figure()
    plt.plot(freq, onp.abs(fft_result[0:N//2]), markersize=1, linewidth=1, color='m', label='Ux')
    plt.legend()
    plt.xlabel('Frequencies [Hz]')
    plt.ylabel('Magnitude [mm]')
    plt.title('Fourier Transform')
    plt.grid(True)
    plt.show()
    

if __name__ == '__main__':
    plot_data()
    # plot_fft()