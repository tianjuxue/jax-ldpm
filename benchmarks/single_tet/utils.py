"""
Visualize results from Abaqus execution results, form Matthew or Ke
"""
import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import os


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')


def generate_specific_rows(fname):
    ts = []
    svar_data = []
    with open(fname) as f:
        for i, line in enumerate(f):
            if i % 6 == 0:
                td = onp.fromstring(line, dtype=float, sep=' ')
                ts.append(td[0])
            if i % 6 == 5:
                myarray = onp.fromstring(line, dtype=float, sep=' ')
                svar_data.append(myarray)         

    return onp.stack(ts), onp.stack(svar_data)


def read_file(flag_tet, flag_case, svar_dir, numpy_dir):
    abaqus_ts, abaqus_svar_data = generate_specific_rows(svar_dir)
    num_faces = 12
    num_svars = 6
    abaqus_svar_data = abaqus_svar_data.reshape(-1, num_faces, num_svars)
    abaqus_ts = abaqus_ts[::12]
    jax_svar_data = onp.load(os.path.join(numpy_dir, 'svar_data.npy'))
    jax_ts = onp.load(os.path.join(numpy_dir, 'ts.npy'))

    print(abaqus_svar_data.shape)
    print(jax_svar_data.shape)

    def compare_plot(face_no, svar_id):
        """
        jax_svar_data[:, face_no, :3] is stress data
        jax_svar_data[:, face_no, 3:] is strain data
        """
        plt.plot(jax_ts, jax_svar_data[:, face_no, svar_id], linestyle='-', marker='s', 
            markersize=0.5, linewidth=0.5, color='red', label='JAX-LDPM')
        plt.plot(abaqus_ts, abaqus_svar_data[:, face_no, svar_id], linestyle='-', marker='s', 
            markersize=0.5, linewidth=0.5, color='blue', label='Abaqus')
        plt.xlabel("Time", fontsize=20)
        plt.ylabel("Value", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.legend(fontsize=20, frameon=False)

    for i in range(12):
        plt.figure()
        compare_plot(i, 0)

    for i in range(12):
        plt.figure()
        compare_plot(i, 1)

    for i in range(12):
        plt.figure()
        compare_plot(i, 2)

    # for i in range(6):
    #     plt.figure()
    #     compare_plot(0, i)

    # plt.figure()
    # compare_plot(11, 0)


def driver():
    flags_tet = ['regular', 'irregular']
    flags_case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']

    flags_tet = ['irregular']
    flags_case = ['case7']

    numpy_dir = os.path.join(output_dir, 'numpy', flags_tet[0], flags_case[0])
    svar_data = onp.load(os.path.join(numpy_dir, 'svar_data.npy'))
    epsV = onp.load(os.path.join(numpy_dir, 'epsV.npy'))
    ts = onp.load(os.path.join(numpy_dir, 'ts.npy'))
    reactions = onp.load(os.path.join(numpy_dir, 'reactions.npy'))

    # print(svar_data.shape)
    # print(epsV.shape)
    # print(ts.shape)
    # print(reactions.shape)

    for i in range(len(flags_tet)):
        for j in range(len(flags_case)):
            svar_dir = os.path.join(input_dir, 'abaqus_result', flags_tet[i], flags_case[j], 'svars-000.txt')
            numpy_dir = os.path.join(output_dir, 'numpy', flags_tet[i], flags_case[j])
            read_file(flags_tet[i], flags_case[j], svar_dir, numpy_dir)

if __name__ == '__main__':
    driver()
    plt.show()
