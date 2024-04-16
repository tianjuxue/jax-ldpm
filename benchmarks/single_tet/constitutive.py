"""For debugging purpose
"""
import numpy as onp
import jax
import jax.numpy as np
import os
import sys
import matplotlib.pyplot as plt

from jax_ldpm.constitutive import stress_fn, calc_st_sKt
from jax_ldpm.utils import json_parse

stress_fn = jax.jit(stress_fn)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=15)


# onp.set_printoptions(threshold=sys.maxsize,
#                      linewidth=1000,
#                      suppress=True,
#                      precision=5)


def plot_stress_strain(strain, stress):

    # plt.figure(figsize=(12, 9))
    plt.plot(strain, stress, linestyle='-', marker='o', markersize=2, linewidth=1, color='red')
 
    # plt.xlabel("Strain", fontsize=20)
    # plt.ylabel("Stress [MPa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20, frameon=False)   


def test_stress():

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    dt = 1e-3
    params['dt'] = dt

    # aLength = 3.5
    aLength = 1

    st, aKt = calc_st_sKt(params, aLength)
    info = {'st': st, 'aKt': aKt, 'edge_l': aLength}

    numpy_dir = os.path.join(output_dir, 'numpy', 'regular', 'case4')
    svar_data = onp.load(os.path.join(numpy_dir, 'svar_data.npy'))
    epsV_data = onp.load(os.path.join(numpy_dir, 'epsV.npy'))
    t_data = onp.load(os.path.join(numpy_dir, 'ts.npy'))

    facet_id = 11
    eps_facet = svar_data[:, facet_id, 3:6]
    epsV_facet = epsV_data[:, facet_id]

    stv = np.zeros(22)
    sigmaNs = []
    sigmaMs = []
    for i in range(len(eps_facet)):

        eps = eps_facet[i]
        epsV = epsV_facet[i]
        stv = stress_fn(eps, epsV, stv, info, params)
        sigmaNs.append(stv[1])
        sigmaMs.append(stv[2])

        print(stv[1:7])

        if i > 400:
            exit()

    sigmaNs = np.array(sigmaNs)
    sigmaMs = np.array(sigmaMs)


    plt.figure()
    plot_stress_strain(t_data, sigmaNs)

    # plt.figure()
    # plot_stress_strain(t_data, eps_facet[:, 0])
 
 

if __name__ == '__main__':
    test_stress()
    plt.show()