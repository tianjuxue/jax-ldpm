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


# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


# onp.set_printoptions(threshold=sys.maxsize,
#                      linewidth=1000,
#                      suppress=True,
#                      precision=5)


def plot_stress_strain(strain, stress):

    # plt.figure(figsize=(12, 9))
    plt.plot(strain, stress/1e6, linestyle='-', marker='o', markersize=2, linewidth=1, color='blue')
 
    plt.xlabel("Strain", fontsize=20)
    plt.ylabel("Stress [MPa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    # plt.legend(fontsize=20, frameon=False)   




def test_stress():

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    dt = 1e-7
    params['dt'] = dt

    aLength = 0.01
    st, aKt = calc_st_sKt(params, aLength)
    info = {'st': st, 'aKt': aKt, 'edge_l': aLength}

    def test_fracture(dt, params):
        epsV = 0.
        epsNs_1 = 1e-3*np.linspace(0., 1., 101)
        epsM = 0.
        epsL = 0.
        stv = np.zeros(22)
        sigmaNs_1 = []
        for epsN in epsNs_1:
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaNs_1.append(stv[1])
        sigmaNs_1 = np.array(sigmaNs_1)

        eps_max = 1e-3
        sqalpha = np.sqrt(params['alpha']) 
        epsNs_2 = 1e-3*np.linspace(0., 1., 101)
        epsL = 0.
        stv = np.zeros(22)
        sigmaNs_2 = []
        epsTs_2 = []
        sigmaTs_2 = []
        for epsN in epsNs_2:
            teta = np.pi/8.
            epsM = epsN/np.tan(teta)/sqalpha
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaNs_2.append(stv[1])
            epsTs_2.append(np.sqrt(epsM**2 + epsL**2))
            sigmaTs_2.append(np.sqrt(stv[2]**2 + stv[3]**2))
        sigmaNs_2 = np.array(sigmaNs_2)
        epsTs_2 = np.array(epsTs_2)
        sigmaTs_2 = np.array(sigmaTs_2)
        inds_table = epsTs_2 < eps_max
        epsTs_2 = epsTs_2[inds_table]
        sigmaTs_2 = sigmaTs_2[inds_table]

        epsN = 0.
        epsMs_3 = 1e-3*np.linspace(0., 1., 101)
        epsL = 0.
        stv = np.zeros(22)
        epsTs_3 = []
        sigmaTs_3 = []
        for epsM in epsMs_3:
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            epsTs_3.append(np.sqrt(epsM**2 + epsL**2))
            sigmaTs_3.append(np.sqrt(stv[2]**2 + stv[3]**2))
        epsTs_3 = np.array(epsTs_3)
        sigmaTs_3 = np.array(sigmaTs_3)


        plt.figure()
        plot_stress_strain(epsNs_1, sigmaNs_1)
        plot_stress_strain(epsNs_2, sigmaNs_2)
        plot_stress_strain(epsTs_2, sigmaTs_2)
        plot_stress_strain(epsTs_3, sigmaTs_3) 


    def test_fracture_cyclic(dt, params):
        epsV = 0.

        path1 = np.linspace(0., 0.4, 101)
        path2 = np.linspace(0.4, 0., 101)
        path3 = np.linspace(0., 0.4, 101)      
        path4 = np.linspace(0.4, 0.2, 101)
        path5 = np.linspace(0.2, 0.4, 101)
        epsNs_1 = 1e-3*np.hstack((path1, path2, path3, path4, path5))

        epsM = 0.
        epsL = 0.
        stv = np.zeros(22)
        sigmaNs_1 = []
        for epsN in epsNs_1:
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaNs_1.append(stv[1])
        sigmaNs_1 = np.array(sigmaNs_1)

        plot_stress_strain(epsNs_1, sigmaNs_1) 


    def test_compression(dt, params):
        params['E0'] = 60000e6
        ec = params['fc'] / params['E0']
        ec0 = params['tsrn_e'] * ec

        path1 = np.linspace(0., -ec0, 51)
        path2 = np.linspace(-ec0, 0, 51)
        path3 = np.linspace(0., -15e-3, 101)
        epsNs_1 = np.hstack((path1, path2, path3)) 
        epsM = 0.
        epsL = 0.
        stv = np.zeros(22)
        sigmaNs_1 = []
        for epsN in epsNs_1:
            epsV = epsN
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaNs_1.append(stv[1])
        sigmaNs_1 = np.array(sigmaNs_1)


        epsNs_2 = np.linspace(0., -15e-3, 101)

        epsM = 0.
        epsL = 0.
        stv = np.zeros(22)
        sigmaNs_2 = []
        for epsN in epsNs_2:
            epsV = epsN/2.1
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaNs_2.append(stv[1])
        sigmaNs_2 = np.array(sigmaNs_2)

        plt.figure()
        plot_stress_strain(-epsNs_1, -sigmaNs_1)
        plot_stress_strain(-epsNs_2, -sigmaNs_2)



    def test_shear(dt, params):
        params['E0'] = 60000e6
        ec = params['fc'] / params['E0']
        ec0 = params['tsrn_e'] * ec

        path1 = np.linspace(0., 15e-3, 101)
        path2 = np.linspace(15e-3, -15e-3, 101)
        path3 = np.linspace(-15e-3, 15e-3, 101)
 
        epsN = -1e-3
        epsMs = np.hstack((path1, path2, path3)) 
        epsL = 0.
        stv = np.zeros(22)
        sigmaMs = []
        for epsM in epsMs:
            epsV = epsN
            eps = np.array([epsN, epsM, epsL])
            stv = stress_fn(eps, epsV, stv, info, params)
            sigmaMs.append(stv[2])
        sigmaMs = np.array(sigmaMs)

        plot_stress_strain(epsMs, sigmaMs)

    # test_fracture(dt, params)
    # test_fracture_cyclic(dt, params)
    test_compression(dt, params)
    # test_shear(dt, params)


if __name__ == '__main__':
    test_stress()
    plt.show()