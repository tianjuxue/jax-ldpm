import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import datetime

from jax.config import config

from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.utils import json_parse
from jax_ldpm.core import *

# config.update("jax_enable_x64", True)

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})



output_dir = os.path.join(os.path.dirname(__file__), 'output')
input_dir = os.path.join(os.path.dirname(__file__), 'input')
vtk_dir = os.path.join(output_dir, 'vtk')


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)

    dt = 1e-7
    params['dt'] = dt

    points = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])

    N_nodes = len(points)
    cells = np.array([[0, 1, 2, 3]])

    params['cells'] = cells
    params['points'] = points

    v_scale = 1.
    state = np.array([[0.01, 0.01, 0.01, 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 0.]])

 
    bundled_info, tet_cells, tet_points = split_tets(cells, points, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)

    kes = []
    ees = []
    steps = []

    ts = np.arange(0., dt*10001, dt)
    for i in range(len(ts[1:])):
        print(f"Step {i + 1}")
        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)

        if (i + 1) % 100 == 0:
            vtk_path = os.path.join(vtk_dir, f'u_{i:05d}.vtu')
            tets_u = process_tet_point_sol(points, bundled_info, state)
            save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)
            ees.append(ee)
            kes.append(ke)
            steps.append(i + 1)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    steps = np.array(steps)
    ees = np.array(ees)
    kes = np.array(kes)

    plt.plot(steps, ees, linestyle='-', marker='s', markersize=2, linewidth=1, color='red', label='Elastic')
    plt.plot(steps, kes, linestyle='-', marker='s', markersize=2, linewidth=1, color='blue', label='Kinetic')
    plt.plot(steps, ees + kes, linestyle='-', marker='s', markersize=2, linewidth=1, color='black', label='Total')
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Energy [J]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


if __name__ == '__main__':
    simulation()
