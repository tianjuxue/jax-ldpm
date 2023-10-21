"""
Reminder: Mass lumping will yield slightly worse conservation result.
"""
import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio
import datetime

from jax_ldpm.utils import json_parse
from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *

# config.update("jax_enable_x64", True)

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=10)


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
numpy_dir = os.path.join(output_dir, 'numpy')
freecad_dir = os.path.join(input_dir, 'freecad/20x20x20')

vel = 10.
dt = 1e-6
num_steps = 2000

# vel = 10.
# dt = 1e-6

def bc_tensile_disp_control_fix(points, Lz):
    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)

    def pre_compute_bc():
        bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
        top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)
        N_btm_ind = len(bottom_inds_node)
        N_tp_ind = len(top_inds_node)

        bottom_inds_tiled = np.tile(bottom_inds_node, 12)
        top_inds_tiled = np.tile(top_inds_node, 12) 
        inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))
        bottom_inds_var = np.repeat(np.arange(12), N_btm_ind)
        top_inds_var = np.repeat(np.arange(12), N_tp_ind)
        inds_var = np.hstack((bottom_inds_var, top_inds_var))

        return N_btm_ind, N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=(1, 2))
    def crt_bc(step, N_btm_ind, N_tp_ind, inds_node, inds_var):
        bc_z_val = dt*vel*step
        max_bc_z_val = dt*vel*num_steps/10 # 1e-3 strain
        bc_z_val = np.minimum(bc_z_val, max_bc_z_val)
        bottom_vals = np.zeros(N_btm_ind*12)
        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*9)))
        bc_vals = np.hstack((bottom_vals, top_vals))
        return inds_node, inds_var, bc_vals

    def calc_forces_btm(node_forces):
        forces_btm = node_forces[bottom_inds_node]
        return np.sum(forces_btm, axis=0)

    return pre_compute_bc, crt_bc, calc_forces_btm


def save_sol_helper(step, tet_points, tet_cells, points, bundled_info, state):
    vtk_path = os.path.join(output_dir, f'vtk/u_{step:05d}.vtu')
    tets_u = process_tet_point_sol(points, bundled_info, state)
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])


def save_energy_data(steps, ees, kes):
    energy_data = np.stack((steps, ees, kes))
    os.makedirs(numpy_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%s%f')
    print(f"Saving stress_strain_{now}.npy to local directory")
    onp.save(os.path.join(numpy_dir, f'energy_{now}.npy'), energy_data)


def plot_energy():
    time_stamp = '1694481179782667'
    steps, ees, kes = onp.load(os.path.join(numpy_dir, f'energy_{time_stamp}.npy'))
    plt.plot(steps, ees, linestyle='-', marker='s', markersize=2, linewidth=1, color='red', label='Elastic')
    plt.plot(steps, kes, linestyle='-', marker='s', markersize=2, linewidth=1, color='blue', label='Kinetic')
    plt.plot(steps, ees + kes, linestyle='-', marker='s', markersize=2, linewidth=1, color='black', label='Total')
    plt.xlabel("Time step", fontsize=20)
    plt.ylabel("Energy [J]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)

    params["E0"] *= 1e-3 # To make kinetic energy comparable with elastic energy
 
    params['dt'] = dt
    Young_mod = (2. + 3.*params['alpha'])/(4. + params['alpha'])*params['E0']

    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facetsVertices.dat'), dtype=float)
    meshio_mesh = meshio.read(os.path.join(freecad_dir, 'LDPMgeo000-para-mesh.000.vtk'))
    cell_type = 'tetra'
    points, cells = onp.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])

    Lx = np.max(points[:, 0])
    Ly = np.max(points[:, 1])
    Lz = np.max(points[:, 2])

    params['cells'] = cells
    params['points'] = points

    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))
 
    pre_compute_bc, crt_bc, calc_forces_btm = bc_tensile_disp_control_fix(points, Lz)

    bc_args = pre_compute_bc()

    ts = np.arange(0., dt*(num_steps + 1), dt)

    kes = [0.]
    ees = [0.]
    steps = [0]

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    for i in range(len(ts[1:])):
 
        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals = crt_bc(i, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        if (i + 1) % 40 == 0:
            print(f"Step {i + 1}")
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)
            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)
            ees.append(ee)
            kes.append(ke)
            steps.append(i + 1)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    steps = np.array(steps)
    ees = np.array(ees)
    kes = np.array(kes)

    save_energy_data(steps, ees, kes)


if __name__ == '__main__':
    simulation()
    # plot_energy()
