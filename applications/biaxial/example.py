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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# onp.set_printoptions(threshold=sys.maxsize,
#                      linewidth=1000,
#                      suppress=True,
#                      precision=5)


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
numpy_dir = os.path.join(output_dir, 'numpy')
freecad_dir = os.path.join(input_dir, 'freecad/200x50x200')


vel = 100.
dt = 1e-7


def bc_tensile_disp_control_fix(points, Lz):
    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
    
    top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)
    def pre_compute_bc():
        bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
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
        bc_z_val = -dt*vel*step
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
    stv = np.vstack((bundled_info['stv'], bundled_info['stv']))
    v = np.vstack((state[bundled_info['ind_i'], 6:9], state[bundled_info['ind_j'], 6:9]))
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)], cell_infos=[('stv', stv), ('v', v)])


def save_stress_strain_data(strains, expected_s, predicted_s):
    stress_strain_data = np.stack((strains, expected_s, predicted_s))
    os.makedirs(numpy_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%s%f')
    print(f"Saving stress_strain_{now}.npy to local directory")
    onp.save(os.path.join(numpy_dir, f'stress_strain_{now}.npy'), stress_strain_data)


def plot_stress_strain():
    time_stamp = '1693837386537882'
    strains, expected_s, predicted_s = onp.load(os.path.join(numpy_dir, f'stress_strain_{time_stamp}.npy'))
    plt.figure(figsize=(12, 9))
    plt.plot(strains, -predicted_s, linestyle='-', marker='s', markersize=2, linewidth=1, color='red', label='predicted')
    plt.xlabel("Strain", fontsize=20)
    plt.ylabel("Stress [Pa]", fontsize=20)
    plt.tick_params(labelsize=20)
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


@jax.jit
def constrain_angle(state):
    max_theta = 0.1
    angle = np.where(state[:, 3:6] < max_theta, state[:, 3:6], max_theta)
    angle = np.where(angle > -max_theta, angle, -max_theta)
    state = state.at[:, 3:6].set(angle)

    max_v = 100.
    v = np.where(state[:, 6:9] < max_v, state[:, 6:9], max_v)
    v = np.where(v > -max_v, v, -max_v)
    state = state.at[:, 6:9].set(v)

    return state


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
 
    params['dt'] = dt

    params["damping_v"] = 1e3
    params["damping_w"] = 1e3
    # params["EAF"] = 1

    Young_mod = (2. + 3.*params['alpha'])/(4. + params['alpha'])*params['E0']
    print(f"Young_mod = {Young_mod}, rho = {params['rho']}")

    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facetsVertices.dat'), dtype=float)
    meshio_mesh = meshio.read(os.path.join(freecad_dir, 'LDPMgeo000-para-mesh.000.vtk'))
    cell_type = 'tetra'
    points, cells = onp.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])

    Lx = np.max(points[:, 0])
    Ly = np.max(points[:, 1])
    Lz = np.max(points[:, 2])

    print(f"Lz = {Lz}")
    print(f"dt = {dt}")

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

    fric_flag = np.ones(len(points), dtype=bool)

    ts = np.arange(0., dt*50001, dt)

    strains = [0.]
    predicted_s = [0.]
    expected_s = [0.]

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    for i in range(len(ts[1:])):

        if (i + 1) % 5000 == 0:
            print(f"Step {i + 1}")
            print(f"max theta 1 = {np.max(state[:, 3])}")
            print(f"max theta 2 = {np.max(state[:, 4])}")
            print(f"max theta 3 = {np.max(state[:, 5])}")
            print(f"max v 1 = {np.max(np.abs(state[:, 6]))}")
            print(f"max v 2 = {np.max(np.abs(state[:, 7]))}")
            print(f"max v 3 = {np.max(np.abs(state[:, 8]))}")

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals = crt_bc(i, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)
        # state = constrain_angle(state)


        if (i + 1) % 5000 == 0:
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)
            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)

            node_forces = compute_node_forces(state, bundled_info, params)
            force_predicted = calc_forces_btm(node_forces)
            stess_predicted = force_predicted[-1]/(Lx*Ly)
            disp = crt_t*vel
            strain = disp/Lz
            stress_expected = strain*Young_mod
            print(f"force at btm = {force_predicted}, strain (x1000) = {strain*1e3}, stess_predicted = {stess_predicted}, stress_expected = {stress_expected}")
            strains.append(strain)
            predicted_s.append(stess_predicted)
            expected_s.append(stress_expected)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    strains = np.array(strains)
    expected_s = np.array(expected_s)
    predicted_s = np.array(predicted_s)

    save_stress_strain_data(strains, expected_s, predicted_s)


if __name__ == '__main__':
    simulation()
    # plot_stress_strain()
