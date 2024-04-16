import numpy as onp
import jax
import jax.numpy as np
import meshio
import time
import datetime
import os

from jax_ldpm.utils import json_parse
from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *

config.update("jax_enable_x64", False)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=6)

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')
os.makedirs(vtk_dir, exist_ok=True)
numpy_dir = os.path.join(output_dir, 'numpy')
os.makedirs(numpy_dir, exist_ok=True)
freecad_dir = os.path.join(input_dir, 'freecad')

"""
Unit system: [mm], [N], [s], [MPa]
"""

vel = 5.
time_to_vel = 0.002
acc = vel/time_to_vel
dt = 1e-7


def bc_disp_control_fixed(top_inds_node, bottom_inds_node):
    def pre_compute_bc():
        N_btm_ind = len(bottom_inds_node)
        N_tp_ind = len(top_inds_node)

        bottom_inds_tiled = np.tile(bottom_inds_node, 12)
        top_inds_tiled = np.tile(top_inds_node, 12) 
        inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))

        bottom_inds_var = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), N_btm_ind)
        top_inds_var = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), N_tp_ind)
        inds_var = np.hstack((bottom_inds_var, top_inds_var))

        return N_btm_ind, N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=(1, 2))
    def crt_bc(step, N_btm_ind, N_tp_ind, inds_node, inds_var):
        crt_time = step*dt
        z1 = 1./2.*acc*crt_time**2
        z2 = 1./2.*acc*time_to_vel**2 + vel*(crt_time - time_to_vel)
        bc_z_val = -np.where(acc*crt_time < vel, z1, z2)
        bc_z_vel = -np.where(acc*crt_time < vel, acc*crt_time, vel)

        bottom_vals = np.zeros(N_btm_ind*12)
        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*3), \
                              np.zeros(N_tp_ind*2), bc_z_vel*np.ones(N_tp_ind), np.zeros(N_tp_ind*3)))

        bc_vals = np.hstack((bottom_vals, top_vals))
        return inds_node, inds_var, bc_vals, bc_z_val, bc_z_vel

    return pre_compute_bc, crt_bc


def bc_disp_control_free(top_inds_node, bottom_inds_node):
    def pre_compute_bc():
        N_btm_ind = len(bottom_inds_node)
        N_tp_ind = len(top_inds_node)

        bottom_inds_tiled = np.tile(bottom_inds_node, 6)
        top_inds_tiled = np.tile(top_inds_node, 6) 
        inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))

        bottom_inds_var = np.repeat(np.array([2, 3, 4, 8, 9, 10]), N_btm_ind)
        top_inds_var = np.repeat(np.array([2, 3, 4, 8, 9, 10]), N_tp_ind)
        inds_var = np.hstack((bottom_inds_var, top_inds_var))

        return N_btm_ind, N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=(1, 2))
    def crt_bc(step, N_btm_ind, N_tp_ind, inds_node, inds_var):
     
        crt_time = step*dt
        z1 = 1./2.*acc*crt_time**2
        z2 = 1./2.*acc*time_to_vel**2 + vel*(crt_time - time_to_vel)
        bc_z_val = -np.where(acc*crt_time < vel, z1, z2)
        bc_z_vel = -np.where(acc*crt_time < vel, acc*crt_time, vel)

        bottom_vals = np.zeros(N_btm_ind*6)
        top_vals = np.hstack((bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*2), \
                              bc_z_vel*np.ones(N_tp_ind), np.zeros(N_tp_ind*2)))

        bc_vals = np.hstack((bottom_vals, top_vals))
        return inds_node, inds_var, bc_vals, bc_z_val, bc_z_vel

    return pre_compute_bc, crt_bc


def save_sol_helper(step, tet_points, tet_cells, points, bundled_info, state):
    vtk_path = os.path.join(output_dir, f'vtk/u_{step:05d}.vtu')
    tets_u = process_tet_point_sol(points, bundled_info, state)
    stv = np.vstack((bundled_info['stv'], bundled_info['stv']))
    v = np.vstack((state[bundled_info['ind_i'], 6:9], state[bundled_info['ind_j'], 6:9]))
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)], cell_infos=[('stv', stv), ('v', v)])


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
    params['dt'] = dt

    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facetsVertices.dat'), dtype=float)
    meshio_mesh = meshio.read(os.path.join(freecad_dir, 'LDPMgeo000-para-mesh.000.vtk'))
    cell_type = 'tetra'
    points, cells = onp.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])

    Lx = np.max(points[:, 0])
    Ly = np.max(points[:, 1])
    Lz = np.max(points[:, 2])
    Area = 1e4

    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
    top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)

    params['cells'] = cells
    params['points'] = points

    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))

    pre_compute_bc, crt_bc = bc_disp_control_fixed(top_inds_node, bottom_inds_node)
    bc_args = pre_compute_bc()

    ts = np.arange(0., dt*4000001, dt)
    # ts = np.arange(0., dt*20001, dt)

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)

    bc_z_val = 0.
    ldpm_node_reactions = np.zeros((len(points), 6))

    @jax.jit
    def calc_forces_top(node_forces):
        forces_top = node_forces[top_inds_node]
        return -np.sum(forces_top, axis=0)

    @jax.jit
    def calc_forces_bottom(node_forces):
        forces_bottom = node_forces[bottom_inds_node]
        return -np.sum(forces_bottom, axis=0)

    @jax.jit
    def energy_and_work(state, ldpm_node_reactions, bc_z_vel):
        node_forces = ldpm_node_reactions[:, :3]
        force = calc_forces_top(node_forces)[-1]
        dW_ext = force*bc_z_vel*dt
        dW_int = np.sum(ldpm_node_reactions*state[:, 6:]*dt)
        E_k = np.sum(0.5*state[:, 6:]*node_lumped_ms*state[:, 6:])
        return dW_ext, dW_int, E_k

    ts_save = [0.]
    bc_z_vals = [0.]
    P_top = [0.]
    R_support = [0.]
    W_ext = [0.]
    W_int = [0.]
    E_kin = [0.]
    W_external = [0.]
    W_internal = [0.]
    sigma = [0.]
    eps = [0.]

    start_time = time.time()
    for i in range(len(ts[1:])):
        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in {len(ts[1:])}, t = {ts[i + 1]}, disp = {bc_z_val}, acceleration needs {time_to_vel/dt} step")

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals, bc_z_val, bc_z_vel = crt_bc(i + 1, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        dW_ext, dW_int, E_k = energy_and_work(state, ldpm_node_reactions, bc_z_vel)
        W_external.append(W_external[-1] + dW_ext)
        W_internal.append(W_internal[-1] + dW_int)

        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)
        ldpm_node_reactions = compute_node_reactions(state, bundled_info, params)

        if (i + 1) % 1000 == 0:
            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            P = calc_forces_top(node_forces)[-1]
            R = calc_forces_bottom(node_forces)[-1]
            print(f"force on top surface = {P}")
  
            ts_save.append(ts[i + 1])
            bc_z_vals.append(-bc_z_val)
            P_top.append(-P)
            R_support.append(R)

            W_ext.append(W_external[-1])
            W_int.append(-W_internal[-1])
            E_kin.append(E_k)

            sigma.append(-P/Area)
            eps.append(-bc_z_val/Lz)

        if (i + 1) % 10000 == 0:
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)

    ts_save = np.array(ts_save)
    P_top = np.array(P_top)
    print(f"disps and forces: \n{np.stack((ts_save, P_top)).T}")
    print(f"Timing: total simulation run for {time.time() - start_time} s")

    post_analysis_data = np.array([ts_save, bc_z_vals, P_top, R_support, W_ext, W_int, E_kin, sigma, eps]).T
    now = datetime.datetime.now().strftime('%s%f')
    onp.save(os.path.join(numpy_dir, f'jax_{now}.npy'), post_analysis_data)
    # print(f"post_analysis_data = {post_analysis_data}")
    print(f"dtype = {post_analysis_data.dtype}")


if __name__ == '__main__':
    simulation()
