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

jax.config.update("jax_enable_x64", False)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

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
Unit system: [mm], [N], [s]
"""

P_max = 50
time_to_P_max = 0.01
time_to_P_min = 0.00004
k_1 = P_max / time_to_P_max
k_2 = P_max / time_to_P_min
dt = 1e-7

def bc_disp_control(bottom_inds_node):
    def pre_compute_bc():
        N_btm_ind = len(bottom_inds_node)

        bottom_inds_tiled = np.tile(bottom_inds_node, 12)
        inds_node = np.hstack((bottom_inds_tiled))

        bottom_inds_var = np.repeat(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), N_btm_ind)
        inds_var = np.hstack((bottom_inds_var))

        return N_btm_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=1)
    def crt_bc(step, N_btm_ind, inds_node, inds_var):
        crt_time = step*dt

        bottom_vals = np.zeros(N_btm_ind*12)
       
        bc_vals = np.hstack((bottom_vals))
        return inds_node, inds_var, bc_vals

    return pre_compute_bc, crt_bc


def bc_force_control(top_bc_inds_node):
    def pre_compute_bc():
        N_tp_ind = len(top_bc_inds_node)

        top_bc_inds_tiled = np.tile(top_bc_inds_node, 6) 
        inds_node = np.hstack((top_bc_inds_tiled))

        top_bc_inds_var = np.repeat(np.array([0, 1, 2, 3, 4, 5]), N_tp_ind)
        inds_var = np.hstack((top_bc_inds_var))

        return N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=1)
    def crt_bc(step, N_tp_ind, inds_node, inds_var):
        crt_time = step * dt
        F1 = k_1 * crt_time
        F2 = k_1 * time_to_P_max - k_2  * (crt_time - time_to_P_max)
        
        bc_F_val = np.where(k_1 * crt_time < P_max, F1, \
                            np.where(k_1 * time_to_P_max - k_2  * (crt_time - time_to_P_max) > 0, F2, 0))
        
        top_vals = np.hstack((bc_F_val*np.ones(N_tp_ind), np.zeros(N_tp_ind * 5)))

        bc_vals = np.hstack((top_vals))
        return inds_node, inds_var, bc_vals, bc_F_val

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

    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
    top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)
    top_bc_inds_node = np.argwhere((points[:, 2] > Lz - 1e-5) & ((points[:, 0] - Lx)**2 + (points[:, 1] - Ly)**2 < 1e-5)).reshape(-1)
    
    target_node = {'0': np.array([50, 50, 0]),
                   '1': np.array([54.4809, 48.6083, 20.97]),
                   '2': np.array([52.1469, 45.7597, 40.2412]),
                   '3': np.array([48.588, 51.7969, 60.8753]),
                   '4': np.array([44.0282, 56.7126, 79.6343]),
                   '5': np.array([57.7684, 50.4392, 100.096]),
                   '6': np.array([46.7897, 32.1821, 119.21]),
                   '7': np.array([45.8158, 56.241, 140.666]),
                   '8': np.array([53.2392, 62.5964, 160.118]),
                   '9': np.array([40.2156, 42.9829, 179.876]),
                   '10': np.array([50, 50, 200])}
    
    def find_node_inds(node_pos):
        node_inds = np.argwhere((points[:, 0] - node_pos[0])**2 + \
                                (points[:, 1] - node_pos[1])**2 + \
                                (points[:, 2] - node_pos[2])**2 < 1e-5).reshape(-1)
        return node_inds
    
    
    node_00_inds = find_node_inds(target_node['0'])
    node_01_inds = find_node_inds(target_node['1'])
    node_02_inds = find_node_inds(target_node['2'])
    node_03_inds = find_node_inds(target_node['3'])
    node_04_inds = find_node_inds(target_node['4'])
    node_05_inds = find_node_inds(target_node['5'])
    node_06_inds = find_node_inds(target_node['6'])
    node_07_inds = find_node_inds(target_node['7'])
    node_08_inds = find_node_inds(target_node['8'])
    node_09_inds = find_node_inds(target_node['9'])
    node_10_inds = find_node_inds(target_node['10'])

    params['cells'] = cells
    params['points'] = points

    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))
    applied_force = np.zeros((len(points), 6))

    pre_compute_disp_bc, crt_disp_bc = bc_disp_control(bottom_inds_node)
    disp_bc_args = pre_compute_disp_bc()
    
    pre_compute_force_bc, crt_force_bc = bc_force_control(top_bc_inds_node)
    force_bc_args = pre_compute_force_bc()
    
    ts = np.arange(0., dt*1000001, dt)

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    
    ldpm_node_reactions = np.zeros((len(points), 6))
    bc_F_val = 0.
    
    @jax.jit
    def calc_forces_top(node_forces):
        forces_top = node_forces[top_inds_node]
        return np.sum(forces_top, axis=0)

    @jax.jit
    def calc_forces_bottom(node_forces):
        forces_bottom = node_forces[bottom_inds_node]
        return np.sum(forces_bottom, axis=0)

    @jax.jit
    def energy_and_work(state, ldpm_node_reactions, x_vel, bc_F_val):
        node_forces = ldpm_node_reactions[:, :3]
        force = bc_F_val
        dW_ext = force*x_vel*dt
        dW_int = np.sum(ldpm_node_reactions*state[:, 6:]*dt)
        E_k = np.sum(0.5*state[:, 6:]*node_lumped_ms*state[:, 6:])
        return dW_ext, dW_int, E_k

    ts_save = [0.]
    bc_F_vals = [0.]
    R_support = [0.]
    
    ux_0 = [0.]
    uy_0 = [0.]
    uz_0 = [0.]
    rx_0 = [0.]
    ry_0 = [0.]
    rz_0 = [0.]
    
    ux_1 = [0.]
    uy_1 = [0.]
    uz_1 = [0.]
    rx_1 = [0.]
    ry_1 = [0.]
    rz_1 = [0.]
    
    ux_2 = [0.]
    uy_2 = [0.]
    uz_2 = [0.]
    rx_2 = [0.]
    ry_2 = [0.]
    rz_2 = [0.]
    
    ux_3 = [0.]
    uy_3 = [0.]
    uz_3 = [0.]
    rx_3 = [0.]
    ry_3 = [0.]
    rz_3 = [0.]
    
    ux_4 = [0.]
    uy_4 = [0.]
    uz_4 = [0.]
    rx_4 = [0.]
    ry_4 = [0.]
    rz_4 = [0.]
    
    ux_5 = [0.]
    uy_5 = [0.]
    uz_5 = [0.]
    rx_5 = [0.]
    ry_5 = [0.]
    rz_5 = [0.]
    
    ux_6 = [0.]
    uy_6 = [0.]
    uz_6 = [0.]
    rx_6 = [0.]
    ry_6 = [0.]
    rz_6 = [0.]
    
    ux_7 = [0.]
    uy_7 = [0.]
    uz_7 = [0.]
    rx_7 = [0.]
    ry_7 = [0.]
    rz_7 = [0.]
    
    ux_8 = [0.]
    uy_8 = [0.]
    uz_8 = [0.]
    rx_8 = [0.]
    ry_8 = [0.]
    rz_8 = [0.]
    
    ux_9 = [0.]
    uy_9 = [0.]
    uz_9 = [0.]
    rx_9 = [0.]
    ry_9 = [0.]
    rz_9 = [0.]
    
    ux_10 = [0.]
    uy_10 = [0.]
    uz_10 = [0.]
    rx_10 = [0.]
    ry_10 = [0.]
    rz_10 = [0.]
    
    W_ext = [0.]
    W_int = [0.]
    E_kin = [0.]
    W_external = [0.]
    W_internal = [0.]


    start_time = time.time()
    for i in range(len(ts[1:])):
        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in {len(ts[1:])}, t = {ts[i + 1]}, force = {bc_F_val}, acceleration needs {time_to_P_max/dt} steps")
        crt_t = ts[i + 1]
        print(f"Step {i + 1}")
        inds_node_disp, inds_var_disp, disp_bc_vals = crt_disp_bc(i + 1, *disp_bc_args)
        inds_node_force, inds_var_force, force_bc_vals, bc_F_val = crt_force_bc(i + 1, *force_bc_args)
        
        state = apply_bc(state, inds_node_disp, inds_var_disp, disp_bc_vals)
        applied_force = apply_bc(applied_force, inds_node_force, inds_var_force, force_bc_vals)
        x_vel = state[top_bc_inds_node, 6][0]
        
        dW_ext, dW_int, E_k = energy_and_work(state, ldpm_node_reactions, x_vel, bc_F_val)
        W_external.append(W_external[-1] + dW_ext)
        W_internal.append(W_internal[-1] + dW_int)

        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params, applied_reactions=applied_force)
        ldpm_node_reactions = compute_node_reactions(state, bundled_info, params)

        if (i + 1) % 200 == 0:
            # Get post-analysis data
            ux_0.append(state[node_00_inds, 0][0])
            uy_0.append(state[node_00_inds, 1][0])
            uz_0.append(state[node_00_inds, 2][0])
            rx_0.append(state[node_00_inds, 3][0])
            ry_0.append(state[node_00_inds, 4][0])
            rz_0.append(state[node_00_inds, 5][0])
            
            ux_1.append(state[node_01_inds, 0][0])
            uy_1.append(state[node_01_inds, 1][0])
            uz_1.append(state[node_01_inds, 2][0])
            rx_1.append(state[node_01_inds, 3][0])
            ry_1.append(state[node_01_inds, 4][0])
            rz_1.append(state[node_01_inds, 5][0])
            
            ux_2.append(state[node_02_inds, 0][0])
            uy_2.append(state[node_02_inds, 1][0])
            uz_2.append(state[node_02_inds, 2][0])
            rx_2.append(state[node_02_inds, 3][0])
            ry_2.append(state[node_02_inds, 4][0])
            rz_2.append(state[node_02_inds, 5][0])
            
            ux_3.append(state[node_03_inds, 0][0])
            uy_3.append(state[node_03_inds, 1][0])
            uz_3.append(state[node_03_inds, 2][0])
            rx_3.append(state[node_03_inds, 3][0])
            ry_3.append(state[node_03_inds, 4][0])
            rz_3.append(state[node_03_inds, 5][0])
            
            ux_4.append(state[node_04_inds, 0][0])
            uy_4.append(state[node_04_inds, 1][0])
            uz_4.append(state[node_04_inds, 2][0])
            rx_4.append(state[node_04_inds, 3][0])
            ry_4.append(state[node_04_inds, 4][0])
            rz_4.append(state[node_04_inds, 5][0])
            
            ux_5.append(state[node_05_inds, 0][0])
            uy_5.append(state[node_05_inds, 1][0])
            uz_5.append(state[node_05_inds, 2][0])
            rx_5.append(state[node_05_inds, 3][0])
            ry_5.append(state[node_05_inds, 4][0])
            rz_5.append(state[node_05_inds, 5][0])
            
            ux_6.append(state[node_06_inds, 0][0])
            uy_6.append(state[node_06_inds, 1][0])
            uz_6.append(state[node_06_inds, 2][0])
            rx_6.append(state[node_06_inds, 3][0])
            ry_6.append(state[node_06_inds, 4][0])
            rz_6.append(state[node_06_inds, 5][0])
            
            ux_7.append(state[node_07_inds, 0][0])
            uy_7.append(state[node_07_inds, 1][0])
            uz_7.append(state[node_07_inds, 2][0])
            rx_7.append(state[node_07_inds, 3][0])
            ry_7.append(state[node_07_inds, 4][0])
            rz_7.append(state[node_07_inds, 5][0])
            
            ux_8.append(state[node_08_inds, 0][0])
            uy_8.append(state[node_08_inds, 1][0])
            uz_8.append(state[node_08_inds, 2][0])
            rx_8.append(state[node_08_inds, 3][0])
            ry_8.append(state[node_08_inds, 4][0])
            rz_8.append(state[node_08_inds, 5][0])
            
            ux_9.append(state[node_09_inds, 0][0])
            uy_9.append(state[node_09_inds, 1][0])
            uz_9.append(state[node_09_inds, 2][0])
            rx_9.append(state[node_09_inds, 3][0])
            ry_9.append(state[node_09_inds, 4][0])
            rz_9.append(state[node_09_inds, 5][0])
            
            ux_10.append(state[node_10_inds, 0][0])
            uy_10.append(state[node_10_inds, 1][0])
            uz_10.append(state[node_10_inds, 2][0])
            rx_10.append(state[node_10_inds, 3][0])
            ry_10.append(state[node_10_inds, 4][0])
            rz_10.append(state[node_10_inds, 5][0])
            
            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            R = calc_forces_bottom(node_forces)[0]

            ts_save.append(ts[i + 1])
            bc_F_vals.append(bc_F_val)
            R_support.append(-R)
            
            W_ext.append(W_external[-1])
            W_int.append(-W_internal[-1])
            E_kin.append(E_k)

        if (i + 1) % 10000 == 0:
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)
    
    ts_save = np.array(ts_save)
    ux_0 = np.array(ux_0)
    uy_0 = np.array(uy_0)
    uz_0 = np.array(uz_0)
    rx_0 = np.array(rx_0)
    ry_0 = np.array(ry_0)
    rz_0 = np.array(rz_0)
    
    ux_1 = np.array(ux_1)
    uy_1 = np.array(uy_1)
    uz_1 = np.array(uz_1)
    rx_1 = np.array(rx_1)
    ry_1 = np.array(ry_1)
    rz_1 = np.array(rz_1)
    
    ux_2 = np.array(ux_2)
    uy_2 = np.array(uy_2)
    uz_2 = np.array(uz_2)
    rx_2 = np.array(rx_2)
    ry_2 = np.array(ry_2)
    rz_2 = np.array(rz_2)
    
    ux_3 = np.array(ux_3)
    uy_3 = np.array(uy_3)
    uz_3 = np.array(uz_3)
    rx_3 = np.array(rx_3)
    ry_3 = np.array(ry_3)
    rz_3 = np.array(rz_3)
    
    ux_4 = np.array(ux_4)
    uy_4 = np.array(uy_4)
    uz_4 = np.array(uz_4)
    rx_4 = np.array(rx_4)
    ry_4 = np.array(ry_4)
    rz_4 = np.array(rz_4)
    
    ux_5 = np.array(ux_5)
    uy_5 = np.array(uy_5)
    uz_5 = np.array(uz_5)
    rx_5 = np.array(rx_5)
    ry_5 = np.array(ry_5)
    rz_5 = np.array(rz_5)
    
    ux_6 = np.array(ux_6)
    uy_6 = np.array(uy_6)
    uz_6 = np.array(uz_6)
    rx_6 = np.array(rx_6)
    ry_6 = np.array(ry_6)
    rz_6 = np.array(rz_6)
    
    ux_7 = np.array(ux_7)
    uy_7 = np.array(uy_7)
    uz_7 = np.array(uz_7)
    rx_7 = np.array(rx_7)
    ry_7 = np.array(ry_7)
    rz_7 = np.array(rz_7)
    
    ux_8 = np.array(ux_8)
    uy_8 = np.array(uy_8)
    uz_8 = np.array(uz_8)
    rx_8 = np.array(rx_8)
    ry_8 = np.array(ry_8)
    rz_8 = np.array(rz_8)
    
    ux_9 = np.array(ux_9)
    uy_9 = np.array(uy_9)
    uz_9 = np.array(uz_9)
    rx_9 = np.array(rx_9)
    ry_9 = np.array(ry_9)
    rz_9 = np.array(rz_9)
    
    ux_10 = np.array(ux_10)
    uy_10 = np.array(uy_10)
    uz_10 = np.array(uz_10)
    rx_10 = np.array(rx_10)
    ry_10 = np.array(ry_10)
    rz_10 = np.array(rz_10)
    
    W_ext = np.array(W_ext)
    W_int = np.array(W_int)
    E_kin = np.array(E_kin)

    print(f"times and x-disps: \n{np.stack((ts_save, ux_0)).T}")
    print(f"Timing: total simulation run for {time.time() - start_time} s")

    post_analysis_data = np.array([ts_save, R_support, W_int, W_ext, E_kin,\
                                   ux_0, uy_0, uz_0, rx_0, ry_0, rz_0,\
                                   ux_1, uy_1, uz_1, rx_1, ry_1, rz_1,\
                                   ux_2, uy_2, uz_2, rx_2, ry_2, rz_2,\
                                   ux_3, uy_3, uz_3, rx_3, ry_3, rz_3,\
                                   ux_4, uy_4, uz_4, rx_4, ry_4, rz_4,\
                                   ux_5, uy_5, uz_5, rx_5, ry_5, rz_5,\
                                   ux_6, uy_6, uz_6, rx_6, ry_6, rz_6,\
                                   ux_7, uy_7, uz_7, rx_7, ry_7, rz_7,\
                                   ux_8, uy_8, uz_8, rx_8, ry_8, rz_8,\
                                   ux_9, uy_9, uz_9, rx_9, ry_9, rz_9,\
                                   ux_10, uy_10, uz_10, rx_10, ry_10, rz_10]).T
    now = datetime.datetime.now().strftime('%S%f')
    onp.save(os.path.join(numpy_dir, f'jax_{now}.npy'), post_analysis_data)
    print(f"dtype = {post_analysis_data.dtype}")


if __name__ == '__main__':
    simulation()
