import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio

from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *

crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')


dt = 1e-7
vel = 1e-4
# dt = 1e-4


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
        # max_load = 1e-2
        # total_load_step = 100
        # crt_speed = (step + 1)/total_load_step*max_load

        # bc_z_val = np.minimum(crt_load, max_load)
        bc_z_val = dt*vel*step

        bottom_vals = np.zeros(N_btm_ind*12)
        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*9)))
        bc_vals = np.hstack((bottom_vals, top_vals))
        return inds_node, inds_var, bc_vals

    def calc_forces_btm(node_forces):
        forces_btm = node_forces[bottom_inds_node]
        return np.sum(forces_btm, axis=0)

    return pre_compute_bc, crt_bc, calc_forces_btm


def bc_tensile_vel_control_fix(points, Lz):

    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)

    def pre_compute_bc():
        bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
        top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)
        N_btm_ind = len(bottom_inds_node)
        N_tp_ind = len(top_inds_node)

        bottom_inds_tiled = np.tile(bottom_inds_node, 6)
        top_inds_tiled = np.tile(top_inds_node, 6) 
        inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))
        bottom_inds_var = np.repeat(np.arange(6) + 6, N_btm_ind)
        top_inds_var = np.repeat(np.arange(6) + 6, N_tp_ind)
        inds_var = np.hstack((bottom_inds_var, top_inds_var))

        return N_btm_ind, N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=(1, 2))
    def crt_bc(step, N_btm_ind, N_tp_ind, inds_node, inds_var):
        # max_load = 0.1
        max_vel = 1e-4
        total_load_step = 100
        crt_load = (step + 1)/total_load_step*max_vel

        # bc_z_val = np.minimum(crt_load, max_vel)
        bc_z_val = max_vel

        bottom_vals = np.zeros(N_btm_ind*6)
        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*3)))
        bc_vals = np.hstack((bottom_vals, top_vals))
        return inds_node, inds_var, bc_vals


    def calc_forces_btm(node_forces):
        forces_btm = node_forces[bottom_inds_node]
        return np.sum(forces_btm, axis=0)

    return pre_compute_bc, crt_bc, calc_forces_btm


def bc_tensile_disp_control_flex():
    # TODO: flexible bc for poisson ratio verification

    # bottom_inds_tiled = np.tile(bottom_inds_node, 4)
    # top_inds_tiled = np.tile(top_inds_node, 4) 
    # inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))
    # bottom_inds_var = np.repeat(np.array([2, 3, 4, 5]), N_btm_ind)
    # top_inds_var = np.repeat(np.array([2, 3, 4, 5]), N_tp_ind)
    # inds_var = np.hstack((bottom_inds_var, top_inds_var))

    # @jax.jit
    # def update_bc_val(disp):
    #     bottom_vals = np.zeros(N_btm_ind*4)
    #     top_vals = np.hstack((disp*np.ones(N_tp_ind), np.zeros(N_tp_ind*3)))
    #     values = np.hstack((bottom_vals, top_vals))
    #     return values

    pass


def save_sol_helper(step, tet_points, tet_cells, points, bundled_info, state):
    vtk_path = os.path.join(output_dir, f'vtk/u_{step:05d}.vtu')
    tets_u = process_tet_point_sol(points, bundled_info, state)
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    params = {}
    c = 300. # cement mass per volume [kg/m^3]
    w_c = 0.5 # water to cement mass ratio
    a_c = 6.5 # aggregate to cement mass ratio
    v_air = 3.5e-2 # air volume fraction
    rho_air = 1.2 # air density [kg/m^3]
    params['rho'] = c + c*w_c + c*a_c + v_air*rho_air
    params['alpha'] = 0.25
    params['E0'] = 43_748e6 # Pa
    params['damping_v'] = 1e4 # artificial damping parameter for velocity
    params['damping_w'] = 1e4 # artificial damping parameter for angular velocity
    Young_mod = (2. + 3.*params['alpha'])/(4. + params['alpha'])*params['E0']

    # Nx, Ny, Nz = 5, 5, 5
    # Lx, Ly, Lz = 1., 1., 1.
    # meshio_mesh = box_mesh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=output_dir)

    freecad_mesh_file = os.path.join(input_dir, 'freecad/LDPMgeo000-para-mesh.000.vtk')
    meshio_mesh = meshio.read(freecad_mesh_file)


    cell_type = 'tetra'
    points, cells = np.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])
    points *= 1e-3
    Lx = np.max(points[:, 0])
    Ly = np.max(points[:, 1])
    Lz = np.max(points[:, 2])


    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))
 
    pre_compute_bc, crt_bc, calc_forces_btm = bc_tensile_disp_control_fix(points, Lz)
    # pre_compute_bc, crt_bc, calc_forces_btm = bc_tensile_vel_control_fix(points, Lz)

    bc_args = pre_compute_bc()

    ts = np.arange(0., dt*5001, dt)

    disps = []
    predicted_f = []
    expected_f = []

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    for i in range(len(ts[1:])):
        print(f"Step {i}")
        state = runge_kutta_4(state, rhs_func, dt, bundled_info, node_true_ms, params)
        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals = crt_bc(i, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        if (i + 1) % 50 == 0:
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)
            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)

            node_forces = compute_node_forces(state, bundled_info, params)
            force_predicted = calc_forces_btm(node_forces)
            disp = crt_t*1e-4
            force_expected = disp/Lz*Young_mod*Lx*Ly
            print(f"force_predicted = {force_predicted}, disp = {disp}, E = {Young_mod}, force_expected = {force_expected}")
            disps.append(disp)
            predicted_f.append(force_predicted[2])
            expected_f.append(force_expected)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    print(state[:, :])

    plt.figure(figsize=(12, 9))
    plt.plot(disps, expected_f, linestyle='-', marker='o', markersize=2, linewidth=1, color='blue', label='expected')
    plt.plot(disps, predicted_f, linestyle='-', marker='s', markersize=2, linewidth=1, color='red', label='predicted')

    plt.xlabel("Displacement", fontsize=20)
    plt.ylabel("Force", fontsize=20)
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()

if __name__ == '__main__':
    simulation()
