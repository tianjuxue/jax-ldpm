import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio

from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *


# onp.set_printoptions(threshold=sys.maxsize,
#                      linewidth=1000,
#                      suppress=True,
#                      precision=5)


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')


vel = 1e-1
dt = 1e-7


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
    params['E0'] = 43_748e6 # [Pa]
    params['damping_v'] = 1e4 # artificial damping parameter for velocity
    params['damping_w'] = 1e4 # artificial damping parameter for angular velocity
    params['ft'] = 4.03e6 # Tensile Strength [Pa]
    params['chLen'] = 0.12 # Tensile characteristic length [m]
    params['fr'] = 2.7 # Shear strength ratio
    params['sen_c'] = 0.2 # Softening exponent
    params['fc'] = 150e6 # Compressive Yield Strength [Pa]
    params['RinHardMod'] = 0.4 # Initial hardening modulus ratio
    params['tsrn_e'] = 2. # Transitional Strain ratio
    params['dk1'] = 1. # Deviatoric strain threshold ratio
    params['dk2'] = 5. # Deviatoric damage parameter
    params['fmu_0'] = 0.2 # Initial friction
    params['fmu_inf'] = 0. # Asymptotic friction
    params['sf0'] = 600e6 # Transitional stress [Pa]
    params['DensRatio'] = 1. # Densification ratio 
    params['beta'] = 0. # Volumetric deviatoric coupling
    params['unkt'] = 0. # Tensile unloading parameter
    params['unks'] = 0. # Shear unloading parameter
    params['unkc'] = 0. # Compressive unloading parameter
    params['Hr'] = 0. # Shear softening modulus ratio
    params['dk3'] = 0.1 # Final hardening modulus ratio

    params['EAF'] = 1 # ElasticAnalysisFlag

    params['dt'] = dt


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

    params['cells'] = cells
    params['points'] = points


    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))
 
    pre_compute_bc, crt_bc, calc_forces_btm = bc_tensile_disp_control_fix(points, Lz)

    bc_args = pre_compute_bc()

    ts = np.arange(0., dt*5001, dt)

    strains = []
    predicted_s = []
    expected_s = []

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    for i in range(len(ts[1:])):
        print(f"Step {i}")

        state, bundled_info = leapfrog(state, rhs_func, dt, bundled_info, node_true_ms, params)

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals = crt_bc(i, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        # print(bundled_info['stv'])
        # exit()

        if (i + 1) % 50 == 0:
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)
            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)

            node_forces = compute_node_forces(state, bundled_info, params)
            force_predicted = calc_forces_btm(node_forces)
            stess_predicted = force_predicted[-1]/(Lx*Ly)
            disp = crt_t*vel
            strain = disp/Lz
            stress_expected = strain*Young_mod
            print(f"force_predicted = {force_predicted}, strain = {strain}, stess_predicted = {stess_predicted}, stress_expected = {stress_expected}")
            strains.append(strain)
            predicted_s.append(stess_predicted)
            expected_s.append(stress_expected)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    print(state[:, :])

    plt.figure(figsize=(12, 9))
    plt.plot(strains, expected_s, linestyle='-', marker='o', markersize=2, linewidth=1, color='blue', label='expected')
    plt.plot(strains, predicted_s, linestyle='-', marker='s', markersize=2, linewidth=1, color='red', label='predicted')

    plt.xlabel("Strain", fontsize=20)
    plt.ylabel("Stress [Pa]", fontsize=20)
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    plt.tick_params(labelsize=20)
    plt.legend(fontsize=20, frameon=False)   
    plt.show()


if __name__ == '__main__':
    simulation()
