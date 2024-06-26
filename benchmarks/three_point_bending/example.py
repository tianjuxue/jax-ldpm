import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio
import datetime
import time
import os 

# from jax_fem.solver import solver
from jax_fem.utils import read_abaqus_and_write_vtk
from jax_fem.utils import save_sol as fem_save_sol

from jax_ldpm.utils import json_parse
from jax_ldpm.generate_mesh import box_mesh
from jax_ldpm.generate_mesh import save_sol
from jax_ldpm.core import *

from benchmarks.three_point_bending.fem_problems import create_fe_problems, get_mass, get_explicit_dynamics
from benchmarks.three_point_bending.interpolation import ldpm_to_fem_mass,ldpm_to_fem_force, fem_to_ldpm_disp, get_transformation_matrix


# from jax import config
# config.update("jax_enable_x64", False)
jax.config.update("jax_enable_x64", False)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
os.makedirs(vtk_dir, exist_ok=True)
numpy_dir = os.path.join(output_dir, 'numpy')
os.makedirs(numpy_dir, exist_ok=True)
freecad_dir = os.path.join(input_dir, 'freecad/LDPM_mesh_8-16')
abaqus_dir = os.path.join(input_dir, 'abaqus')

"""
Unit system: [mm], [N], [s]
"""

# Abaqus uses 
# time_to_vel = 0.002
# vel = 5.
# dt = 1e-7

vel = 15.
dt = 1e-7

# vel = 10.
# dt = 2*1e-7

time_to_vel = 0.002
acc = vel/time_to_vel


def bc_tensile_disp_control_fix(points, Lx, Lz):
    steel_Lx = 25.
    flag1 = points[:, 2] > Lz - 1e-5
    flag2 = points[:, 0] >= Lx/2. - steel_Lx/2.
    flag3 = points[:, 0] <= Lx/2. + steel_Lx/2.

    flag = flag1 & flag2 & flag3
    # flag = flag1

    top_inds_node = onp.argwhere(flag).reshape(-1)

    def pre_compute_bc():
        N_tp_ind = len(top_inds_node)
        top_inds_tiled = np.tile(top_inds_node, 12) 
        inds_node = top_inds_tiled
        top_inds_var = np.repeat(np.arange(12), N_tp_ind)
        inds_var = top_inds_var
        return N_tp_ind, inds_node, inds_var

    @partial(jax.jit, static_argnums=(1,))
    def crt_bc(step, N_tp_ind, inds_node, inds_var):
        crt_time = step*dt
        z1 = 1./2.*acc*crt_time**2
        z2 = 1./2.*acc*time_to_vel**2 + vel*(crt_time - time_to_vel)
        bc_z_val = -np.where(acc*crt_time < vel, z1, z2)
        bc_z_vel = -np.where(acc*crt_time < vel, acc*crt_time, vel)

        # bc_z_val = np.maximum(bc_z_val, -0.05)

        # jax.debug.print("disp = {x}", x=bc_z_val)

        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*3), \
                              np.zeros(N_tp_ind*2), bc_z_vel*np.ones(N_tp_ind), np.zeros(N_tp_ind*3)))
        bc_vals = top_vals
        return inds_node, inds_var, bc_vals, bc_z_val, bc_z_vel

    @jax.jit
    def calc_forces_top(node_forces):
        forces_top = node_forces[top_inds_node]
        return -np.sum(forces_top, axis=0)

    return pre_compute_bc, crt_bc, calc_forces_top


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

    abaqus_file = os.path.join(abaqus_dir, 'FEA_mesh.inp')
    vtk_file = os.path.join(vtk_dir, 'FEA_mesh.vtu')
    read_abaqus_and_write_vtk(abaqus_file, vtk_file)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
 
    params['dt'] = dt

    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPMgeo000-data-facetsVertices.dat'), dtype=float)
    meshio_mesh = meshio.read(os.path.join(freecad_dir, 'LDPMgeo000-para-mesh.000.vtk'))
    cell_type = 'tetra'
    points, cells = onp.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])

    Lx = np.max(points[:, 0]) - np.min(points[:, 0])
    Ly = np.max(points[:, 1]) - np.min(points[:, 1])
    Lz = np.max(points[:, 2]) - np.min(points[:, 2])

    params['cells'] = cells
    params['points'] = points

    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'LDPM_mesh.vtu')
    save_sol(points, cells, vtk_path)

    problem_left_block, problem_right_block, problem_left_block_mass, problem_right_block_mass = create_fe_problems()
    fem_interface_inds_left = problem_left_block.fe.node_inds_list[-1]
    fem_interface_points_yz_left = problem_left_block.fe.points[fem_interface_inds_left][:, 1:3]
    fem_support_inds_left = problem_left_block.fe.node_inds_list[0]
    fem_interface_inds_right = problem_right_block.fe.node_inds_list[-1]
    fem_interface_points_yz_right = problem_right_block.fe.points[fem_interface_inds_right][:, 1:3]
    fem_support_inds_right = problem_right_block.fe.node_inds_list[0]

    fem_support_inds_single_left = onp.argwhere((problem_left_block.fe.points[:, 1] < 1e-5) & \
                                                (problem_left_block.fe.points[:, 2] < 1e-5) & \
                                                (problem_left_block.fe.points[:, 0] < -525.5 + 1e-5)).reshape(-1)

    fem_support_inds_single_right = onp.argwhere((problem_right_block.fe.points[:, 1] < 1e-5) & \
                                                 (problem_right_block.fe.points[:, 2] < 1e-5) & \
                                                 (problem_right_block.fe.points[:, 0] > 625.5 - 1e-5)).reshape(-1)



    ldpm_node_inds_left = onp.argwhere(points[:, 0] < 1e-5).reshape(-1)
    ldpm_points_yz_left = points[ldpm_node_inds_left][:, 1:3]
    ldpm_node_inds_right = onp.argwhere(points[:, 0] > Lx - 1e-5).reshape(-1)
    ldpm_points_yz_right = points[ldpm_node_inds_right][:, 1:3]

    ldpm_node_inds_CMOD = onp.argwhere((points[:, 1] < 1e-5) & \
                                       (points[:, 2] < 1e-5) & \
                                       (points[:, 0] > 48. - 1e-5) & 
                                       (points[:, 0] < 52. + 1e-5)).reshape(-1)


    fem_bc_values_left = np.zeros(len(np.hstack(problem_left_block.fe.node_inds_list[:1])))
    fem_inds_node_left = np.hstack(problem_left_block.fe.node_inds_list[:1])
    fem_inds_var_left = np.hstack(problem_left_block.fe.vec_inds_list[:1])
    fem_bc_values_right = np.zeros(len(np.hstack(problem_right_block.fe.node_inds_list[:1])))
    fem_inds_node_right = np.hstack(problem_right_block.fe.node_inds_list[:1])
    fem_inds_var_right = np.hstack(problem_right_block.fe.vec_inds_list[:1])

    T_mat_left = get_transformation_matrix(ldpm_points_yz_left, fem_interface_points_yz_left)
    T_mat_right = get_transformation_matrix(ldpm_points_yz_right, fem_interface_points_yz_right)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)

    ldpm_mass_to_add_left = node_lumped_ms[ldpm_node_inds_left, :3]
    fem_added_mass_left = ldpm_to_fem_mass(T_mat_left, ldpm_mass_to_add_left)
    ldpm_mass_to_add_right = node_lumped_ms[ldpm_node_inds_right, :3]
    fem_added_mass_right = ldpm_to_fem_mass(T_mat_right, ldpm_mass_to_add_right)

    added_mass_left = np.zeros((len(problem_left_block.fe.points), 3))
    added_mass_left = added_mass_left.at[fem_interface_inds_left].add(fem_added_mass_left)
    added_mass_right = np.zeros((len(problem_right_block.fe.points), 3))
    added_mass_right = added_mass_right.at[fem_interface_inds_right].add(fem_added_mass_right)

    fem_leapfrog_left, fem_apply_bc_left, fem_kinetic_energy_left \
    = get_explicit_dynamics(problem_left_block, problem_left_block_mass, added_mass_left)
    fem_leapfrog_right, fem_apply_bc_right, fem_kinetic_energy_right \
     = get_explicit_dynamics(problem_right_block, problem_right_block_mass, added_mass_right)

    pre_compute_bc, crt_bc, calc_forces_top = bc_tensile_disp_control_fix(points, Lx, Lz)
    bc_args = pre_compute_bc()

    # ts = np.arange(0., dt*2010001, dt)
    ts = np.arange(0., dt*1010001, dt)
    # ts = np.arange(0., dt*510000, dt)

    disps = [0.]
    forces = [0.]

    state = np.zeros((len(points), 12))
    applied_reactions = np.zeros((len(points), 6))
    fem_state_left = np.zeros((problem_left_block.fe.num_total_nodes, 6))
    fem_state_right = np.zeros((problem_right_block.fe.num_total_nodes, 6))
    fem_state_prev_left = fem_state_left
    fem_state_prev_right = fem_state_right
    bc_z_val = 0.
    bc_z_vel = 0.
    fem_rhs_force_left = np.zeros((problem_left_block.fe.num_total_nodes, 3))
    fem_rhs_force_right = np.zeros((problem_right_block.fe.num_total_nodes, 3))
    ldpm_node_reactions = np.zeros((len(points), 6))

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    fem_zero_sol = np.zeros((problem_left_block.fe.num_total_nodes, problem_left_block.fe.vec))
    fem_save_sol(problem_left_block.fe, fem_zero_sol, os.path.join(output_dir, f'vtk/ul_{0:05d}.vtu'))
    fem_save_sol(problem_right_block.fe, fem_zero_sol, os.path.join(output_dir, f'vtk/ur_{0:05d}.vtu'))


    @jax.jit
    def tie_disp(fem_state_left, fem_state_right, state):
        fem_disp_left_block = fem_state_left[fem_interface_inds_left, :3]
        ldpm_disp_left = fem_to_ldpm_disp(T_mat_left, fem_disp_left_block)
        fem_disp_right_block = fem_state_right[fem_interface_inds_right, :3]
        ldpm_disp_right = fem_to_ldpm_disp(T_mat_right, fem_disp_right_block)

        fem_vel_left_block = fem_state_left[fem_interface_inds_left, 3:]
        ldpm_vel_left = fem_to_ldpm_disp(T_mat_left, fem_vel_left_block)
        fem_vel_right_block = fem_state_right[fem_interface_inds_right, :3]
        ldpm_vel_right = fem_to_ldpm_disp(T_mat_right, fem_vel_right_block)

        state = state.at[ldpm_node_inds_left, 0:3].set(ldpm_disp_left)
        state = state.at[ldpm_node_inds_right, 0:3].set(ldpm_disp_right)
        state = state.at[ldpm_node_inds_left, 6:9].set(ldpm_vel_left)
        state = state.at[ldpm_node_inds_right, 6:9].set(ldpm_vel_right)

        return state

    @jax.jit
    def tie_force(fem_state_left, fem_state_right, ldpm_node_reactions):
        force_on_ldpm_left = ldpm_node_reactions[ldpm_node_inds_left, :3]
        force_on_fem_left = ldpm_to_fem_force(T_mat_left, force_on_ldpm_left)
        force_on_ldpm_right = ldpm_node_reactions[ldpm_node_inds_right, :3]
        force_on_fem_right = ldpm_to_fem_force(T_mat_right, force_on_ldpm_right)

        fem_applied_reactions_left = np.zeros((fem_state_left.shape[0], 3))
        fem_applied_reactions_right = np.zeros((fem_state_right.shape[0], 3))
        fem_applied_reactions_left = fem_applied_reactions_left.at[fem_interface_inds_left].add(force_on_fem_left)
        fem_applied_reactions_right = fem_applied_reactions_right.at[fem_interface_inds_right].add(force_on_fem_right)

        return fem_applied_reactions_left, fem_applied_reactions_right

    @jax.jit
    def energy_and_work(fem_state_left, fem_state_right, state, fem_rhs_force_left, fem_rhs_force_right, ldpm_node_reactions, bc_z_vel):
        node_forces = ldpm_node_reactions[:, :3]
        force = calc_forces_top(node_forces)[-1]
        dW_ext = force*bc_z_vel*dt

        ldpm_node_reactions = ldpm_node_reactions.at[ldpm_node_inds_left, :3].set(0.)
        ldpm_node_reactions = ldpm_node_reactions.at[ldpm_node_inds_right, :3].set(0.)
        dW_int_fem_left = np.sum(fem_rhs_force_left*fem_state_left[:, 3:]*dt) # time (n - 1) and (n - 1/2)
        dW_int_fem_right = np.sum(fem_rhs_force_right*fem_state_right[:, 3:]*dt) # time (n - 1) and (n - 1/2)
        dW_int_ldpm = np.sum(ldpm_node_reactions*state[:, 6:]*dt) # time (n - 1) and (n - 1/2)
        dW_int = dW_int_fem_left + dW_int_fem_right + dW_int_ldpm

        E_k_fem_left = fem_kinetic_energy_left(fem_state_left)
        E_k_fem_right = fem_kinetic_energy_right(fem_state_right)
        E_k_ldpm = 0.5*state[:, 6:]*node_lumped_ms*state[:, 6:]
        E_k_ldpm = E_k_ldpm.at[ldpm_node_inds_left, :3].set(0.)
        E_k_ldpm = E_k_ldpm.at[ldpm_node_inds_right, :3].set(0.)
        E_k_ldpm = np.sum(E_k_ldpm)
        
        E_k = E_k_fem_left + E_k_fem_right + E_k_ldpm

        return dW_ext, dW_int, E_k

    ts_save = [0.]
    bc_z_vals = [0.]
    P_top = [0.]
    R_support = [0.]
    CMOD = [0.]
    v1 = [0.]
    v2 = [0.]
    W_ext = [0.]
    W_int = [0.]
    E_kin = [0.]

    W_external = [0.]
    W_internal = [0.]

    start_time = time.time()
    for i in range(len(ts[1:])):

        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in {len(ts[1:])}, t = {ts[i + 1]}, disp = {bc_z_val}, acceleration needs {time_to_vel/dt} step")

        # crt_wall_time = time.time()

        inds_node, inds_var, bc_vals, bc_z_val, bc_z_vel = crt_bc(i + 1, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        fem_state_left = fem_apply_bc_left(fem_state_left, fem_inds_node_left, fem_inds_var_left, fem_bc_values_left, fem_state_prev_left, dt)
        fem_state_right = fem_apply_bc_right(fem_state_right, fem_inds_node_right, fem_inds_var_right, fem_bc_values_right, fem_state_prev_right, dt)

        state = tie_disp(fem_state_left, fem_state_right, state)
 
        # compute energy and work
        dW_ext, dW_int, E_k = energy_and_work(fem_state_left, fem_state_right, state, fem_rhs_force_left, fem_rhs_force_right, ldpm_node_reactions, bc_z_vel)
        W_external.append(W_external[-1] + dW_ext)
        W_internal.append(W_internal[-1] + dW_int)
      
        state_prev = state
        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)
        ldpm_node_reactions = compute_node_reactions(state, bundled_info, params)
        fem_applied_reactions_left, fem_applied_reactions_right = tie_force(fem_state_left, fem_state_right, ldpm_node_reactions)

        fem_state_prev_left = fem_state_left
        fem_state_prev_right = fem_state_right
        fem_state_left, fem_rhs_force_left = fem_leapfrog_left(fem_state_left, fem_applied_reactions_left, dt)
        fem_state_right, fem_rhs_force_right = fem_leapfrog_right(fem_state_right, fem_applied_reactions_right, dt)

        # print(f"Timing4: FEM left solve used {time.time() - crt_wall_time}")
        # crt_wall_time = time.time()

        if (i + 1) % 1000 == 0:
            print(f"Debug: fem vel max  = {np.max(fem_state_right[:, 3:6])}, min = {np.min(fem_state_right[:, 3:6])}")
            print(f"Debug: ldpm vel max  = {np.max(state[:, 6:9])}, min = {np.min(state[:, 6:9])}")
            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            P = calc_forces_top(node_forces)[-1]
            R = np.sum(fem_rhs_force_left[fem_support_inds_left, 2]) + np.sum(fem_rhs_force_right[fem_support_inds_right, 2])
            print(f"Debug: force on top surface = {P}, support reaction = {R}")
            print(f"W_ext = {W_external[-1]}, W_int = {W_internal[-1]}, E_k = {E_k}")

            ts_save.append(ts[i + 1])
            bc_z_vals.append(-bc_z_val)
            P_top.append(-P)
            R_support.append(-R)
            CMOD.append(state[ldpm_node_inds_CMOD[0], 0] - state[ldpm_node_inds_CMOD[1], 0])
            v1.append(-fem_state_left[fem_support_inds_single_left[0], 0])
            v2.append(fem_state_right[fem_support_inds_single_right[0], 0])
            W_ext.append(W_external[-1])
            W_int.append(-W_internal[-1])
            E_kin.append(E_k)

        if (i + 1) % 10000 == 0:
            fem_save_sol(problem_left_block.fe, fem_state_left[:, :3], os.path.join(output_dir, f'vtk/ul_{i + 1:05d}.vtu'))
            fem_save_sol(problem_right_block.fe, fem_state_right[:, :3], os.path.join(output_dir, f'vtk/ur_{i + 1:05d}.vtu'))
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)

            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            force = calc_forces_top(node_forces)[-1]
            disp = ts[i + 1]*vel
            disps.append(disp)
            forces.append(force)

    disps = np.array(disps)
    forces = np.array(forces)
    print(f"disps and forces: \n{np.stack((disps, forces)).T}")
    print(f"Timing: total simulation run for {time.time() - start_time} s")

    post_analysis_data = np.array([ts_save, bc_z_vals, P_top, R_support, CMOD, v1, v2, W_ext, W_int, E_kin]).T
    # print(post_analysis_data)
    now = datetime.datetime.now().strftime('%s%f')
    onp.save(os.path.join(numpy_dir, f'jax_{now}.npy'), post_analysis_data)
    print(f"dtype = {post_analysis_data.dtype}")


if __name__ == '__main__':
    simulation()
