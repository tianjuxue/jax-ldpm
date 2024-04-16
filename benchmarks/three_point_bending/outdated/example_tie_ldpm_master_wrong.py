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

from benchmarks.three_point_bending.fem_problems import create_fe_problems, explicit_problems, get_mass
from benchmarks.three_point_bending.interpolation import ldpm_to_fem, fem_to_ldpm, fem_to_ldpm_debug_mass


# from jax.config import config
# config.update("jax_enable_x64", True)

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

# Abaqus uses 
# time_to_vel = 0.002
# vel = 5.
# dt = 1e-7

vel = 10.
dt = 2*1e-7

time_to_vel = 0.002
# time_to_vel = 0.01
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

        # bc_z_val = np.maximum(bc_z_val, -0.05)

        # jax.debug.print("disp = {x}", x=bc_z_val)

        top_vals = np.hstack((np.zeros(N_tp_ind*2), bc_z_val*np.ones(N_tp_ind), np.zeros(N_tp_ind*9)))
        bc_vals = top_vals
        return inds_node, inds_var, bc_vals, bc_z_val

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
    """
    Unit system: [mm], [N], [s]
    """
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

    print(points.dtype)

    ldpm_node_inds_left = onp.argwhere(points[:, 0] < 1e-5).reshape(-1)
    ldpm_points_yz_left = points[ldpm_node_inds_left][:, 1:3]

    ldpm_node_inds_right = onp.argwhere(points[:, 0] > Lx - 1e-5).reshape(-1)
    ldpm_points_yz_right = points[ldpm_node_inds_right][:, 1:3]

    problem_left_block, problem_right_block, problem_left_block_mass, problem_right_block_mass = create_fe_problems()
    fem_interface_inds_left = problem_left_block.fe.node_inds_list[-1]
    fem_interface_points_yz_left = problem_left_block.fe.points[fem_interface_inds_left][:, 1:3]
    fem_support_inds_left = problem_left_block.fe.node_inds_list[0]
    fem_interface_inds_right = problem_right_block.fe.node_inds_list[-1]
    fem_interface_points_yz_right = problem_right_block.fe.points[fem_interface_inds_right][:, 1:3]
    fem_support_inds_right = problem_right_block.fe.node_inds_list[0]

    fem_padding_zeros_left = np.zeros(len(np.hstack(problem_left_block.fe.node_inds_list[:1])))
    fem_inds_node_left = np.hstack(problem_left_block.fe.node_inds_list)
    fem_inds_var_left = np.hstack(problem_left_block.fe.vec_inds_list)
    fem_padding_zeros_right = np.zeros(len(np.hstack(problem_right_block.fe.node_inds_list[:1])))
    fem_inds_node_right = np.hstack(problem_right_block.fe.node_inds_list)
    fem_inds_var_right = np.hstack(problem_right_block.fe.vec_inds_list)

    fem_leapfrog_left, fem_apply_bc_left, fem_nodal_force_left, fem_internal_force_left, \
    fem_leapfrog_right, fem_apply_bc_right, fem_nodal_force_right, fem_internal_force_right = explicit_problems(problem_left_block, 
        problem_right_block, problem_left_block_mass, problem_right_block_mass)


    def add_mass_to_ldpm(node_true_mass, added_mass):
        node_lumped_mass = np.diag(node_true_mass)
        ldpm_mass = node_lumped_mass.at[:3].add(added_mass)
        return np.diag(ldpm_mass)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)

    fem_interface_mass_left = get_mass(problem_left_block_mass)[fem_interface_inds_left]
    fem_interface_mass_right = get_mass(problem_left_block_mass)[fem_interface_inds_right]
    ldpm_added_mass_left = fem_to_ldpm_debug_mass(ldpm_points_yz_left, fem_interface_points_yz_left, fem_interface_mass_left)
    ldpm_added_mass_right = fem_to_ldpm_debug_mass(ldpm_points_yz_right, fem_interface_points_yz_right, fem_interface_mass_right)
    added_mass = np.zeros((len(points), 3))
    added_mass = added_mass.at[ldpm_node_inds_left].add(ldpm_added_mass_left)
    added_mass = added_mass.at[ldpm_node_inds_right].add(ldpm_added_mass_right)
    node_modified_ms = jax.vmap(add_mass_to_ldpm)(node_true_ms, added_mass)

    pre_compute_bc, crt_bc, calc_forces_top = bc_tensile_disp_control_fix(points, Lx, Lz)
    bc_args = pre_compute_bc()

    ts = np.arange(0., dt*500001, dt)
    # ts = np.arange(0., dt*101, dt)

    disps = [0.]
    forces = [0.]

    state = np.zeros((len(points), 12))
    applied_reactions = np.zeros((len(points), 6))
    fem_state_left = np.zeros((problem_left_block.fe.num_total_nodes, 6))
    fem_state_right = np.zeros((problem_right_block.fe.num_total_nodes, 6))
    fem_state_prev_left = fem_state_left
    fem_state_prev_right = fem_state_right
    bc_z_val = 0.

    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    fem_zero_sol = np.zeros((problem_left_block.fe.num_total_nodes, problem_left_block.fe.vec))
    fem_save_sol(problem_left_block.fe, fem_zero_sol, os.path.join(output_dir, f'vtk/ul_{0:05d}.vtu'))
    fem_save_sol(problem_right_block.fe, fem_zero_sol, os.path.join(output_dir, f'vtk/ur_{0:05d}.vtu'))

    start_time = time.time()
    for i in range(len(ts[1:])):

        if (i + 1) % 1000 == 0:
            print(f"\nStep {i + 1} in total {len(ts[1:])}, disp = {bc_z_val}, acceleration needs {time_to_vel/dt} step")

        crt_wall_time = time.time()

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals, bc_z_val = crt_bc(i + 1, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        ldpm_disp_left = state[ldpm_node_inds_left, :3]
        fem_disp_left_block = ldpm_to_fem(ldpm_points_yz_left, fem_interface_points_yz_left, ldpm_disp_left)
        fem_bc_values_left = np.hstack((fem_padding_zeros_left, fem_disp_left_block.reshape(-1, order='F')))
        fem_state_left = fem_apply_bc_left(fem_state_left, fem_inds_node_left, fem_inds_var_left, fem_bc_values_left, fem_state_prev_left, dt)
        nodal_forces_left_internal, nodal_forces_left_inertial = fem_nodal_force_left(fem_state_left, fem_state_prev_left, dt)

        ldpm_disp_right = state[ldpm_node_inds_right, :3]
        fem_disp_right_block = ldpm_to_fem(ldpm_points_yz_right, fem_interface_points_yz_right, ldpm_disp_right)
        fem_bc_values_right = np.hstack((fem_padding_zeros_right, fem_disp_right_block.reshape(-1, order='F')))
        fem_state_right = fem_apply_bc_right(fem_state_right, fem_inds_node_right, fem_inds_var_right, fem_bc_values_right, fem_state_prev_right, dt)
        nodal_forces_right_internal, nodal_forces_right_inertial = fem_nodal_force_right(fem_state_right, fem_state_prev_right, dt)

        force_on_fem_left = fem_internal_force_left(fem_state_left)[fem_interface_inds_left] # Should be negative
        force_on_ldpm_left = -fem_to_ldpm(ldpm_points_yz_left, fem_interface_points_yz_left, force_on_fem_left)
        force_on_fem_right = fem_internal_force_left(fem_state_right)[fem_interface_inds_right] # Should be negative
        force_on_ldpm_right = -fem_to_ldpm(ldpm_points_yz_right, fem_interface_points_yz_right, force_on_fem_right)

        applied_reactions = np.zeros((len(points), 3))
        applied_reactions = applied_reactions.at[ldpm_node_inds_left].add(force_on_ldpm_left)
        applied_reactions = applied_reactions.at[ldpm_node_inds_right].add(force_on_ldpm_right)
        applied_reactions = np.hstack((applied_reactions, np.zeros((len(points), 3))))

        fem_applied_reactions_left = np.zeros((fem_state_left.shape[0], 3))
        fem_applied_reactions_right = np.zeros((fem_state_right.shape[0], 3))
 
        state_prev = state
        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_modified_ms, params, applied_reactions)

        fem_state_prev_left = fem_state_left
        fem_state_prev_right = fem_state_right
        fem_state_left = fem_leapfrog_left(fem_state_left, fem_applied_reactions_left, dt)
        fem_state_right = fem_leapfrog_right(fem_state_right, fem_applied_reactions_right, dt)

        if (i + 1) % 1000 == 0:
            print(f"Debug: sum of nodal force left internal = {np.sum(nodal_forces_left_internal[fem_interface_inds_left], axis=0)}")
            print(f"Debug: sum of nodal force left inertial = {np.sum(nodal_forces_left_inertial[fem_interface_inds_left], axis=0)}")
            print(f"Debug: sum of nodal force left support = {np.sum(nodal_forces_left_internal[fem_support_inds_left], axis=0)}")

            print(f"Debug: sum of nodal force right internal = {np.sum(nodal_forces_right_internal[fem_interface_inds_right], axis=0)}")
            print(f"Debug: sum of nodal force right inertial = {np.sum(nodal_forces_right_inertial[fem_interface_inds_right], axis=0)}")
            print(f"Debug: sum of nodal force right support = {np.sum(nodal_forces_right_internal[fem_support_inds_right], axis=0)}")

            print(f"Debug: fem vel max  = {np.max(fem_state_right[:, 3:6])}, min = {np.min(fem_state_right[:, 3:6])}")
            print(f"Debug: ldpm vel max  = {np.max(state[:, 6:9])}, min = {np.min(state[:, 6:9])}")
            print(f"Debug: (New) sum of nodal force left internal = {np.sum(force_on_fem_left, axis=0)}")
            print(f"Debug: (New) sum of nodal force right internal = {np.sum(force_on_fem_right, axis=0)}")
            print(f"Debug: sum of applied force on ldpm = {np.sum(applied_reactions, axis=0)[:3]}")
            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            force = calc_forces_top(node_forces)
            print(f"Debug: force on top surface = {force}")

        if (i + 1) % 10000 == 0:
            fem_save_sol(problem_left_block.fe, fem_state_left[:, :3], os.path.join(output_dir, f'vtk/ul_{i + 1:05d}.vtu'))
            fem_save_sol(problem_right_block.fe, fem_state_right[:, :3], os.path.join(output_dir, f'vtk/ur_{i + 1:05d}.vtu'))
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)

            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)

            node_reactions = compute_node_reactions(state, bundled_info, params)
            node_forces = node_reactions[:, :3]
            force = calc_forces_top(node_forces)[-1]
            disp = crt_t*vel

            disps.append(disp)
            forces.append(force)
            
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    disps = np.array(disps)
    forces = np.array(forces)

    print(f"disps and forces: \n{np.stack((disps, forces)).T}")
    print(f"Timing: total simulation run for {time.time() - start_time} s")
    # save_stress_strain_data(strains, expected_s, predicted_s)


if __name__ == '__main__':
    simulation()
    # plot_stress_strain()
