import jax
import jax.numpy as np
import numpy as onp
import scipy
import meshio
import os
import glob
from functools import partial
import matplotlib.pyplot as plt

from jax_fem.problem import Problem
# from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh, box_mesh, get_meshio_cell_type
from jax_fem.utils import save_sol

from benchmarks.three_point_bending.fem_models import LinearElasticityMass, LinearElasticity


# from jax.config import config
# config.update("jax_enable_x64", True)


os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def create_fe_problems():
    input_dir = os.path.join(os.path.dirname(__file__), 'input')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')

    ele_type = 'TET4'
    cell_type = get_meshio_cell_type(ele_type)
 
    mesh_file = os.path.join(input_dir, f"abaqus/FEA_mesh.inp")
    meshio_mesh = meshio.read(mesh_file)

    meshio_mesh.points[:, 0] = meshio_mesh.points[:, 0] - onp.min(meshio_mesh.points[:, 0])
    meshio_mesh.points[:, 1] = meshio_mesh.points[:, 1] - onp.min(meshio_mesh.points[:, 1])
    meshio_mesh.points[:, 2] = meshio_mesh.points[:, 2] - onp.min(meshio_mesh.points[:, 2])

    Lx = onp.max(meshio_mesh.points[:, 0]) - onp.min(meshio_mesh.points[:, 0])
    Ly = onp.max(meshio_mesh.points[:, 1]) - onp.min(meshio_mesh.points[:, 1])
    Lz = onp.max(meshio_mesh.points[:, 2]) - onp.min(meshio_mesh.points[:, 2])
    print(f"Lx = {Lx}, Ly = {Ly}, Lz = {Lz}")

    middle_part_Lx = 100.

    move_vec_left = onp.array([-Lx, 0., 0.])
    move_vec_right = onp.array([middle_part_Lx, 0., 0.])

    left_block_mesh = Mesh(meshio_mesh.points + move_vec_left[None, :], meshio_mesh.cells_dict[cell_type])
    right_block_mesh = Mesh(meshio_mesh.points + move_vec_right[None, :], meshio_mesh.cells_dict[cell_type])

    def left_block_fix_side(point):
        # flagx = point[0] <= -Lx + 25.
        flagx = point[0] <= -Lx + 1e-3

        flagz = np.isclose(point[2], 0., atol=1e-5)
        return flagx & flagz

    def left_block_right_side(point):
        return np.isclose(point[0], 0., atol=1e-5)

    def right_block_fix_side(point):
        # flagx = point[0] >=  Lx + middle_part_Lx - 25.
        flagx = point[0] >=  Lx + middle_part_Lx - 1e-3

        flagz = np.isclose(point[2], 0., atol=1e-5)
        return flagx & flagz

    def right_block_left_side(point):
        return np.isclose(point[0], middle_part_Lx, atol=1e-5)

    def zero_dirichlet_val(point):
        return 0.

    # dirichlet_bc_info_left_block = [[left_block_fix_side]*3 + [left_block_right_side]*3, 
    #                                 [0, 1, 2, 0, 1, 2], 
    #                                 [zero_dirichlet_val]*6]
 
    # dirichlet_bc_info_right_block = [[right_block_fix_side]*3 + [right_block_left_side]*3, 
    #                                  [0, 1, 2, 0, 1, 2], 
    #                                  [zero_dirichlet_val]*6]


    dirichlet_bc_info_left_block = [[left_block_fix_side] + [left_block_right_side]*3, 
                                    [2, 0, 1, 2], 
                                    [zero_dirichlet_val]*4]
 
    dirichlet_bc_info_right_block = [[right_block_fix_side] + [right_block_left_side]*3, 
                                     [2, 0, 1, 2], 
                                     [zero_dirichlet_val]*4]


    problem_left_block = LinearElasticity(left_block_mesh, vec=3, dim=3, ele_type=ele_type, gauss_order=1,
        dirichlet_bc_info=dirichlet_bc_info_left_block, additional_info=[onp.array([1., 0., 0.])])
    problem_right_block = LinearElasticity(right_block_mesh, vec=3, dim=3, ele_type=ele_type, gauss_order=1,
        dirichlet_bc_info=dirichlet_bc_info_right_block, additional_info=[onp.array([-1., 0., 0.])])

    problem_left_block_mass = LinearElasticityMass(left_block_mesh, vec=3, dim=3, ele_type=ele_type)
    problem_right_block_mass = LinearElasticityMass(right_block_mesh, vec=3, dim=3, ele_type=ele_type)

    return problem_left_block, problem_right_block, problem_left_block_mass, problem_right_block_mass


def get_mass(problem_mass):
    dofs = np.zeros(problem_mass.num_total_dofs_all_vars)
    sol_list = problem_mass.unflatten_fn_sol_list(dofs)
    problem_mass.newton_update(sol_list)
    A_sp_scipy = scipy.sparse.csr_array((onp.array(problem_mass.V), (problem_mass.I, problem_mass.J)),
        shape=(problem_mass.num_total_dofs_all_vars, problem_mass.num_total_dofs_all_vars))
    M = A_sp_scipy.sum(axis=1)
    M = M.reshape((problem_mass.fe.num_total_nodes, problem_mass.fe.vec))
    return M


def get_explicit_dynamics(problem, problem_mass, added_mass=None):
    M = get_mass(problem_mass)

    if added_mass is not None:
        M += added_mass

    M_inv = 1./M

    def force_func(pos, vel, applied_force):
        """
        pos: (N, 3)
        vel: (N, 3)
        """
        sol_list = [pos]
        res = -problem.compute_residual(sol_list)[0]
        rhs_val = res + applied_force
        return rhs_val

    @jax.jit
    def fem_apply_bc(pos_vel, inds_node, inds_var, values, pos_vel_prev, dt):
        pos = pos_vel[:, :3] 
        vel = pos_vel[:, 3:]  
        pos_prev = pos_vel_prev[:, :3] 
        pos = pos.at[inds_node, inds_var].set(values)
        vel = vel.at[inds_node, inds_var].set((pos[inds_node, inds_var] - pos_prev[inds_node, inds_var])/dt)
        pos_vel = np.hstack((pos, vel))
        return pos_vel
        
    @jax.jit
    def fem_leapfrog(pos_vel, applied_force, dt):
        pos = pos_vel[:, :3] # Step n
        vel = pos_vel[:, 3:] # Step n - 1/2
        # facet_var updated through bundled_info
        # Input facet_var: Step n - 1
        # Output facet_var: Step n
        rhs_force = force_func(pos, vel, applied_force)
        rhs_val = M_inv*rhs_force
        vel += dt*rhs_val # Step n + 1/2
        pos += dt*vel # Step n + 1
        pos_vel = np.hstack((pos, vel))
        return pos_vel, rhs_force
   
    @jax.jit
    def fem_kinetic_energy(pos_vel):
        vel = pos_vel[:, 3:] # Step n - 1/2
        Ke = np.sum(1./2.*vel * M * vel)
        return Ke

    return fem_leapfrog, fem_apply_bc, fem_kinetic_energy


def debug_explicit_problems():
    """
    Unit system: [mm], [N], [s]
    """

    vtk_dir = os.path.join(os.path.dirname(__file__), 'output/vtk/debug')
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    rho = 2.338e-9
    E = 39000
    nu = 0.2
    dx = 20.

    wave_speed = onp.sqrt( (E*(1-nu)) / ((1+nu)*(1-2*nu)*rho) )

    print(f"wave_speed = {wave_speed}")

    dt_c = dx/wave_speed

    print(f"critical dt = {dt_c}")

    problem_left_block, problem_right_block, problem_left_block_mass, problem_right_block_mass = create_fe_problems()
    fem_leapfrog_left, fem_apply_bc_left, fem_nodal_force_left, fem_leapfrog_right,\
     fem_apply_bc_right, fem_nodal_force_right = explicit_problems(problem_left_block, 
        problem_right_block, problem_left_block_mass, problem_right_block_mass)
    dt = 1e-6
    fem_state_left = np.zeros((problem_left_block.fe.num_total_nodes, 6))
    fem_inds_node_left = np.hstack(problem_left_block.fe.node_inds_list[-3:])
    fem_inds_var_left = np.hstack(problem_left_block.fe.vec_inds_list[-3:])
    fem_bc_inds_left = problem_left_block.fe.node_inds_list[-1]

    vel = 1e3

    time_to_vel = 1e-5

    # time_to_vel = 1e-3

    acc = vel/time_to_vel
  
    disp = 0.
    ts = np.arange(0., dt*3001, dt)

    # ts = np.arange(0., dt*21, dt)

    forces = []
    for i in range(len(ts[1:])):
        print(f"Step {i + 1} in total {len(ts[1:])}, disp = {disp}")

        crt_time = (i + 1)*dt
        z1 = 1./2.*acc*crt_time**2
        z2 = 1./2.*acc*time_to_vel**2 + vel*(crt_time - time_to_vel)
        bc_z_val = np.where(acc*crt_time < vel, z1, z2)

        disp = bc_z_val

        # if disp < dx:
        # if i < 200:
        #     disp = vel*(i + 1)*dt

        fem_disp_left_block = np.stack((-disp*np.ones(66), np.zeros(66), np.zeros(66))).T
        # fem_state_left = fem_apply_bc_left(fem_state_left, fem_inds_node_left, fem_inds_var_left, fem_disp_left_block.reshape(-1, order='F'))
        # fem_state_left = fem_leapfrog_left(fem_state_left, dt)

        fem_state_prev_left =  fem_state_left
        applied_force = np.zeros((fem_state_left.shape[0], 3))
        fem_state_left = fem_leapfrog_left(fem_state_left, applied_force, dt)   

        values = fem_disp_left_block.reshape(-1, order='F')
        
        fem_state_left = fem_apply_bc_left(fem_state_left, fem_inds_node_left, fem_inds_var_left, values, fem_state_prev_left, dt)

        sol_list_left = [fem_state_left[:, :3]]

        
        if (i + 1) % 100 == 0:
            nodal_forces_left_internal, nodal_forces_left_inertial = fem_nodal_force_left(fem_state_left, fem_state_prev_left, dt)

            print(f"Debug: u \n{np.hstack((fem_state_left[fem_bc_inds_left][:, 0:3], fem_state_prev_left[fem_bc_inds_left][:, 0:3]))}")
            print(f"Debug: v \n{np.hstack((fem_state_left[fem_bc_inds_left][:, 3:6], fem_state_prev_left[fem_bc_inds_left][:, 3:6]))}")
            print(f"Debug: force \n{nodal_forces_left_inertial[fem_bc_inds_left]}")

            internal_force = np.sum(nodal_forces_left_internal[fem_bc_inds_left], axis=0)
            inertial_force = np.sum(nodal_forces_left_inertial[fem_bc_inds_left], axis=0)

            print(f"Debug: sum of nodal force left internal = {internal_force}")
            print(f"Debug: sum of nodal force left inertial = {inertial_force}")
            forces.append(internal_force[0])
            # save_sol(problem_left_block.fe, sol_list_left[0], os.path.join(vtk_dir, f'u_{i + 1:05d}.vtu'))

    fig = plt.figure()
    plt.plot(forces)
    plt.show()


def debug_bc_quad_force():
    problem_left_block, problem_right_block = create_fe_problem()
    fem_disp_left_block = np.stack((np.zeros(66), np.zeros(66), -0.1*np.ones(66))).T
    problem_left_block.set_params(fem_disp_left_block)
    sol_list_left = solver(problem_left_block, use_petsc=True)
    quad_forces_left = problem_left_block.compute_quad_forces(sol_list_left[0])

    print(problem_left_block.fe.points[:20])
    print(quad_forces_left)


if __name__ == "__main__":
    # problem()
    # explicit_problems()
    debug_explicit_problems()

