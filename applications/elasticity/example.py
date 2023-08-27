import numpy as onp
import jax
import jax.numpy as np

from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *


data_dir = os.path.join(os.path.dirname(__file__), 'output')
vtk_dir = os.path.join(data_dir, 'vtk')


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    Nx, Ny, Nz = 5, 5, 5
    Lx, Ly, Lz = 1., 1., 1.
    meshio_mesh = box_mesh(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, data_dir=data_dir)
    cell_type = 'tetra'
    points, cells = np.array(meshio_mesh.points), np.array(meshio_mesh.cells_dict[cell_type])
    N_nodes = len(points)

    bottom_inds_node = np.argwhere(points[:, 2] < 1e-5).reshape(-1)
    top_inds_node = np.argwhere(points[:, 2] > Lz - 1e-5).reshape(-1)
    N_btm_ind = len(bottom_inds_node)
    N_tp_ind = len(top_inds_node)


    # bottom_inds_tiled = np.tile(bottom_inds_node, 4)
    # top_inds_tiled = np.tile(top_inds_node, 4) 
    # inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))
    # bottom_inds_var = np.repeat(np.array([2, 3, 4, 5]), N_btm_ind)
    # top_inds_var = np.repeat(np.array([2, 3, 4, 5]), N_tp_ind)
    # inds_var = np.hstack((bottom_inds_var, top_inds_var))

    # @jax.jit
    # def update_bc_val(disp):
    #     bottom_vals = np.zeros(N_btm_ind*4)
    #     top_vals = np.hstack((disp*np.ones(N_btm_ind), np.zeros(N_btm_ind*3)))
    #     values = np.hstack((bottom_vals, top_vals))
    #     return values


    bottom_inds_tiled = np.tile(bottom_inds_node, 12)
    top_inds_tiled = np.tile(top_inds_node, 12) 
    inds_node = np.hstack((bottom_inds_tiled, top_inds_tiled))
    bottom_inds_var = np.repeat(np.arange(12), N_btm_ind)
    top_inds_var = np.repeat(np.arange(12), N_tp_ind)
    inds_var = np.hstack((bottom_inds_var, top_inds_var))

    @jax.jit
    def update_bc_val(disp):
        bottom_vals = np.zeros(N_btm_ind*12)
        top_vals = np.hstack((np.zeros(N_btm_ind*2), disp*np.ones(N_btm_ind), np.zeros(N_btm_ind*9)))
        values = np.hstack((bottom_vals, top_vals))
        return values


    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info = compute_info_vmap(cells, points)
    bundled_info = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), bundled_info)

    tet_points = np.vstack((bundled_info['tet_points_i'].reshape(-1, 3), 
                            bundled_info['tet_points_j'].reshape(-1, 3)))
 
    tet_cells = np.arange(len(tet_points)).reshape(-1, 4)

    # qlts = check_mesh_TET4(tet_points, tet_cells)
    # print(qlts)
    
    state = np.zeros((len(points), 12))
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info)


    vtk_path = os.path.join(data_dir, f'vtk/u_{0:05d}.vtu')
    tets_u = process_tet_point_sol(points, bundled_info, state)
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

    # t_total = 0.01
    dt = 1e-4
    ts = np.arange(0., dt*1001, dt)
    for i in range(len(ts[1:])):
        print(f"Step {i}")
        dt = ts[i + 1] - ts[i]
        state = runge_kutta_4(state, rhs_func, dt, bundled_info, node_true_ms)

        crt_t = ts[i + 1]
        max_load = 0.1
        total_load_step = 100
        crt_load = (i + 1)/total_load_step*max_load

        bc_z_val = np.minimum(crt_load, max_load)

        bc_vals = update_bc_val(bc_z_val)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        if (i + 1) % 10 == 0:
            vtk_path = os.path.join(data_dir, f'vtk/u_{i + 1:05d}.vtu')
            tets_u = process_tet_point_sol(points, bundled_info, state)
            save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

            ee = compute_elastic_energy(state, bundled_info)
            ke = compute_kinetic_energy(state, node_true_ms)
            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")

    print(state[:, :])

    # print(np.sum(mass, axis=0))
    # rhs_vals = rhs_func(initial_state, bundled_info, mass)
    # tets_u = process_tet_point_sol(points, bundled_info, initial_state)

    # save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

    # print(rhs_vals)

    # tets_points = bundled_info['ind_i']


if __name__ == '__main__':
    simulation()
