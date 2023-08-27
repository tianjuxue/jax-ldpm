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

    points = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])

    N_nodes = len(points)

    cells = np.array([[0, 1, 2, 3]])

    bundled_info = compute_info_vmap(cells, points)
    bundled_info = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), bundled_info)

    tet_points = np.vstack((bundled_info['tet_points_i'].reshape(-1, 3), 
                            bundled_info['tet_points_j'].reshape(-1, 3)))
    tet_cells = np.arange(len(tet_points)).reshape(-1, 4)

    v_scale = 1e-1
    state = np.array([[0.01, 0.01, 0.01, 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 0.]])

 
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info)

    print(f"node_true_ms.shape = {node_true_ms.shape}")

    t_total = 0.001
    ts = np.linspace(0., t_total, 101)
    for i in range(len(ts[1:])):
        print(f"Step {i}")
        dt = ts[i + 1] - ts[i]
        state = runge_kutta_4(state, rhs_func, dt, bundled_info, node_true_ms) # runge_kutta_4, explicit_euler

        if i % 10 == 0:
            vtk_path = os.path.join(vtk_dir, f'u_{i:05d}.vtu')
            tets_u = process_tet_point_sol(points, bundled_info, state)
            save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

            ee = compute_elastic_energy(state, bundled_info)
            ke = compute_kinetic_energy(state, node_true_ms)

            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")
 

if __name__ == '__main__':
    simulation()