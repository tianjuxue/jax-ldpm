import numpy as onp
import jax
import jax.numpy as np
from jax.config import config

from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *

# config.update("jax_enable_x64", True)


data_dir = os.path.join(os.path.dirname(__file__), 'output')
vtk_dir = os.path.join(data_dir, 'vtk')


def simulation():
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    params = {}
    params['rho'] = 1e3

    E = 1e7
    nu = 0.2
    E0 = 1./(1. - 2*nu)*E
    alpha = (1. - 4.*nu)/(1. + nu)

    params['alpha'] = alpha
    params['E0'] = E0 # [Pa]
    params['damping_v'] = 0. # artificial damping parameter for velocity
    params['damping_w'] = 0. # artificial damping parameter for angular velocity
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

    dt = 1e-5
    params['dt'] = dt

    points = np.array([[0., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])

    N_nodes = len(points)
    cells = np.array([[0, 1, 2, 3]])

    params['cells'] = cells
    params['points'] = points

    v_scale = 1e-1
    state = np.array([[0.01, 0.01, 0.01, 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 1., 0., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 1., 0.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 1.],
                      [0., 0., 0., 0., 0., 0., v_scale*1., v_scale*2., v_scale*3., 0., 0., 0.]])

 
    bundled_info, tet_cells, tet_points = split_tets(cells, points, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)

    ts = np.arange(0., dt*101, dt)
    for i in range(len(ts[1:])):
        print(f"Step {i}")
        state, bundled_info = leapfrog(state, rhs_func, dt, bundled_info, node_true_ms, params) # runge_kutta_4

        if i % 10 == 0:
            vtk_path = os.path.join(vtk_dir, f'u_{i:05d}.vtu')
            tets_u = process_tet_point_sol(points, bundled_info, state)
            save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)])

            ee = compute_elastic_energy(state, bundled_info, params)
            ke = compute_kinetic_energy(state, node_true_ms)

            print(f"ee = {ee}, ke = {ke}, ee + ke = {ee + ke}")
 

if __name__ == '__main__':
    simulation()
