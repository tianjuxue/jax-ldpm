"""
Some remarks:

Units are in SI(mm)
"""
import numpy as onp
import jax
import jax.numpy as np
import matplotlib.pyplot as plt
import meshio
import datetime

from jax_ldpm.utils import json_parse
from jax_ldpm.generate_mesh import box_mesh, save_sol
from jax_ldpm.core import *

config.update("jax_enable_x64", True)

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=15)


crt_dir = os.path.dirname(__file__)
input_dir = os.path.join(crt_dir, 'input')
output_dir = os.path.join(crt_dir, 'output')
vtk_dir = os.path.join(output_dir, 'vtk')


def bc_disp_control(N_nodes):
    dt = 1e-3
    t0 = 0.
    t1 = 0.2
    t2 = 2.4
    t3 = 4.6
    t4 = 4.8
    ns01 = round((t1 - t0)/dt)
    ns12 = round((t2 - t1)/dt)
    ns23 = round((t3 - t2)/dt)
    ns34 = round((t4 - t3)/dt)

    ns = ns01 + ns12 + ns23 + ns34 + 1

    def amp_helper(v0, v1, v2, v3, v4):
        v01 = np.linspace(v0, v1, ns01 + 1)
        v12 = np.linspace(v1, v2, ns12 + 1)
        v23 = np.linspace(v2, v3, ns23 + 1)
        v34 = np.linspace(v3, v4, ns34 + 1)
        vs = np.hstack((v01, v12[1:], v23[1:], v34[1:]))
        return vs

    def pre_compute_bc():
        inds_node = np.repeat(np.arange(N_nodes), 12) # [0, 0, ..., 0, 1, 1..., 1, ...]
        inds_var = np.tile(np.arange(12), N_nodes) # [0, 1, 2, ..., 11, 0, 1, 2, ..., 11, ...]
        amp1 = amp_helper(0., 0.02, -0.2, 0.02, 0.)
        amp2 = amp_helper(0., 0.01632993162, -0.1632993162, 0.01632993162, 0.)
        amp3 = amp_helper(0., -0.00942809042, 0.0942809042, -0.00942809042, 0.)
        amp4 = amp_helper(0., -0.00666666667, 0.0666666667, -0.00666666667, 0.)
        amp5 = amp_helper(0., 0.01885618083, -0.1885618083, 0.01885618083, 0.)
        return inds_node, inds_var, amp1, amp2, amp3, amp4, amp5

    return pre_compute_bc, [dt, [t0, t1, t2, t3, t4], ns]


def bc_rot_control(N_nodes):
    dt = 1e-3
    t0 = 0.
    t1 = 0.5
    t2 = 1.5
    t3 = 2.5
    t4 = 3.
    ns01 = round((t1 - t0)/dt)
    ns12 = round((t2 - t1)/dt)
    ns23 = round((t3 - t2)/dt)
    ns34 = round((t4 - t3)/dt)

    ns = ns01 + ns12 + ns23 + ns34 + 1

    def amp_helper(v0, v1, v2, v3, v4):
        v01 = np.linspace(v0, v1, ns01 + 1)
        v12 = np.linspace(v1, v2, ns12 + 1)
        v23 = np.linspace(v2, v3, ns23 + 1)
        v34 = np.linspace(v3, v4, ns34 + 1)
        vs = np.hstack((v01, v12[1:], v23[1:], v34[1:]))
        return vs

    def pre_compute_bc():
        inds_node = np.repeat(np.arange(N_nodes), 12) # [0, 0, ..., 0, 1, 1..., 1, ...]
        inds_var = np.tile(np.arange(12), N_nodes) # [0, 1, 2, ..., 11, 0, 1, 2, ..., 11, ...]
        amp1 = amp_helper(0., 0.2, -0.2, 0.2, 0.)
        return inds_node, inds_var, amp1

    return pre_compute_bc, [dt, [t0, t1, t2, t3, t4], ns]


@jax.jit
def crt_bc_case1(step, inds_node, inds_var, amp1, amp2, amp3, amp4, amp5):
    a1 = amp1[step]
    a2 = amp2[step]
    a3 = amp3[step]
    a4 = amp4[step]
    a5 = amp5[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([a1, 0., 0., 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case2(step, inds_node, inds_var, amp1, amp2, amp3, amp4, amp5):
    a1 = amp1[step]
    a2 = amp2[step]
    a3 = amp3[step]
    a4 = amp4[step]
    a5 = amp5[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., a1, 0., 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case3(step, inds_node, inds_var, amp1, amp2, amp3, amp4, amp5):
    a1 = amp1[step]
    a2 = amp2[step]
    a3 = amp3[step]
    a4 = amp4[step]
    a5 = amp5[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., a1, 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case4(step, inds_node, inds_var, amp1, amp2, amp3, amp4, amp5):
    a1 = amp1[step]
    a2 = amp2[step]
    a3 = amp3[step]
    a4 = amp4[step]
    a5 = amp5[step]
    node1_bc_vals = np.hstack((np.array([-a2, a3, a4, 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([a2,  a3, a4, 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0.,  a5, a4, 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0.,  0., a1, 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case5(step, inds_node, inds_var, amp1):
    a1 = amp1[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([0., 0., 0., a1, 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case6(step, inds_node, inds_var, amp1):
    a1 = amp1[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., 0., 0., 0., a1, 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


@jax.jit
def crt_bc_case7(step, inds_node, inds_var, amp1):
    a1 = amp1[step]
    node1_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node2_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node3_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., 0.]), np.zeros(6)))
    node4_bc_vals = np.hstack((np.array([0., 0., 0., 0., 0., a1]), np.zeros(6)))
    bc_vals = np.hstack((node1_bc_vals, node2_bc_vals, node3_bc_vals, node4_bc_vals))
    return inds_node, inds_var, bc_vals


def save_sol_helper(step, tet_points, tet_cells, points, bundled_info, state):
    vtk_path = os.path.join(output_dir, f'vtk/u_{step:05d}.vtu')
    tets_u = process_tet_point_sol(points, bundled_info, state)
    stv = np.vstack((bundled_info['stv'], bundled_info['stv']))
    v = np.vstack((state[bundled_info['ind_i'], 6:9], state[bundled_info['ind_j'], 6:9]))
    save_sol(tet_points, tet_cells, vtk_path, point_infos=[('u', tets_u)], cell_infos=[('stv', stv), ('v', v)])


def save_stress_strain_data(strains, expected_s, predicted_s):
    stress_strain_data = np.stack((strains, expected_s, predicted_s))
    os.makedirs(numpy_dir, exist_ok=True)
    now = datetime.datetime.now().strftime('%s%f')
    print(f"Saving stress_strain_{now}.npy to local directory")
    onp.save(os.path.join(numpy_dir, f'stress_strain_{now}.npy'), stress_strain_data)


def simulation(bc_control, crt_bc, freecad_dir, numpy_dir, flag_tet, flag_case):
    """
    Case 1-7 have different bc_control, crt_bc, 
    regular/irregular tet have different freecad_dir
    """
    files = glob.glob(os.path.join(vtk_dir, f'*'))
    for f in files:
        os.remove(f)

    json_file = os.path.join(input_dir, 'json/params.json')
    params = json_parse(json_file)
 
    Young_mod = (2. + 3.*params['alpha'])/(4. + params['alpha'])*params['E0']

    if flag_tet == 'regular':
        tmp = 'Reg'
    else:
        tmp = 'Irreg'

    facet_data = onp.genfromtxt(os.path.join(freecad_dir, 'LDPM_debug' + tmp + 'Tet-data-facets.dat'), dtype=float)
    facet_vertices = onp.genfromtxt(os.path.join(freecad_dir, 'LDPM_debug' + tmp + 'Tet-data-facetsVertices.dat'), dtype=float)
    meshio_mesh = meshio.read(os.path.join(freecad_dir, 'LDPM_debug' + tmp + 'Tet-para-mesh.000.vtk'))
    cell_type = 'tetra'
    points, cells = onp.array(meshio_mesh.points), onp.array(meshio_mesh.cells_dict[cell_type])
    params['cells'] = cells
    params['points'] = points

    N_nodes = len(points)
    print(f"Num of nodes = {N_nodes}")
    vtk_path = os.path.join(vtk_dir, f'mesh.vtu')
    save_sol(points, cells, vtk_path)

    bundled_info, tet_cells, tet_points = split_tets(cells, points, facet_data.reshape(len(cells), 12, -1), facet_vertices, params)
    node_true_ms, node_lumped_ms = compute_mass(N_nodes, bundled_info, params)
    state = np.zeros((len(points), 12))

    pre_compute_bc, t_info = bc_control(N_nodes)
    dt, ti_tf, ns = t_info
    params['dt'] = dt

    bc_args = pre_compute_bc()

    ts = np.linspace(ti_tf[0], ti_tf[-1], ns)

    assert len(ts) == ns, f"len(ts) = {len(ts)}, should be = {ns}"

    svar_data = []
    t_data = []
    epsV_data = []
    reactions = []
    save_sol_helper(0, tet_points, tet_cells, points, bundled_info, state)
    for i in range(len(ts[1:])):

        crt_t = ts[i + 1]
        inds_node, inds_var, bc_vals = crt_bc(i + 1, *bc_args)
        state = apply_bc(state, inds_node, inds_var, bc_vals)

        state, bundled_info = leapfrog(state, rhs_func, bundled_info, node_true_ms, params)


        if (i + 1) % 1 == 0:
            print(f"\nStep {i + 1}, t = {(i + 1)*dt}")

            node_Fs = compute_node_reactions(state, bundled_info, params)

            state = apply_bc(state, inds_node, inds_var, bc_vals)
            save_sol_helper(i + 1, tet_points, tet_cells, points, bundled_info, state)

            print(f"aLength = \n{bundled_info['edge_l']}")

            print(f"{bundled_info['stv'][0, 4]}")

            svar_d = onp.array(bundled_info['stv'][:, 1:7])
            epsV = bundled_info['stv'][:, 16]
            print(f"bundled_info['stv'] = \n{svar_d}")
            print(f"epsV = \n{epsV}")
            print(f"node_Fs = \n{node_Fs}")
            # print(f"state =\n{state}")

            svar_data.append(svar_d)
            t_data.append(crt_t)
            epsV_data.append(epsV)
            reactions.append(node_Fs)

    onp.save(os.path.join(numpy_dir, 'svar_data.npy'), onp.stack(svar_data))
    onp.save(os.path.join(numpy_dir, 'ts.npy'), onp.stack(t_data))
    onp.save(os.path.join(numpy_dir, 'epsV.npy'), onp.stack(epsV_data))
    onp.save(os.path.join(numpy_dir, 'reactions.npy'), onp.stack(reactions))


def driver():
    flags_tet = ['regular', 'irregular']
    flags_case = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7']
    bc_controls =[bc_disp_control]*4 + [bc_rot_control]*3
    crt_bcs = [crt_bc_case1, crt_bc_case2, crt_bc_case3, crt_bc_case4, crt_bc_case5, crt_bc_case6, crt_bc_case7]

    # flags_tet = ['regular']
    # flags_case = ['case4']
    # bc_controls =[bc_disp_control]
    # crt_bcs = [crt_bc_case4]

    for i in range(len(flags_tet)):
        for j in range(len(flags_case)):
            freecad_dir = os.path.join(input_dir, 'freecad', flags_tet[i])
            numpy_dir = os.path.join(output_dir, 'numpy', flags_tet[i], flags_case[j])
            simulation(bc_controls[j], crt_bcs[j], freecad_dir, numpy_dir, flags_tet[i], flags_case[j])

if __name__ == '__main__':
    driver()
