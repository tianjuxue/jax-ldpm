import numpy as onp
import jax
import jax.numpy as np
import jax.tree_util
from jax.config import config

import os
import glob
import sys
from functools import partial

from jax_ldpm.tetrahedron import tetrahedra_volumes, tetra_first_moments_helper, tetra_inertia_tensors_helper
from jax_ldpm.constitutive import stress_fn, calc_st_sKt

# onp.set_printoptions(threshold=sys.maxsize,
#                      linewidth=1000,
#                      suppress=True,
#                      precision=5)


def friction_bc(pos, vel, node_reactions):
    friciton_force = np.zeros((len(node_reactions), 2))
    return friciton_force



@partial(jax.jit, static_argnums=(1, 6))
def leapfrog(node_var, rhs_func, bundled_info, node_true_ms, params, applied_reactions=0., friction_bc=friction_bc):
    pos = node_var[:, :6] # Step n
    vel = node_var[:, 6:] # Step n - 1/2
    # facet_var updated through bundled_info
    # Input facet_var: Step n - 1
    # Output facet_var: Step n
    vel_rhs, bundled_info = rhs_func(pos, vel, bundled_info, node_true_ms, params, applied_reactions, friction_bc)
    vel += params['dt']*vel_rhs # Step n + 1/2
    pos += params['dt']*vel # Step n + 1
    node_var = np.hstack((pos, vel))
    return node_var, bundled_info


def check_mesh_TET4(points, cells):
    def quality(pts):
        p1, p2, p3, p4 = pts
        v1 = p2 - p1
        v2 = p3 - p1
        v12 = np.cross(v1, v2)
        v3 = p4 - p1
        return np.dot(v12, v3)
    qlts = jax.vmap(quality)(points[cells])
    return qlts


def projected_face_area(A, B, C, normal_1, normal_2):
    BA = B - A
    CA = C - A
    proj_BA = np.sum(BA*normal_1)*normal_1 + np.sum(BA*normal_2)*normal_2
    proj_CA = np.sum(CA*normal_1)*normal_1 + np.sum(CA*normal_2)*normal_2
    area = 0.5*np.linalg.norm(np.cross(proj_BA, proj_CA))
    return area


def get_A_matrix(x_i, x):
    return np.array([[1., 0., 0., 0., x[2] - x_i[2], x_i[1] - x[1]],
                     [0., 1., 0., x_i[2] - x[2], 0., x[0] - x_i[0]],
                     [0., 0., 1., x[1] - x_i[1], x_i[0] - x[0], 0.]])


def compute_info(cell, ft_data, points, facet_vertices, params):
    # TODO: some comments highly needed

    # TODO: very risky
    inds = np.array([[0, 1],
                     [0, 1],
                     [0, 2],
                     [0, 2],
                     [0, 3],
                     [0, 3],
                     [1, 2],
                     [1, 2],
                     [1, 3],
                     [1, 3],
                     [2, 3],
                     [2, 3]])

    inds_i = cell[inds[:, 0]]
    inds_j = cell[inds[:, 1]]


    def compute_info_helper(f_ind, ind_i, ind_j, f_data, params):
        point_i = points[ind_i]
        point_j = points[ind_j]
  
        f_v_inds = np.array(f_data[1:4], dtype=np.int32)
        tet_points_i = np.vstack((point_i[None, :], facet_vertices[f_v_inds]))
        tet_points_j = np.vstack((point_j[None, :], facet_vertices[f_v_inds]))

        edge_vec = point_j - point_i
        edge_l = np.linalg.norm(edge_vec)

        debug_normal_N = edge_vec/edge_l

        facet_centroid = f_data[6:9]
        normal_N = f_data[9:12]
        normal_M = f_data[12:15]
        normal_L = f_data[15:18]

        cell_id = np.array(f_data[0], dtype=np.int32)
        vol = f_data[4]
        proj_area = f_data[5]

        A_i = get_A_matrix(point_i, facet_centroid)
        A_j = get_A_matrix(point_j, facet_centroid)
        B_N_i = 1./edge_l * normal_N[None, :] @ A_i
        B_N_j = 1./edge_l * normal_N[None, :] @ A_j
        B_M_i = 1./edge_l * normal_M[None, :] @ A_i
        B_M_j = 1./edge_l * normal_M[None, :] @ A_j
        B_L_i = 1./edge_l * normal_L[None, :] @ A_i
        B_L_j = 1./edge_l * normal_L[None, :] @ A_j

        st, aKt = calc_st_sKt(params, edge_l)

        # jax.debug.print('edge_l = {edge_l}', edge_l=edge_l)

        bundled_info = {'tet_points_i': tet_points_i,
                        'tet_points_j': tet_points_j,
                        'edge_l': edge_l,
                        'st': st,
                        'aKt': aKt,
                        'proj_area': proj_area,
                        'normal_N': normal_N,
                        'normal_M': normal_M,
                        'normal_L': normal_L,
                        'ind_i': ind_i,
                        'ind_j': ind_j,
                        'cell_id': cell_id,
                        'B_mats': np.stack((B_N_i, B_N_j, B_M_i, B_M_j, B_L_i, B_L_j)),
                        'debug_vol': vol,
                        'debug_normal_N': debug_normal_N}

        return bundled_info

    compute_info_helper_vmap = jax.vmap(compute_info_helper, in_axes=(0, 0, 0, 0, None))

    return compute_info_helper_vmap(np.arange(12), inds_i, inds_j, ft_data, params)

compute_info_vmap = jax.jit(jax.vmap(compute_info, in_axes=(0, 0, None, None, None)))


def facet_contribution(info, params, pos):
    B_N_i, B_N_j, B_M_i, B_M_j, B_L_i, B_L_j = info['B_mats']
    ind_i = info['ind_i']
    ind_j = info['ind_j']
    edge_l = info['edge_l']
    normal_N = info['normal_N']
    normal_M = info['normal_M']
    normal_L = info['normal_L']
    proj_area = info['proj_area']
    epsV = info['epsV']
    stv = info['stv']
    Q_i = pos[ind_i][:, None]
    Q_j = pos[ind_j][:, None]

    eps_N = (B_N_j @ Q_j - B_N_i @ Q_i).squeeze()
    eps_M = (B_M_j @ Q_j - B_M_i @ Q_i).squeeze()
    eps_L = (B_L_j @ Q_j - B_L_i @ Q_i).squeeze()
    
    eps = np.array([eps_N, eps_M, eps_L])

    epsV /= 3.

    stv = stress_fn(eps, epsV, stv, info, params)

    info['stv'] = stv
    sigma = stv[1:4]
    sigma_N, sigma_M, sigma_L = sigma

    F_i = -edge_l*proj_area*(sigma_N*B_N_i + sigma_M*B_M_i + sigma_L*B_L_i).squeeze()
    F_j =  edge_l*proj_area*(sigma_N*B_N_j + sigma_M*B_M_j + sigma_L*B_L_j).squeeze()

    energy = 1./2.*edge_l*proj_area*np.sum(sigma*eps)

    info['F_i'] = -F_i
    info['F_j'] = -F_j
    info['elastic_energy'] = energy

    return ind_i, -F_i, ind_j, -F_j, info

facet_contributions = jax.vmap(facet_contribution, in_axes=(0, None, None))


def compute_epsV(pos, bundled_info, params):
    u = pos[:, :3]
    crt_points = params['points'] + u
    facet_vols = compute_facet_vols(params['cells'], crt_points, bundled_info)
    facet_vols_initial = bundled_info['facet_vols_initial']
    epsV = (facet_vols - facet_vols_initial)/facet_vols_initial
    bundled_info['epsV'] = epsV
    return bundled_info


@jax.jit
def compute_node_reactions(node_var, bundled_info, params):
    # TODO: pos should not be passed since it's actually not used
    pos = node_var[:, :6]
    inds_i, inds_j = bundled_info['ind_i'], bundled_info['ind_j']
    F_i, F_j = bundled_info['F_i'], bundled_info['F_j']
    inds = np.hstack((inds_i, inds_j))
    facet_forces = np.vstack((F_i, F_j))
    node_reactions = np.zeros((len(pos), 6))
    node_reactions = node_reactions.at[inds].add(facet_forces)
    return node_reactions


@jax.jit
def compute_elastic_energy(node_var, bundled_info, params):
    pos = node_var[:, :6]
    elastic_energy = bundled_info['elastic_energy']
    return np.sum(elastic_energy)


@jax.jit
def compute_kinetic_energy(node_var, ms):
    vel = node_var[:, 6:]
    def node_ke(node_vel, node_mass):
        ke = 0.5*node_vel[None, :] @ node_mass @ node_vel[:, None]
        return ke.squeeze()

    energy = jax.vmap(node_ke)(vel, ms)
    return np.sum(energy)


def rhs_func(pos, vel, bundled_info, mass, params, applied_reactions, friction_bc):
    # This is terrible - to change. RHS computation should be coherent.
    bundled_info = compute_epsV(pos, bundled_info, params)

    inds_i, forces_i, inds_j, forces_j, bundled_info = facet_contributions(bundled_info, params, pos)

    inds = np.hstack((inds_i, inds_j))
    reactions = np.vstack((forces_i, forces_j))
    node_reactions = np.zeros((pos.shape))
    node_reactions = node_reactions.at[inds].add(reactions)

    friciton_force = friction_bc(pos, vel, node_reactions)

    node_reactions = node_reactions.at[:, :2].add(friciton_force)

    node_reactions += applied_reactions

    def de_mass(node_pos, node_mass):
        # return np.linalg.solve(node_mass, node_pos) # No mass lumping
        return node_pos/np.diag(node_mass) # Mass lumping 1
        # return node_pos/np.sum(node_mass, axis=-1) # Mass lumping 2

    vel_rhs = jax.vmap(de_mass)(node_reactions, mass)

    damping_v = params['damping_v']
    damping_w = params['damping_w']
    damping_rhs = -np.hstack((damping_v*vel[:, :3], damping_w*vel[:, 3:])) 

    vel_rhs += damping_rhs

    return vel_rhs, bundled_info


def cross_prod_w_to_Omega(w):
    return np.array([[   0., -w[2],  w[1]],
                     [ w[2],    0., -w[0]],
                     [-w[1],  w[0],    0.]])


def compute_mass(N_nodes, bundled_info, params):
    tet_points_i = bundled_info['tet_points_i']
    tet_points_j = bundled_info['tet_points_j']
    inds_i = bundled_info['ind_i']
    inds_j = bundled_info['ind_j']

    def true_mass(vol, fm, inertia):
        V = vol*np.eye(3)
        Omega = cross_prod_w_to_Omega(fm)
        I = inertia
        M = np.block([[V, -Omega], [Omega, I]])
        return M 

    def lumped_mass(vol, fm, inertia):
        M = true_mass(vol, fm, inertia)
        # return np.sum(M, axis=1) # Mass lumping 1
        return np.diag(M) # Mass lumping 2

    def mass_helper(tet_points):
        Os, Ds, Es, Fs = np.transpose(tet_points, axes=(1, 0, 2))
        vols = tetrahedra_volumes(Os, Ds, Es, Fs)
        fms = tetra_first_moments_helper(Os, Ds, Es, Fs)
        inertias = tetra_inertia_tensors_helper(Os, Ds, Es, Fs)
        return vols, fms, inertias     

    def compute_true_mass(tet_points):
        vols, fms, inertias = mass_helper(tet_points)
        true_ms = jax.vmap(true_mass)(vols, fms, inertias)
        return true_ms

    def compute_lumped_mass(tet_points):
        vols, fms, inertias = mass_helper(tet_points)
        lumped_ms = jax.vmap(lumped_mass)(vols, fms, inertias)
        return lumped_ms

    true_ms_i = compute_true_mass(tet_points_i)
    true_ms_j = compute_true_mass(tet_points_j)

    lumped_ms_i = compute_lumped_mass(tet_points_i)
    lumped_ms_j = compute_lumped_mass(tet_points_j)

    inds = np.hstack((inds_i, inds_j))
    true_ms = np.concatenate((true_ms_i, true_ms_j), axis=0)
    lumped_ms = np.vstack((lumped_ms_i, lumped_ms_j))

    node_true_ms = np.zeros((N_nodes, 6, 6))
    node_true_ms = node_true_ms.at[inds].add(true_ms)

    node_lumped_ms = np.zeros((N_nodes, 6))
    node_lumped_ms = node_lumped_ms.at[inds].add(lumped_ms)

    rho = params['rho']
    node_true_ms *= rho
    node_lumped_ms *= rho

    return node_true_ms, node_lumped_ms


def compute_facet_vols(cells, points, bundled_info):
    points_transposed = np.transpose(points[cells], axes=(1, 0, 2))
    cell_vols = tetrahedra_volumes(*points_transposed)
    facet_vols = cell_vols[bundled_info['cell_id']]
    return facet_vols


def split_tets(cells, points, facet_data, facet_vertices, params):
    """Compute useful facet information

    Parameters
    ----------
    cells : onp.ndarray
        (num_cells, 4)
    points : onp.ndarray
        (num_points, 3)
    facet_data : onp.ndarray
        (num_cells, 12, 19)
    facet_vertices : onp.ndarray
        (num_facets*3, 3)
    params : dict

    Returns
    -------
    bundled_info : dict
    tet_cells : jax.Array
        (num_tet_cells, 4)
    tet_points : jax.Array
        (num_tet_points, 3)
    """

    bundled_info = compute_info_vmap(cells, facet_data, points, facet_vertices, params)
    bundled_info = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), bundled_info)
    tet_points = np.vstack((bundled_info['tet_points_i'].reshape(-1, 3), 
                            bundled_info['tet_points_j'].reshape(-1, 3)))
    tet_cells = np.arange(len(tet_points)).reshape(-1, 4)

    bundled_info['facet_vols_initial'] = compute_facet_vols(cells, points, bundled_info)
    num_facets = len(bundled_info['facet_vols_initial'])
    bundled_info['stv'] = np.zeros((num_facets, 22))

    qlts = check_mesh_TET4(points, cells)
    tet_qlts = check_mesh_TET4(tet_points, tet_cells)

    try:
        assert np.all(qlts > 0.), f"Mesh tetrahedron orientation test failed"
    except Exception as e:
        print(e)
        print(f"Warning: Mesh tetrahedra do NOT all have the same orientation!")
        print(f"This should not be a problem in the current implementation, but be careful.")

    # print(np.argwhere(tet_qlts<0.))
    # print(tet_qlts[:20])
    # print(tet_qlts[-20:])

    try:
        assert np.all(tet_qlts > 0.), f"Facet tetrahedra orientation test failed"
    except Exception as e:
        print(e)
        print(f"Warning: Facet tetrahedra do NOT all have the same orientation!")
        print(f"This should not be a problem in the current implementation, but be careful.")

    return bundled_info, tet_cells, tet_points


@jax.jit
def process_tet_point_sol(points, bundled_info, node_var):
    pos = node_var[:, :6]
    tets_x = np.vstack((bundled_info['tet_points_i'], bundled_info['tet_points_j']))
    inds = np.hstack((bundled_info['ind_i'], bundled_info['ind_j']))
    def get_disp(tet_x, ind):
        u, theta = pos[ind][:3], pos[ind][3:]
        x_i = points[ind]
        tet_u = u + np.cross(theta, tet_x - x_i)
        return tet_u
    tets_u = jax.vmap(get_disp)(tets_x, inds)
    tets_u = tets_u.reshape(-1, 3)
    return tets_u


@jax.jit
def apply_bc(node_var, inds_node, inds_var, values):
    node_var = node_var.at[inds_node, inds_var].set(values)
    return node_var
