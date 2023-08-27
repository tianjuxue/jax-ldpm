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


# config.update("jax_enable_x64", True)

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


@partial(jax.jit, static_argnums=(1,))
def explicit_euler(variable, rhs, dt, *args):
    return variable + dt * rhs(variable, *args)


@partial(jax.jit, static_argnums=(1,))
def runge_kutta_4(variable, rhs, dt, *args):
    y_0 = variable
    k_0 = rhs(y_0, *args)
    k_1 = rhs(y_0 + dt/2 * k_0, *args)
    k_2 = rhs(y_0 + dt/2 * k_1, *args)
    k_3 = rhs(y_0 + dt * k_2, *args)
    k = 1./6. * (k_0 + 2. * k_1 + 2. * k_2 + k_3)
    y_1 = y_0 + dt * k
    return y_1


def check_mesh_TET4(points, cells):
    # TODO
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



def compute_info(cell, points):
    cell_centroid = np.mean(points[cell], axis=0)
    # Ordered in a way that all tets are in positive orientation
    inds = np.array([[0, 1, 2],
                     [1, 0, 3],
                     [2, 0, 1],
                     [0, 2, 3],
                     [0, 3, 1],
                     [3, 0, 2],
                     [1, 2, 0],
                     [2, 1, 3],
                     [3, 1, 0],
                     [1, 3, 2],
                     [2, 3, 0],
                     [3, 2, 1]])

    inds_i = cell[inds[:, 0]]
    inds_j = cell[inds[:, 1]]
    inds_nb = cell[inds[:, 2]]

    def compute_info_helper(ind_i, ind_j, ind_nb):
        point_i = points[ind_i]
        point_j = points[ind_j]
        point_nb = points[ind_nb]
        face_centroid = (point_i + point_j + point_nb)/3.
        edge_centroid = (point_i + point_j)/2.

        # tet_points_i = np.stack((edge_centroid, cell_centroid, face_centroid, point_i))
        # tet_points_j = np.stack((edge_centroid, face_centroid, cell_centroid, point_j))
 
        tet_points_i = np.stack((point_i, edge_centroid, face_centroid, cell_centroid))
        tet_points_j = np.stack((point_j, edge_centroid, cell_centroid, face_centroid))
 
        edge_vec = point_j - point_i
        edge_l = np.linalg.norm(edge_vec)  
        normal_N = edge_vec/edge_l
        tmp_vec1 = point_nb - edge_centroid
        tmp_vec2 = np.cross(normal_N, tmp_vec1)
        normal_M = tmp_vec2/np.linalg.norm(tmp_vec2)
        normal_L = np.cross(normal_N, normal_M)
        proj_area = projected_face_area(edge_centroid, cell_centroid, face_centroid, normal_M, normal_L)

        facet_centroid = (edge_centroid + cell_centroid + face_centroid)/3.
        A_i = get_A_matrix(point_i, facet_centroid)
        A_j = get_A_matrix(point_j, facet_centroid)
        B_N_i = 1./edge_l * normal_N[None, :] @ A_i
        B_N_j = 1./edge_l * normal_N[None, :] @ A_j
        B_M_i = 1./edge_l * normal_M[None, :] @ A_i
        B_M_j = 1./edge_l * normal_M[None, :] @ A_j
        B_L_i = 1./edge_l * normal_L[None, :] @ A_i
        B_L_j = 1./edge_l * normal_L[None, :] @ A_j

        bundled_info = {'tet_points_i': tet_points_i,
                        'tet_points_j': tet_points_j,
                        'edge_l': edge_l,
                        'proj_area': proj_area,
                        'normal_N': normal_N,
                        'normal_M': normal_M,
                        'normal_L': normal_L,
                        'ind_i': ind_i,
                        'ind_j': ind_j,
                        'B_mats': np.stack((B_N_i, B_N_j, B_M_i, B_M_j, B_L_i, B_L_j))}

        return bundled_info

    compute_info_helper_vmap = jax.vmap(compute_info_helper)

    return compute_info_helper_vmap(inds_i, inds_j, inds_nb)

compute_info_vmap = jax.jit(jax.vmap(compute_info, in_axes=(0, None)))



def stress_fn(eps):
    eps_N, eps_M, eps_L = eps
    E = 1e7
    nu = 0.2
    E0 = 1./(1. - 2*nu)*E
    alpha = (1. - 4.*nu)/(1. + nu)
    E_N = E0
    E_T = alpha*E0
    sigma = np.array([E_N*eps_N, E_T*eps_M, E_T*eps_L])
    return sigma


def facet_contribution(info, pos):
    B_N_i, B_N_j, B_M_i, B_M_j, B_L_i, B_L_j = info['B_mats']
    ind_i = info['ind_i']
    ind_j = info['ind_j']
    edge_l = info['edge_l']
    normal_N = info['normal_N']
    normal_M = info['normal_M']
    normal_L = info['normal_L']
    proj_area = info['proj_area']
    Q_i = pos[ind_i][:, None]
    Q_j = pos[ind_j][:, None]

    eps_N = (B_N_j @ Q_j - B_N_i @ Q_i).squeeze()
    eps_M = (B_M_j @ Q_j - B_M_i @ Q_i).squeeze()
    eps_L = (B_L_j @ Q_j - B_L_i @ Q_i).squeeze()
    
    eps = np.array([eps_N, eps_M, eps_L])
    sigma = stress_fn(eps)
    sigma_N, sigma_M, sigma_L = sigma

    F_i = -edge_l*proj_area*(sigma_N*B_N_i + sigma_M*B_M_i + sigma_L*B_L_i).squeeze()
    F_j =  edge_l*proj_area*(sigma_N*B_N_j + sigma_M*B_M_j + sigma_L*B_L_j).squeeze()

    energy = 1./2.*edge_l*proj_area*np.sum(sigma*eps)

    facet_force_i = proj_area*(sigma_N*normal_N + sigma_M*normal_M + sigma_L*normal_L)
    facet_force_j = -facet_force_i

    return ind_i, -F_i, ind_j, -F_j, energy, facet_force_i, facet_force_j

facet_contributions = jax.vmap(facet_contribution, in_axes=(0, None))


@jax.jit
def compute_node_forces(state, bundled_info):
    pos = state[:, :6]
    inds_i, _, inds_j, _, _, facet_forces_i, facet_forces_j = facet_contributions(bundled_info, pos)
    inds = np.hstack((inds_i, inds_j))
    facet_forces = np.vstack((facet_forces_i, facet_forces_j))
    node_forces = np.zeros((len(pos, 3)))
    node_forces = node_forces.at[inds].add(facet_forces)
    return node_forces


@jax.jit
def compute_elastic_energy(state, bundled_info):
    pos = state[:, :6]
    _, _, _, _, energy, _, _ = facet_contributions(bundled_info, pos)
    return np.sum(energy)


@jax.jit
def compute_kinetic_energy(state, true_ms):
    vel = state[:, 6:]
    def node_ke(node_vel, node_mass):
        ke = 0.5*node_vel[None, :] @ node_mass @ node_vel[:, None]
        return ke.squeeze()

    energy = jax.vmap(node_ke)(vel, true_ms)
    return np.sum(energy)
    

def rhs_func(state, bundled_info, mass):
    pos = state[:, :6]
    vel = state[:, 6:]
 
    inds_i, forces_i, inds_j, forces_j, _, _, _ = facet_contributions(bundled_info, pos)

    inds = np.hstack((inds_i, inds_j))
    reactions = np.vstack((forces_i, forces_j))
    node_reactions = np.zeros((pos.shape))
    node_reactions = node_reactions.at[inds].add(reactions)

    def de_mass(node_pos, node_mass):
        return np.linalg.solve(node_mass, node_pos)

    pos_rhs = vel
    vel_rhs = jax.vmap(de_mass)(node_reactions, mass)

    damping_v = 1e1 * 0.
    damping_w = 1e1 * 0.
    damping_rhs = -np.hstack((damping_v*vel[:, :3], damping_w*vel[:, 3:])) 

    vel_rhs += damping_rhs

    rhs = np.hstack((pos_rhs, vel_rhs))

    # rhs_vals /= mass

    return rhs


def cross_prod_w_to_Omega(w):
    return np.array([[   0., -w[2],  w[1]],
                     [ w[2],    0., -w[0]],
                     [-w[1],  w[0],    0.]])


def compute_mass(N_nodes, bundled_info):
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
        return np.sum(M, axis=1)

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

    rho = 1e3
    node_true_ms *= rho
    node_lumped_ms *= rho

    return node_true_ms, node_lumped_ms


@jax.jit
def process_tet_point_sol(points, bundled_info, state):
    pos = state[:, :6]
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
def apply_bc(state, inds_node, inds_var, values):
    state = state.at[inds_node, inds_var].set(values)
    return state
