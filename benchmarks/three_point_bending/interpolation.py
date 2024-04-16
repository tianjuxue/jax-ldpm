import numpy as onp
import jax
import jax.numpy as np


# master-slave
# master: FEM
# slave: LDPM

@jax.jit
def get_transformation_matrix(ldpm_points_yz, fem_points_yz):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_points_yz: (M, 2)

    Returns
    -------
    T_mat: (N, M)
    """
    # TODO:
    edge_l = 20. 
    def shape_value(eval_point, node_point):
        xe, ye = eval_point
        xn, yn = node_point
        eta_x = (xe - xn)/edge_l
        eta_y = (ye - yn)/edge_l
        val1 = np.where((eta_x >= 0.) & (eta_x <= 1.) & (eta_y >= 0.) & (eta_y <= 1.), (1 - eta_x)*(1 - eta_y), 0.)
        val2 = np.where((eta_x >= 0.) & (eta_x <= 1.) & (eta_y < 0.) & (eta_y >= -1.), (1 - eta_x)*(1 + eta_y), 0.)
        val3 = np.where((eta_x < 0.) & (eta_x >= -1.) & (eta_y >= 0.) & (eta_y <= 1.), (1 + eta_x)*(1 - eta_y), 0.)
        val4 = np.where((eta_x < 0.) & (eta_x >= -1.) & (eta_y < 0.) & (eta_y >= -1.), (1 + eta_x)*(1 + eta_y), 0.)
        return val1 + val2 + val3 + val4

    T_mat = jax.vmap(jax.vmap(shape_value, in_axes=(None, 0)), in_axes=(0, None))(ldpm_points_yz, fem_points_yz)
    return T_mat


@jax.jit
def ldpm_to_fem_mass(T_mat, ldpm_mass):
    """
    Parameters
    ----------
    T_mat: (N, M)
    ldpm_mass: (N, X)

    Returns
    -------
    fem_mass: (M, X)
    """
    f = jax.vmap(np.diag)
    ldpm_mass_mat = f(ldpm_mass.T) # (X, N, N)
    # (1, M, N) @ (X, N, N) @ (1, N, M) -> (X, M, M) -> (X, M) -> (M, X)
    fem_mass = f(T_mat.T[None, :, :] @ ldpm_mass_mat @ T_mat[None, :, :]).T 
    return fem_mass


@jax.jit
def ldpm_to_fem_force(T_mat, ldpm_force):
    """
    Parameters
    ----------
    T_mat: (N, M)
    ldpm_force: (N, X)

    Returns
    -------
    fem_force: (M, X)
    """
    fem_force = np.sum(ldpm_force[:, None, :] * T_mat[:, :, None], 0) # (N, 1, X) * (N, M, 1) -> (N, M, X) -> (M, X)
    return fem_force


@jax.jit
def fem_to_ldpm_disp(T_mat, fem_disp):
    """
    Parameters
    ----------
    T_mat: (N, M)
    fem_disp: (M, X)

    Returns
    -------
    ldpm_disp: (N, X)
    """
    ldpm_disp = np.sum(fem_disp[None, :, :] * T_mat[:, :, None], 1) # (1, M, X) * (N, M, 1) -> (N, M, X) -> (N, X)
    return ldpm_disp
