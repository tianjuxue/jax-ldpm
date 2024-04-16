import numpy as onp
import jax
import jax.numpy as np



###########################################################################################
# Correct master-slave
# master: FEM
# slave: LDPM
# But bad interpolation: not using FEM shape function


@jax.jit
def ldpm_to_fem_mass(ldpm_points_yz, fem_points_yz, ldpm_mass):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_node_forces: (M, 3)
    ldpm_mass: (N, X)

    Returns
    -------
    fem_mass: (M, X)
    """
    l = 10.
    f = jax.vmap(np.diag)
    mutual_dist = np.linalg.norm(fem_points_yz[None, :, :] - ldpm_points_yz[:, None, :], axis=-1) # (N, M)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (N, M)
    ldpm_mass_mat = f(ldpm_mass.T) # (X, N, N)
    # (1, M, N) @ (X, N, N) @ (1, N, M) -> (X, M, M) -> (X, M) -> (M, X)
    fem_mass = f(weight.T[None, :, :] @ ldpm_mass_mat @ weight[None, :, :]).T 
    return fem_mass


@jax.jit
def ldpm_to_fem_force(ldpm_points_yz, fem_points_yz, ldpm_force):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_points_yz: (M, 2)
    ldpm_force: (N, X)

    Returns
    -------
    fem_force: (M, X)
    """
    l = 10.
    mutual_dist = np.linalg.norm(fem_points_yz[None, :, :] - ldpm_points_yz[:, None, :], axis=-1) # (N, M)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (N, M)
    fem_force = np.sum(ldpm_force[:, None, :] * weight[:, :, None], 0) # (N, 1, X) * (N, M, 1) -> (N, M, X) -> (M, X)
    return fem_force


@jax.jit
def fem_to_ldpm_disp(ldpm_points_yz, fem_points_yz, fem_disp):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_points_yz: (M, 2)
    fem_disp: (M, X)

    Returns
    -------
    ldpm_disp: (N, X)
    """
    l = 10.
    mutual_dist = np.linalg.norm(fem_points_yz[None, :, :] - ldpm_points_yz[:, None, :], axis=-1) # (N, M)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (N, M)
    ldpm_disp = np.sum(fem_disp[None, :, :] * weight[:, :, None], 1) # (1, M, X) * (N, M, 1) -> (N, M, X) -> (N, X)
    return ldpm_disp


###########################################################################################
# Wrong master-slave
# master: LDPM
# slave: FEM

@jax.jit
def ldpm_to_fem(ldpm_points_yz, fem_points_yz, ldpm_quantity):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_node_forces: (M, 3)
    ldpm_quantity: (N, X)

    Returns
    -------
    fem_quantity: (M, X)
    """
    l = 10.
    mutual_dist = np.linalg.norm(fem_points_yz[:, None, :] - ldpm_points_yz[None, :, :], axis=-1) # (M, N)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (M, N)
    fem_quantity = np.sum(ldpm_quantity[None, :, :] * weight[:, :, None], 1) # (1, N, X) * (M, N, 1) -> (M, N, X) -> (M, X)
    return fem_quantity


@jax.jit
def fem_to_ldpm(ldpm_points_yz, fem_points_yz, fem_quantity):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_node_forces: (M, 3)
    fem_quantity: (M, X)

    Returns
    -------
    ldpm_quantity: (N, X)
    """
    l = 10.
    mutual_dist = np.linalg.norm(fem_points_yz[:, None, :] - ldpm_points_yz[None, :, :], axis=-1) # (M, N)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (M, N)
    ldpm_quantity = np.sum(fem_quantity[:, None, :] * weight[:, :, None], 0) # (M, 1, X) * (M, N, 1) -> (M, N, X) -> (N, X)
    return ldpm_quantity


@jax.jit
def fem_to_ldpm_debug_mass(ldpm_points_yz, fem_points_yz, fem_quantity):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    fem_node_forces: (M, 3)
    fem_quantity: (M, X)

    Returns
    -------
    ldpm_quantity: (N, X)
    """
    l = 10.
    f = jax.vmap(np.diag)
    mutual_dist = np.linalg.norm(fem_points_yz[:, None, :] - ldpm_points_yz[None, :, :], axis=-1) # (M, N)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (M, N)
    fem_mass_mat = f(fem_quantity.T) # (X, M, M)
    # (1, N, M) @ (X, M, M) @ (1, M, N) -> (X, N, N) -> (X, N) -> (N, X)
    ldpm_quantity = f(weight.T[None, :, :] @ fem_mass_mat @ weight[None, :, :]).T 
    return ldpm_quantity


###########################################################################################
# Spring

@jax.jit
def mutual_force(ldpm_points_yz, ldpm_disp, fem_points_yz, fem_disp):
    """
    Parameters
    ----------
    ldpm_points_yz: (N, 2)
    ldpm_disp: (N, 3)
    fem_points_yz: (M, 2)
    fem_disp: (M, 3)

    Returns
    -------

    force_on_fem: (M, 3)
    force_on_ldpm: (N, 3)
    """
    l = 0.1
    mutual_dist = np.linalg.norm(fem_points_yz[:, None, :] - ldpm_points_yz[None, :, :], axis=-1) # (M, N)
    weight = jax.nn.softmax(-mutual_dist**2/(2*l**2), axis=1) # (M, N)
    k = 1e7
    mutual_force = k*(fem_disp[:, None, :] - ldpm_disp[None, :, :]) * weight[:, :, None] # (M, N, 3)

    force_on_fem = -np.sum(mutual_force, axis=1) # (M, 3)
    force_on_ldpm = np.sum(mutual_force, axis=0) # (N, 3)
    return force_on_fem, force_on_ldpm
