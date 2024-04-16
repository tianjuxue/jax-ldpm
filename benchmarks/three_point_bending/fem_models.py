import jax
import jax.numpy as np
import numpy as onp
import meshio
import os
import glob

from jax_fem.problem import Problem


class LinearElasticityMass(Problem):
    def custom_init(self):
        self.fe = self.fes[0]

    def get_mass_map(self):
        def mass_map(u, x):
            rho = 2.338e-9
            return rho*u
        return mass_map


class LinearElasticity(Problem):
    def custom_init(self, normal):
        self.fe = self.fes[0]
        self.normal = normal
        self.location_fn = self.fe.dirichlet_bc_info[0][-1]
        # (num_selected_faces, 2)
        self.boundary_inds = self.fe.get_boundary_conditions_inds([self.location_fn])[0]
        self.face_shape_grads_physical, self.nanson_scale = self.fe.get_face_shape_grads(self.boundary_inds)
        self.compute_quad_forces = self.get_compute_quad_forces()

    def get_tensor_map(self):
        def stress(u_grad):
            E = 39000
            nu = 0.2
            mu = E/(2.*(1. + nu))
            lmbda = E*nu/((1+nu)*(1-2*nu))
            epsilon = 0.5*(u_grad + u_grad.T)
            sigma = lmbda*np.trace(epsilon)*np.eye(self.dim) + 2*mu*epsilon
            return sigma
        return stress

    def get_compute_quad_forces(self):
        """For post-processing only
        """
        def quad_force_fn(sol):
            stress = self.get_tensor_map()
            vmap_stress = jax.vmap(stress)
            def traction_fn(u_grads):
                """
                Returns
                -------
                traction: ndarray
                    (num_selected_faces, num_face_quads, vec)
                """
                # (num_selected_faces, num_face_quads, vec, dim) -> (num_selected_faces*num_face_quads, vec, dim)
                u_grads_reshape = u_grads.reshape(-1, self.fe.vec, self.dim)
                sigmas = vmap_stress(u_grads_reshape).reshape(u_grads.shape)
                # TODO: a more general normals with shape (num_selected_faces, num_face_quads, dim, 1) should be supplied
                # (num_selected_faces, num_face_quads, vec, dim) @ (1, 1, dim, 1) -> (num_selected_faces, num_face_quads, vec, 1)
                # normal = np.array([1., 0., 0.])
                traction = (sigmas @ self.normal[None, None, :, None])[:, :, :, 0]
                return traction

            location_fn = self.location_fn
            boundary_inds = self.boundary_inds
            face_shape_grads_physical, nanson_scale = self.face_shape_grads_physical, self.nanson_scale

            # (num_selected_faces, 1, num_nodes, vec, 1) * (num_selected_faces, num_face_quads, num_nodes, 1, dim)
            u_grads_face = sol[self.fe.cells][boundary_inds[:, 0]][:, None, :, :, None] * face_shape_grads_physical[:, :, :, None, :]
            u_grads_face = np.sum(u_grads_face, axis=2) # (num_selected_faces, num_face_quads, vec, dim)
            traction = traction_fn(u_grads_face) # (num_selected_faces, num_face_quads, vec)

            # (num_selected_faces, num_face_quads, vec) * (num_selected_faces, num_face_quads, 1)
            quad_forces = (traction * nanson_scale[:, :, None]).reshape(-1, self.fe.vec)
            return quad_forces
        return jax.jit(quad_force_fn)


    def get_quad_points(self):
        location_fn = self.fe.dirichlet_bc_info[0][-1]
        # (num_selected_faces, 2)
        boundary_inds = self.fe.get_boundary_conditions_inds([location_fn])[0]
        # (num_selected_faces, num_face_quads, dim)
        quad_points = (self.fe.get_physical_surface_quad_points(boundary_inds)).reshape(-1, self.fe.vec)
        return quad_points


    def get_boundary_points(self):
        node_inds = self.fe.node_inds_list[-1]
        return self.fe.points[node_inds]

    def set_params(self, params):
        fem_disp = params
        self.fe.vals_list[-1] = fem_disp[:, -1]
        self.fe.vals_list[-2] = fem_disp[:, -2]
        self.fe.vals_list[-3] = fem_disp[:, -3]
