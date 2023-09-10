import numpy as onp
import jax
import jax.numpy as np
import unittest
import numpy.testing as nptest

dim = 3

class TestTetrahedron(unittest.TestCase):

    def test_arbitrary_tetrahedron(self):
        '''
        Tetrahedron (O, D, E, F) has known centroid at G and inertia tensor I_G w.r.t. G
        Reference: doi.org/10.3844/jmssp.2005.8.11 (looks like Eq. 2 has typos)
        '''
        O = np.array([8.33220,  -11.86875, 0.93355])
        D = np.array([0.75523,   5.00000,  16.37072])
        E = np.array([52.61236,  5.00000, -5.38580])
        F = np.array([2.00000,   5.00000,  3.00000])
        G = np.array([15.92492,  0.78281,  3.72962])
        I_G = np.array([[ 43520.33257, -11996.20119,   46343.16662],
                        [-11996.20119,  194711.28938, -4417.66150],
                        [ 46343.16662, -4417.66150,    191168.76173]])

        nptest.assert_array_almost_equal(tetrahedron_centroid(O, D, E, F), G, decimal=3)
        nptest.assert_array_almost_equal(tetra_inertia_tensor(O, D, E, F, G), I_G, decimal=1)
 
    def test_regular_tetrahedron(self):
        '''
        Tetrahedron (O, D, E, F) is regular and has an analytical formula for inertia tensor
        Reference: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
        '''
        O = np.array([ 1.,   1.,  1.])
        D = np.array([ 1.,  -1., -1.])
        E = np.array([-1.,  -1.,  1.])
        F = np.array([-1.,   1., -1.])
        G = np.array([ 0.,   0.,  0.])
       
        s = np.linalg.norm(D - O)
        vol = tetrahedron_volume(O, D, E, F)
        I_G = 1./20. * vol * s**2 * np.eye(dim)

        nptest.assert_array_almost_equal(tetrahedron_centroid(O, D, E, F), G, decimal=5)
        nptest.assert_array_almost_equal(tetra_inertia_tensor(O, D, E, F, G), I_G, decimal=5)

    def test_cube(self):
        O = np.array([0., 0., 0.])
        A = np.array([1., 0., 0.])
        B = np.array([0., 2., 0.])
        C = np.array([0., 0., 3.])
        D = np.array([1., 2., 3.])
        E = np.array([0., 2., 3.])
        F = np.array([1., 0., 3.])
        G = np.array([1., 2., 0.])

        fm_OEFG = tetra_first_moment(O, E, F, G, O)
        fm_OGFA = tetra_first_moment(O, G, F, A, O)
        fm_OEGB = tetra_first_moment(O, E, G, B, O)
        fm_OFEC = tetra_first_moment(O, F, E, C, O)
        fm_EFGD = tetra_first_moment(E, F, G, D, O)

        fm_cube = fm_OEFG + fm_OGFA + fm_OEGB + fm_OFEC + fm_EFGD
        gt = np.array([3., 6., 9.])

        nptest.assert_array_almost_equal(fm_cube, gt, decimal=5)


def signed_tetrahedron_volume(O, D, E, F):
    '''
    Signed volume of a tetrahedron with vertices (O, D, E, F)
    '''
    DO = D - O
    ED = E - D
    FD = F - D
    return np.dot(DO, np.cross(ED, FD)) / 6.

signed_tetrahedra_volumes = jax.vmap(signed_tetrahedron_volume, in_axes=(0, 0, 0, 0), out_axes=0)


def tetrahedron_volume(O, D, E, F):
    '''
    Volume of a tetrahedron with vertices (O, D, E, F)
    ''' 
    return np.absolute(signed_tetrahedron_volume(O, D, E, F))

tetrahedra_volumes = jax.vmap(tetrahedron_volume, in_axes=(0, 0, 0, 0), out_axes=0)



def tetrahedron_centroid(O, D, E, F):
    '''
    Mass center of a tetrahedron with vertices (O, D, E, F) 
    '''
    return (O + D + E + F) / 4.


tetrahedra_centroids = jax.vmap(tetrahedron_centroid, in_axes=(0, 0, 0, 0), out_axes=0)


def tetra_inertia_tensor_helper(O, D, E, F):
    '''
    Inertia tensor of a tetrahedron with vertices (O, D, E, F) w.r.t point O
    Reference: https://doi.org/10.1006/icar.1996.0243
    The orientation of the vertices matters.
    '''    
    DO = D - O
    EO = E - O
    FO = F - O

    # If vol is unsigned, then orientation of the vertices does not matter.
    vol = tetrahedron_volume(O, D, E, F)
    P = []
    for i in range(dim):
        P.append([])
        for j in range(dim):
            tmp = 2 * (DO[i] * DO[j] + EO[i] * EO[j] + FO[i] * FO[j]) + DO[i] * EO[j] + DO[j] * EO[i] + \
                  DO[i] * FO[j] +  DO[j] * FO[i] + EO[i] * FO[j] +  EO[j] * FO[i]
            P[i].append(tmp)
    I = [[ P[1][1] + P[2][2], -P[0][1],           -P[0][2]], 
         [-P[1][0],            P[0][0] + P[2][2], -P[1][2]], 
         [-P[2][0],           -P[2][1],            P[0][0] + P[1][1]]]

    I = vol / 20. * np.array(I)

    # jax.jit has no runtime error mechanism, so return nan if the orientation of the tetrahedron is not applicable.
    return np.where( (I[0, 0] > 0.) & (I[1, 1] > 0.) & (I[2, 2] > 0.), I, np.nan)
 

tetra_inertia_tensors_helper = jax.vmap(tetra_inertia_tensor_helper, in_axes=(0, 0, 0, 0), out_axes=0)


def tetra_inertia_tensor(O, D, E, F, P):
    '''
    Inertia tensor of a tetrahedron with vertices (O, D, E, F) w.r.t an arbitrary point point P
    Use parallel axis theorem (see "Tensor generalization" on Wikipeida page "Parallel axis theorem")
    '''
    vol = tetrahedron_volume(O, D, E, F)
    center = tetrahedron_centroid(O, D, E, F)
    r_P = P - center
    r_O = O - center
    tmp_P = vol * (np.dot(r_P, r_P) * np.eye(dim) - np.outer(r_P, r_P))
    tmp_O = vol * (np.dot(r_O, r_O) * np.eye(dim) - np.outer(r_O, r_O))
    I_O = tetra_inertia_tensor_helper(O, D, E, F)
    I_P = I_O - tmp_O + tmp_P
    return I_P

tetra_inertia_tensors = jax.vmap(tetra_inertia_tensor, in_axes=(0, 0, 0, 0, 0), out_axes=0)


def tetra_first_moment_helper(O, D, E, F):
    """
    Reference: https://doi.org/10.1006/icar.1996.0243
    Further derived by Tianju
    """
    DO = D - O
    EO = E - O
    FO = F - O
    vol = tetrahedron_volume(O, D, E, F)
    first_moment = 1./4.*vol*(DO + EO + FO)
    return np.where(vol > 0., first_moment, np.nan)

tetra_first_moments_helper = jax.vmap(tetra_first_moment_helper, in_axes=(0, 0, 0, 0), out_axes=0)


def tetra_first_moment(O, D, E, F, P):
    OP = O - P
    vol = tetrahedron_volume(O, D, E, F)
    first_moment_of_O = tetra_first_moment_helper(O, D, E, F)
    first_moment = first_moment_of_O + vol*OP
    return first_moment

tetra_first_moments = jax.vmap(tetra_first_moment, in_axes=(0, 0, 0, 0, 0), out_axes=0)


def debug():
    O = np.array([0., 0., 0.])
    A = np.array([1., 0., 0.])
    B = np.array([0., 2., 0.])
    C = np.array([0., 0., 3.])
    D = np.array([1., 2., 3.])
    E = np.array([0., 2., 3.])
    F = np.array([1., 0., 3.])
    G = np.array([1., 2., 0.])

    fm_OEFG = tetra_first_moment(O, E, F, G, O)
    fm_OGFA = tetra_first_moment(O, G, F, A, O)
    fm_OEGB = tetra_first_moment(O, E, G, B, O)
    fm_OFEC = tetra_first_moment(O, F, E, C, O)
    fm_EFGD = tetra_first_moment(E, F, G, D, O)

    # print(fm_OEFG, fm_OGFA, fm_OEGB, fm_OFEC, fm_EFGD)
    print(fm_OEFG + fm_OGFA + fm_OEGB + fm_OFEC + fm_EFGD)
    print("ground truth = [3., 6., 9.]")

    inertia_OEFG = tetra_inertia_tensor(O, E, F, G, O)
    inertia_OGFA = tetra_inertia_tensor(O, G, F, A, O)
    inertia_OEGB = tetra_inertia_tensor(O, E, G, B, O)
    inertia_OFEC = tetra_inertia_tensor(O, F, E, C, O)
    inertia_EFGD = tetra_inertia_tensor(E, F, G, D, O)

    # print(fm_OEFG, fm_OGFA, fm_OEGB, fm_OFEC, fm_EFGD)
    print(inertia_OEFG + inertia_OGFA + inertia_OEGB + inertia_OFEC + inertia_EFGD)
    print(f"ground truth of diagonal = {[6.*4/3. + 6.*9/3., 6.*1./3. + 6.*9/3., 6.*1./3. + 6.*4/3.]}")


if __name__ == '__main__':
    # unittest.main()
    debug()
