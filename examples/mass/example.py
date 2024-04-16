import numpy as onp
import jax
import jax.numpy as np
import sys

from jax_ldpm.tetrahedron import tetrahedra_volumes, tetra_first_moments_helper, tetra_inertia_tensors_helper
from jax_ldpm.core import cross_prod_w_to_Omega

onp.set_printoptions(threshold=sys.maxsize,
                     linewidth=1000,
                     suppress=True,
                     precision=5)


def volumtetra(O, D, E, F):
    '''
    Signed volume of a tetrahedron with vertices (O, D, E, F)
    '''
    DO = D - O
    ED = E - D
    FD = F - D
    return onp.abs(onp.dot(DO, onp.cross(ED, FD)) / 6.)



def fortran_MI(Xi, Xa, Xb, Xc):

    density = 1.

    coef = onp.zeros((3, 3))

    coef[0, 0] = 0.1
    coef[0, 1] = 0.05
    coef[0, 2] = 0.05      
    coef[1, 0] = 0.05
    coef[1, 1] = 0.1
    coef[1, 2] = 0.05      
    coef[2, 0] = 0.05
    coef[2, 1] = 0.05
    coef[2, 2] = 0.1  

    tetvol = volumtetra(Xi, Xa, Xb, Xc)

    vol = density*tetvol 

    Xg = (Xi+Xa+Xb+Xc)/4.0

    X, Y, Z = onp.zeros((3)), onp.zeros((3)), onp.zeros((3))

    X[0] = Xa[0]-Xi[0]
    X[1]=Xb[0]-Xi[0]
    X[2]=Xc[0]-Xi[0]
    Y[0] = Xa[1]-Xi[1]
    Y[1]=Xb[1]-Xi[1]
    Y[2]=Xc[1]-Xi[1]
    Z[0] = Xa[2]-Xi[2]
    Z[1]=Xb[2]-Xi[2]; Z[2]=Xc[2]-Xi[2]

    Sx = (Xg[0]-Xi[0])*vol
    Sy = (Xg[1]-Xi[1])*vol
    Sz = (Xg[2]-Xi[2])*vol

    vIx = onp.dot(X,  coef @ X)*vol
    vIy = onp.dot(Y, coef @ Y)*vol
    vIz = onp.dot(Z, coef @ Z)*vol
    vIxy = onp.dot(X, coef @ Y)*vol
    vIxz = onp.dot(X, coef @ Z)*vol
    vIyz = onp.dot(Y, coef @ Z)*vol

    pMi = onp.zeros((6, 6))
    pMi[0,0]=vol
    pMi[0,4]= Sz
    pMi[0,5]=-Sy
    pMi[1,1]=vol 
    pMi[1,3]=-Sz
    pMi[1,5]= Sx
    pMi[2,2]=vol
    pMi[2,3]= Sy
    pMi[2,4]=-Sx
    pMi[3,1]=-Sz
    pMi[3,2]= Sy
    pMi[3,3]=vIy+vIz
    pMi[3,4]=-vIxy
    pMi[3,5]=-vIxz 
    pMi[4,0]= Sz
    pMi[4,2]=-Sx
    pMi[4,3]=-vIxy
    pMi[4,4]=vIx+vIz
    pMi[4,5]=-vIyz 
    pMi[5,0]=-Sy
    pMi[5,1]= Sx
    pMi[5,3]=-vIxz
    pMi[5,4]=-vIyz
    pMi[5,5]=vIx+vIy 

    return pMi


def python_MI(tet_points):
    def true_mass(vol, fm, inertia):
        V = vol*np.eye(3)
        Omega = cross_prod_w_to_Omega(fm)
        I = inertia
        M = np.block([[V, -Omega], [Omega, I]])
        return M 

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

    return compute_true_mass(tet_points)


def test():
    points = onp.array([[0.1, 0.2, 0.3],
                        [1.2, 0., 0.],
                        [0., 1.3, 0.],
                        [0., 0., 1.4]])
    f_MI = fortran_MI(*points)
    p_MI = python_MI(points[None, :, :])
    print(f"\n")
    print(f_MI)
    print(f"\n")
    print(p_MI)
    print(f"\n")
    print(f_MI - p_MI)

    points[[1, 2], :] = points[[2, 1], :]

    f_MI = fortran_MI(*points)
    p_MI = python_MI(points[None, :, :])
    print(f"\n")
    print(f_MI)
    print(f"\n")
    print(p_MI)
    print(f"\n")
    print(f_MI - p_MI)


if __name__ == '__main__':
    test()
    