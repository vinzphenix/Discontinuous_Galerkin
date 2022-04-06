# C:\Users\vince\anaconda3\share\doc\gmsh\tutorials\python
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def my_initial_condition(x):
    xc = x[:, 0] - 1. / 2.
    yc = x[:, 1] - 1. / 2.
    l2 = xc ** 2 + yc ** 2
    return np.exp(-l2 * 10.)


def my_velocity_condition(x):
    return x * 0. + 1.


if __name__ == "__main__":

    p = 4

    gmsh.initialize()
    gmsh.open("t1.msh")

    gmsh.model.mesh.setOrder(p)

    # Matrices M and D, and jacobians
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
    elementType = elemTypes[0]
    Nt = len(elemTags[0])

    # Name (?), dimension, order, nb vertices / element
    name, dim, order, Np, _, _ = gmsh.model.mesh.getElementProperties(elementType)
    prop = gmsh.model.mesh.getElementProperties(elementType)

    # location of gauss points in 3d space, and associated weights
    uvw, weights = gmsh.model.mesh.getIntegrationPoints(elementType, "Gauss" + str(2 * p))

    weights, numGaussPoints = np.array(weights), len(weights)

    # sf for shape function (probably)
    _, sf, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
    sf = np.array(sf).reshape((numGaussPoints, -1))

    _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'GradLagrange')
    dsfdu = np.array(dsfdu).reshape((numGaussPoints, Np, 3))[:, :, :-1]

    M = np.einsum("k,ki,kj -> ij", weights, sf, sf)
    D = np.einsum("k,kil,kj ->ijl", weights, dsfdu, sf)

    jacobians, determinants, _ = gmsh.model.mesh.getJacobians(elementType, [1. / 3., 1. / 3., 1. / 3.])
    jacobians = np.array(jacobians).reshape((Nt, 3, 3))
    jacobians = np.swapaxes(jacobians[:, :-1, :-1], 1, 2)

    determinants = np.array(determinants)

    inv_jac = np.empty_like(jacobians)  # trick to inverse 2x2 matrix
    inv_jac[:, 0, 0] = +jacobians[:, 1, 1] / determinants
    inv_jac[:, 0, 1] = -jacobians[:, 1, 0] / determinants
    inv_jac[:, 1, 0] = -jacobians[:, 0, 1] / determinants
    inv_jac[:, 1, 1] = +jacobians[:, 0, 0] / determinants

    # Edge matrix
    elemTypes, _, _ = gmsh.model.mesh.getElements(1, -1)
    edgeType = elemTypes[0]

    uvw_1d, weights_1d = gmsh.model.mesh.getIntegrationPoints(edgeType, "Gauss" + str(2*p))
    weights_1d, numGaussPoints_1d = np.array(weights_1d), len(weights_1d)
    _, sf_1d, _ = gmsh.model.mesh.getBasisFunctions(edgeType, uvw_1d, 'Lagrange')
    sf_1d = np.array(sf_1d).reshape((numGaussPoints_1d, -1))
    M_edge = np.einsum("k,ki,kj -> ij", weights_1d, sf_1d, sf_1d)

    print(M)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()
