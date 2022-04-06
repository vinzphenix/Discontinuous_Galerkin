import gmsh
import sys
import numpy as np

gmsh.initialize()

gmsh.open("t4.msh")
gmsh.model.mesh.setOrder(3)
elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)

print(elemTypes)
elementType = elemTypes[0]

name, dim, order, numv, parv, _ = gmsh.model.mesh.getElementProperties(elementType)

prop = gmsh.model.mesh.getElementProperties(elementType)
uvw, weights = gmsh.model.mesh.getIntegrationPoints(
    elementType, "Gauss" + str(2 * prop[2]))

weights = np.array(weights)
numGaussPoints = len(weights)

print('numGaussPoints = g =', numGaussPoints,
      ', %weights (g) =', weights.shape)

numcomp, sf, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
sf = np.array(sf).reshape((numGaussPoints, -1))

numcomp, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'GradLagrange')
dsfdu = np.array(dsfdu).reshape((numGaussPoints, numv, 3))[:, :, :-1]

print('%dsfdu (g,n,u) =', dsfdu.shape)

M = np.zeros((numv, numv))
Dxi = np.zeros((numv, numv))
Deta = np.zeros((numv, numv))

for k in range(numGaussPoints):
    for i in range(numv):
        for j in range(numv):
            M[i, j] += weights[k] * sf[k, i] * sf[k, j]
            Dxi[i, j] += weights[k] * dsfdu[k, i, 0] * sf[k, j]
            Deta[i, j] += weights[k] * dsfdu[k, i, 1] * sf[k, j]

M2 = np.einsum("k,ki,kj -> ji", weights, sf, sf);
D = np.einsum("k,kil,kj ->ijl", weights, dsfdu, sf);

print(D[:, :, 0] - Dxi)
print(D[:, :, 1] - Deta)

# M2 = np.einsum("k,ki,kj->ij", weights, sf, sf)
# D = np.einsum("k,kil,kj->ijl", weights, dsfdu, sf)
#
# print(D)
# print(Dxi - D[:, :, 0])
# print(Deta - D[:, :, 1])
# print(Deta)
