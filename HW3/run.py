import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys
import os
from matplotlib.animation import FuncAnimation
from scipy.special import roots_legendre
from numpy import pi, sin, sqrt
from tqdm import tqdm
from advection2d import advection2d, initial_Zalezak, velocity_Zalezak

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


H_ = lambda x : (x <= 0).astype(int)


def get_Area_and_L1(phi_0, phi_m, L):
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
    elementType = elemTypes[0]
    Nt = len(elemTags[0])
    name, dim, order, Np, _, _ = gmsh.model.mesh.getElementProperties(elementType)

    uvw, weights = gmsh.model.mesh.getIntegrationPoints(elementType, "Gauss" + str(2 * order))
    weights, numGaussPoints = np.array(weights), len(weights)

    _, sf, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
    sf = np.array(sf).reshape((numGaussPoints, -1))

    _, determinants, _ = gmsh.model.mesh.getJacobians(elementType, [1. / 3., 1. / 3., 1. / 3.])
    determinants = np.array(determinants)

    L1 = 0
    for el in range(Nt):
        L1 += weights @ np.abs(H_(sf @ phi_0[el]) - H_(sf @ phi_m[el])) * determinants[el]

    L1 /= L
    print("L1 error : {}".format(L1))

    Area = 0
    for el in range(Nt):
        Area += weights @ H_(sf @ phi_m[el]) * determinants[el]

    print("Area : {} -> % Area loss : {}%".format(Area, (582.2 - Area)/582.2*100))



if __name__ == "__main__":
    print("hello")

    global M_inv, D, ME, IJ, det, edgesInInfo, edgesBdInfo, velocity,\
        nodesIndices_fwd, nodesIndices_bwd, Flux_edge_temp, idx

    phi = advection2d("./mesh/circle.msh", 0.5, 1256, initial_Zalezak, velocity_Zalezak,
                order=3, a=1., display=False, animation=False, interactive=False)

    meshfilename = "./mesh/circle.msh"
    order = 3
    gmsh.initialize()
    gmsh.open(meshfilename)
    gmsh.model.mesh.setOrder(order)

    get_Area_and_L1(phi[0], phi[-1], 144.29)

    gmsh.finalize()