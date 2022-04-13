import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys
import os
from matplotlib.animation import FuncAnimation
from scipy.special import roots_legendre
from numpy import pi, sin, sqrt
from tqdm import tqdm
from advection2d import advection2d, initial_Zalezak, velocity_Zalezak, initial_Vortex, velocity_Vortex

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
    return Area, L1


def plot_L1errors(hs, getFromTXT, save):

    Areas = np.zeros((len(hs), 5))
    L1 = np.zeros((len(hs), 5))
    if not getFromTXT:

        for i, h in enumerate(hs):
            for order in range(1,6):
                meshfilename = "./mesh/circle_h" + str(h) + ".msh"
                phi, _ = advection2d(meshfilename, 0.25, 2512, initial_Zalezak, velocity_Zalezak,
                                  order=order, a=1., display=False, animation=False, interactive=False, plotReturn=True)

                gmsh.initialize()
                gmsh.open(meshfilename)
                gmsh.model.mesh.setOrder(order)

                area, l1 = get_Area_and_L1(phi[0], phi[-1], 144.29)
                Areas[i, order-1] = area
                L1[i, order-1] = l1
                gmsh.finalize()

        np.savetxt("./Figures/L1.txt", L1, fmt="%.5f")
        np.savetxt("./Figures/Area.txt", Areas, fmt="%.5f")

    else:
        L1 = np.loadtxt('./Figures/L1.txt')
        Areas = np.loadtxt('./Figures/Area.txt')


    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    for i, h in enumerate(hs):
        ax.plot(np.arange(5) + 1, L1[i], 'o-', label="$h = {}$".format(h))

    print(Areas)
    print(L1)
    ax.legend(fontsize=ftSz3)
    ax.grid(ls=':')
    ax.set_xlabel("order", fontsize=ftSz2)
    ax.set_ylabel("$L_1$ error", fontsize=ftSz2)
    ax.set_yscale("log")

    if save:
        fig.savefig("./Figures/L1_errors.svg", format="svg", bbox_inches='tight')
    plt.show()


iso_zero = lambda x : (np.abs(x) <= 1e-3).astype(int)


def iso_zero_contour(h, order, dt, meshfilename):

    m = int(4//dt)+1
    phi, coords = advection2d(meshfilename, dt, m, initial_Vortex, velocity_Vortex,
                 order=order, a=1., display=False, animation=False, interactive=False, plotReturn=True)

    _, Nt, Np = phi.shape
    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]


    fig, axs = plt.subplots(1, 5, figsize=(18., 4.), constrained_layout=True, sharex="all", sharey="all")
    for i, ax in enumerate(axs.flatten()):
        ax.tricontour(coords[:, 0], coords[:, 1], phi[(i * m) // 5].flatten(), 0, colors="mediumblue")
        ax.set_aspect("equal")
        ax.set_xlabel("t = {} s".format(i))

    fig.suptitle('h = {}, order = {}'.format(h, order))
    plt.show()


if __name__ == "__main__":

    # TAKES AN ETERNITY
    plot_L1errors([2,4,6,8], getFromTXT=True, save=False)

    iso_zero_contour(h=1/20, order=3, dt=0.005, meshfilename="./mesh/square.msh")
