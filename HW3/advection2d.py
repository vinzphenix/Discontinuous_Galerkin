# C:\Users\vince\anaconda3\share\doc\gmsh\tutorials\python
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys
import os
from scipy.special import roots_legendre
from numpy import pi, sin, sqrt

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def blockPrint():  # Disable
    sys.stdout = open(os.devnull, 'w')


def enablePrint():  # Restore
    sys.stdout = sys.__stdout__


def get_edges_mapping(order, Nt, Np):
    # Find mapping from edge to element
    gmsh.model.mesh.createEdges()
    elementType = gmsh.model.mesh.getElementType("triangle", order)
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)
    edgeNodes = gmsh.model.mesh.getElementEdgeNodes(elementType).reshape(-1, order + 1)  # (order+1) nodes on each edge

    edgeTags, edgeOrientations = gmsh.model.mesh.getEdges(edgeNodes[:, :2].flatten())
    nodeTags, coords, _ = gmsh.model.mesh.getNodes()

    coords = np.array(coords).reshape((-1, 3))[:, :-1]
    dic_edges = {}

    for i, (tag, orientation) in enumerate(zip(edgeTags, edgeOrientations)):  # 3 edges per triangle
        # elemTag = elementTags[i // 3]
        dic_edges.setdefault(tag, {"elem": [], "number": [], "normal": None, "length": -1.})
        dic_edges[tag]["elem"].append(i // 3)  # dic_edges[tag]["elem"].append(elemTag)
        # dic_edges[tag]["orientation"].append(orientation)
        dic_edges[tag]["number"].append(i % 3)

        org, dst = int(edgeNodes[i][0]) - 1, int(edgeNodes[i][1]) - 1
        if dic_edges[tag]["normal"] is None:
            dic_edges[tag]["normal"] = np.array([coords[dst, 1] - coords[org, 1], coords[org, 0] - coords[dst, 0]])
            dic_edges[tag]["normal"] /= np.hypot(*dic_edges[tag]["normal"])  # UNIT normal vector

        # dic_edges[tag]["orientation"].append(orientation)
        dic_edges[tag]["length"] = np.hypot(coords[dst, 0] - coords[org, 0], coords[dst, 1] - coords[org, 1])
        # print(tag, edgeNodes[i], coords[org, 0], coords[org, 1], coords[dst, 0], coords[dst, 1])

    # for key, val in dic_edges.items():
    #     print(key, val)

    # Populate initial condition
    coordinates_matrix = np.empty((Nt * Np, 2))
    for i, nodeTag in enumerate(elementNodeTags):
        coordinates_matrix[i] = coords[int(nodeTag) - 1]

    return coordinates_matrix, dic_edges


def get_matrices():
    # Matrices M and D, and jacobians
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(2, -1)
    elementType = elemTypes[0]
    Nt = len(elemTags[0])

    # Name (?), dimension, order, nb vertices / element
    name, dim, order, Np, _, _ = gmsh.model.mesh.getElementProperties(elementType)

    # location of gauss points in 3d space, and associated weights
    uvw, weights = gmsh.model.mesh.getIntegrationPoints(elementType, "Gauss" + str(2 * order))
    weights, numGaussPoints = np.array(weights), len(weights)
    # print(numGaussPoints, order, uvw)

    # sf for shape function (probably)
    _, sf, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
    sf = np.array(sf).reshape((numGaussPoints, -1))

    _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'GradLagrange')
    dsfdu = np.array(dsfdu).reshape((numGaussPoints, Np, 3))[:, :, :-1]

    M_matrix = np.einsum("k,ki,kj -> ij", weights, sf, sf)
    D_matrix = np.einsum("k,kil,kj ->lij", weights, dsfdu, sf)

    jacobians, determinants, _ = gmsh.model.mesh.getJacobians(elementType, [1. / 3., 1. / 3., 1. / 3.])
    jacobians = np.array(jacobians).reshape((Nt, 3, 3))
    jacobians = np.swapaxes(jacobians[:, :-1, :-1], 1, 2)

    determinants = np.array(determinants)

    inv_jac = np.empty_like(jacobians)  # trick to inverse 2x2 matrix
    inv_jac[:, 0, 0] = +jacobians[:, 1, 1] / determinants
    inv_jac[:, 0, 1] = -jacobians[:, 1, 0] / determinants
    inv_jac[:, 1, 0] = -jacobians[:, 0, 1] / determinants
    inv_jac[:, 1, 1] = +jacobians[:, 0, 0] / determinants

    return Nt, Np, M_matrix, D_matrix, inv_jac, determinants, elementType, order


def get_matrices_edges(elementType, order):
    # Edges matrices
    roots, weights = roots_legendre(2 * order)
    roots, weights, numGaussPoints = (roots + 1) / 2., weights / 2., len(weights)

    uvw = np.c_[roots, np.zeros(numGaussPoints), np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M1 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)
    # np.savetxt("M1.txt", M1, fmt="%.5f")

    uvw = np.c_[roots, 1. - roots, np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M2 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)  # Hypotenuse > short sides, but sqrt2 disappears anyway
    # np.savetxt("M2.txt", M2, fmt="%.5f")

    uvw = np.c_[np.zeros(numGaussPoints), roots, np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M3 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)
    # np.savetxt("M3.txt", M3, fmt="%.5f")

    return M1, M2, M3


def local_dot(phi, a, Nt, Np, order):
    Fx = velocity[0] * phi
    Fy = velocity[1] * phi
    Flux_ksi = Fx * IJ[:, 0, 0, np.newaxis] + Fy * IJ[:, 0, 1, np.newaxis]
    Flux_eta = Fx * IJ[:, 1, 0, np.newaxis] + Fy * IJ[:, 1, 1, np.newaxis]
    Flux_edge = np.zeros((3, Nt, Np))

    # TODO Handle boundary conditions
    for edgeTag, dic in edgesInfo.items():  # loop over all the edges
        if len(dic["elem"]) == 2:
            elemIn, elemOut = dic["elem"]
            l, _ = dic["number"]
            length = dic["length"]

            cornerNodes = [l, (l + 1) % 3]
            edgeNodes = [3 + l * (order - 1) + i for i in range(order - 1)]
            nodesThisElem = cornerNodes + edgeNodes
            nodesNextElem = cornerNodes[::-1] + edgeNodes[::-1]

            # print(edgeTag, nodesThisElem, nodesNextElem)

            for nodeIn, nodeOut in zip(nodesThisElem, nodesNextElem):  # TODO: correctly loop over nodes
                normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])
                avg = (phi[elemIn][nodeIn] + phi[elemOut][nodeOut]) * 0.5
                dif = (phi[elemIn][nodeIn] - phi[elemOut][nodeOut]) * 0.5 * np.sign(normal_velocity)
                Flux_edge[l][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * length

                if normal_velocity > 0:  # take phi value inside
                    Flux_edge[l][elemIn][nodeIn] = phi[elemIn][nodeIn] * normal_velocity * length  # TODO: take correct node
                else:  # take phi value outside
                    Flux_edge[l][elemIn][nodeIn] = phi[elemOut][nodeOut] * normal_velocity * length  # TODO: take correct node

                # TODO: add opposite flux to the other triangle
                Flux_edge[l][elemOut][nodeOut] = -Flux_edge[l][elemIn][nodeIn]
        else:
            elemIn, = dic["elem"]
            l, = dic["number"]
            length = dic["length"]

            cornerNodes = [l, (l + 1) % 3]
            edgeNodes = [3 + l * (order - 1) + i for i in range(order - 1)]
            nodesThisElem = cornerNodes + edgeNodes

            # print("BOUNDARY", edgeTag, nodesThisElem)

            for nodeIn in nodesThisElem:
                normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])
                # avg = (phi[elemIn][nodeIn] + 0.) * 0.5
                # dif = (phi[elemIn][nodeIn] - 0.) * 0.5 * np.sign(normal_velocity)
                # Flux_edge[l][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * length

                if normal_velocity > 0:  # take phi value inside
                    Flux_edge[l][elemIn][nodeIn] = phi[elemIn][nodeIn] * normal_velocity * length
                else:  # take phi value outside
                    Flux_edge[l][elemIn][nodeIn] = 0. * normal_velocity * length

    # sum_edge_flux = sum(np.dot(Flux_edge[i], ME[i]) for i in range(3))
    sum_edge_flux = np.dot(Flux_edge[0], ME[0]) + np.dot(Flux_edge[1], ME[1]) + np.dot(Flux_edge[2], ME[2])
    sum_all_flux = np.dot(Flux_ksi, D[0]) + np.dot(Flux_eta, D[1]) - 1. / det[:, np.newaxis] * sum_edge_flux
    return np.dot(sum_all_flux, M_inv)


def fwd_euler(phi, dt, m, a, Nt, Np, order):
    for i in range(m):
        phi[i + 1] = phi[i] + dt * local_dot(phi[i], a, Nt, Np, order)

    return phi


def rk22(phi, dt, m, a, Nt, Np, order):
    for i in range(m):
        K1 = local_dot(phi[i], a, Nt, Np, order)
        K2 = local_dot(phi[i] + dt / 2. * K1, a, Nt, Np, order)
        phi[i + 1] = phi[i] + dt * K2

    return phi


def rk44(phi, dt, m, a, Nt, Np, order):
    for i in range(m):
        K1 = local_dot(phi[i], a, Nt, Np, order)
        K2 = local_dot(phi[i] + K1 * dt / 2., a, Nt, Np, order)
        K3 = local_dot(phi[i] + K2 * dt / 2., a, Nt, Np, order)
        K4 = local_dot(phi[i] + K3 * dt, a, Nt, Np, order)
        phi[i + 1] = phi[i] + dt * (K1 + 2. * K2 + 2. * K3 + K4) / 6.

    return phi


def advection2d(meshfilename, dt, m, f, u, order=3, rktype="RK44", a=1., interactive=False):
    global M_inv, D, ME, IJ, det, edgesInfo, velocity

    gmsh.initialize()
    gmsh.open(meshfilename)
    gmsh.model.mesh.setOrder(order)  # this is an option

    Nt, Np, M, D, IJ, det, elementType, order = get_matrices()
    M_inv = np.linalg.inv(M)
    ME = get_matrices_edges(elementType, order)
    coordinates_matrix, edgesInfo = get_edges_mapping(order, Nt, Np)

    velocity = np.array(u(coordinates_matrix)).reshape((2, Nt, Np))

    phi = np.zeros((m+1, Nt * Np))
    phi[0] = f(coordinates_matrix)
    phi = phi.reshape((m+1, Nt, Np))

    if rktype == 'ForwardEuler':
        fwd_euler(phi, dt, m, a, Nt, Np, order)
    elif rktype == 'RK22':
        rk22(phi, dt, m, a, Nt, Np, order)
    elif rktype == 'RK44':
        rk44(phi, dt, m, a, Nt, Np, order)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    # coords = coordinates_matrix
    # fig, axs = plt.subplots(1, 3, figsize=(14., 5.), constrained_layout=True)
    # for ax, idx in zip(axs, [0, m//2, m]):
    #     ax.tricontourf(coords[:, 0], coords[:, 1], phi[idx].flatten(), cmap=plt.get_cmap('jet'))
    #     ax.triplot(coords[:, 0], coords[:, 1])
    #     ax.set_aspect("equal")
    # plt.show()

    if interactive:
        t_idx = 1
        viewTag = gmsh.view.add("stress-data")
        modelName = gmsh.model.list()[0]
        data = phi[t_idx].reshape(Nt * Np, -1)
        elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)
        gmsh.view.addModelData(viewTag, 0, modelName, "NodeData", elementNodeTags, data, time=0.)

    if '-nopopup' not in sys.argv:
        gmsh.fltk.run()
    gmsh.finalize()

    return


def my_initial_condition(x):
    xc = x[:, 0] - 1. / 2.
    yc = x[:, 1] - 1. / 2.
    l2 = xc ** 2 + yc ** 2
    return np.exp(-l2 * 10.)


def my_velocity_condition(x):
    # return np.c_[np.ones_like(x[:, 0]), np.zeros_like(x[:, 1])]
    return x * 0. + 1.


def initial_Zalezak(x):
    """
    Ugly function, but handles correctly the distances in the corners
    """
    xc, yc, r, h, w = 50., 75, 15., 25., 5.
    y_top, y_bot = yc - r + h, yc - sqrt(r ** 2 - (0.5 * w) ** 2)
    xl, xr = xc - w / 2., xc + w / 2.
    phi_zero = 0. * x[:, 0]

    for i, (xi, yi) in enumerate(zip(x[:, 0], x[:, 1])):
        d2center = np.hypot(xi - xc, yi - yc)
        if d2center > r:
            if (yi - y_bot) > (yc - y_bot) / (xc - xl) * (xi - xl):
                d = d2center - r
            elif (yi - y_bot) > (yc - y_bot) / (xc - xr) * (xi - xr):
                d = d2center - r
            else:
                d = min(np.hypot(xi - xl, yi - y_bot), np.hypot(xi - xr, yi - y_bot))
        elif yi < y_bot:
            d = min(np.hypot(xi - xl, yi - y_bot), np.hypot(xi - xr, yi - y_bot))
        elif yi < y_top:
            if xl < xi < xr:
                d = min(xi - xl, xr - xi, y_top - yi)
            elif xi < xl:
                d = max(xi - xl, d2center - r)  # negative values -> max gives smallest absolute value
            else:
                d = max(xr - xi, d2center - r)  # negative values -> max gives smallest absolute value
        else:
            if xl < xi < xr:
                d = max(y_top - yi, d2center - r)
            elif xi < xl:
                d = max(-np.hypot(xi - xl, yi - y_top), d2center - r)
            else:
                d = max(-np.hypot(xi - xr, yi - y_top), d2center - r)

        alpha = 3
        phi_zero[i] = np.tanh(alpha * d)

    return phi_zero


def velocity_Zalezak(x):
    u = pi / 314. * (50. - x[:, 1])
    v = pi / 314. * (x[:, 0] - 50.)
    return np.c_[u, v]


def initial_Vortex(x):
    return (x[:, 0] - 0.5) ** 2 + (x[:, 1] - 0.5) ** 2 - 0.15 ** 2


def velocity_Vortex(x):
    u = sin(2. * pi * x[:, 1]) * sin(pi * x[:, 0]) ** 2
    v = sin(2. * pi * x[:, 0]) * sin(pi * x[:, 1]) ** 2
    return np.c_[u, v]


def spyMatrices():
    M1 = np.loadtxt("M1.txt")
    M2 = np.loadtxt("M2.txt")
    M3 = np.loadtxt("M3.txt")

    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    ax.spy(M1, markersize=10, color='C0', alpha=0.5)
    ax.spy(M2, markersize=7.5, color='C1', alpha=0.75)
    ax.spy(M3, markersize=5, color='C2')
    plt.show()


if __name__ == "__main__":
    global M_inv, D, ME, IJ, det, edgesInfo, velocity

    advection2d("t2.msh", 0.01, 5, my_initial_condition, my_velocity_condition, order=3, a=1., interactive=True)
    # spyMatrices()
