# C:\Users\vince\anaconda3\share\doc\gmsh\tutorials\python
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys
import os
from matplotlib.animation import FuncAnimation
from scipy.special import roots_legendre
from numpy import pi, sin, cos, sqrt
from tqdm import tqdm

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
        dic_edges.setdefault(tag, {"elem": [], "number": [], "normal": None, "length": -1.})
        dic_edges[tag]["elem"].append(i // 3)
        # dic_edges[tag]["elem"].append(elementTags[i // 3])  # equivalent with gmsh tags
        dic_edges[tag]["number"].append(i % 3)

        org, dst = int(edgeNodes[i][0]) - 1, int(edgeNodes[i][1]) - 1
        if dic_edges[tag]["normal"] is None:
            dic_edges[tag]["normal"] = np.array([coords[dst, 1] - coords[org, 1], coords[org, 0] - coords[dst, 0]])
            dic_edges[tag]["normal"] /= np.hypot(*dic_edges[tag]["normal"])  # UNIT normal vector

        dic_edges[tag]["length"] = np.hypot(coords[dst, 0] - coords[org, 0], coords[dst, 1] - coords[org, 1])

    dic_edges_in = {}
    dic_edges_bd = {}
    for key, dic in dic_edges.items():
        if len(dic["elem"]) == 1:
            dic_edges_bd[key] = dic
        else:
            dic_edges_in[key] = dic

    # Populate initial condition
    coordinates_matrix = np.empty((Nt * Np, 2))
    for i, nodeTag in enumerate(elementNodeTags):
        coordinates_matrix[i] = coords[int(nodeTag) - 1]

    # Get nodes indices for each edge number
    # The first two are the corners, the next ones are inside the edge
    nodesIndices1 = [
        [side, (side + 1) % 3] + [3 + side * (order - 1) + i for i in range(order - 1)] for side in range(3)
    ]
    nodesIndices2 = [
        [side, (side + 1) % 3][::-1] + [3 + side * (order - 1) + i for i in range(order - 1)][::-1] for side in range(3)
    ]

    return coordinates_matrix, dic_edges_in, dic_edges_bd, nodesIndices1, nodesIndices2


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

    # sf for shape function
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
    inv_jac[:, 0, 0] = +jacobians[:, 1, 1]
    inv_jac[:, 0, 1] = -jacobians[:, 0, 1]
    inv_jac[:, 1, 0] = -jacobians[:, 1, 0]
    inv_jac[:, 1, 1] = +jacobians[:, 0, 0]

    return Nt, Np, M_matrix, D_matrix, inv_jac, determinants, elementType, order


def get_matrices_edges(elementType, order):
    # Edges matrices
    roots, weights = roots_legendre(2 * order)
    roots, weights, numGaussPoints = (roots + 1) / 2., weights / 2., len(weights)

    uvw = np.c_[roots, np.zeros(numGaussPoints), np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M1 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)

    uvw = np.c_[roots, 1. - roots, np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M2 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)  # Hypotenuse > short sides, but sqrt2 disappears anyway

    uvw = np.c_[np.zeros(numGaussPoints), roots, np.zeros(numGaussPoints)]
    _, sf_edge, _ = gmsh.model.mesh.getBasisFunctions(elementType, list(uvw.flatten()), 'Lagrange')
    sf_edge = np.array(sf_edge).reshape((numGaussPoints, -1))
    M3 = np.einsum("k,ki,kj -> ij", weights, sf_edge, sf_edge)

    return M1, M2, M3


def get_edge_flux_matrix(Nt, Np):
    flux_edge_temp = np.zeros((3, Nt, Np, 3, 3))
    indices = [[[], [], []], [[], [], []]]
    zeros = np.zeros((3,3))

    for edgeTag, dic in edgesInInfo.items():  # loop over all the edges
        elemIn, elemOut = dic["elem"]
        lIn, lOut = dic["number"]

        for nodeIn, nodeOut in zip(nodesIndices_fwd[lIn], nodesIndices_bwd[lOut]):
            An = A1 * dic["normal"][0] + A2 * dic["normal"][1]
            lambdas, L = np.linalg.eig(An)
            lambdas = np.diag(lambdas)
            L_inv = np.linalg.inv(L)

            flux_edge_temp[lIn][elemIn][nodeIn] = L @ np.maximum(zeros, lambdas) @ L_inv * dic["length"]
            flux_edge_temp[lOut][elemOut][nodeOut] = L @ np.minimum(zeros, lambdas) @ L_inv * dic["length"]

            indices[0][0].append(lIn), indices[0][1].append(elemIn), indices[0][2].append(nodeIn)
            indices[1][0].append(lOut), indices[1][1].append(elemOut), indices[1][2].append(nodeOut)


    # Velocity is in fact 0 on boundary
    for edgeTag, dic in edgesBdInfo.items():
        elemIn, = dic["elem"]
        l, = dic["number"]

        for nodeIn in nodesIndices_fwd[l]:
            An = A1 * dic["normal"][0] + A2 * dic["normal"][1]
            lambdas, L = np.linalg.eig(An)
            lambdas = np.diag(lambdas)
            L_inv = np.linalg.inv(L)

            flux_edge_temp[l][elemIn][nodeIn] = L @ np.maximum(zeros, lambdas) @ L_inv * dic["length"]

    indices[0], indices[1] = tuple(indices[0]), tuple(indices[1])
    return flux_edge_temp, indices


def local_dot(phi, a, Nt, Np, t=0.):
    # T = 8.
    # velocity_local = velocity * cos(pi * t / T)

    Fx = np.zeros((Nt, Np, 3))
    Fy = np.zeros((Nt, Np, 3))
    for i in range(Nt):
        for j in range(Np):
            Fx[i,j,:] = A1 @ phi[i,j,:]
            Fy[i,j,:] = A2 @ phi[i,j,:]

    Flux_ksi = np.zeros((Nt, Np, 3))
    Flux_eta = np.zeros((Nt, Np, 3))
    for k in range(3):
        Flux_ksi[:,:,k] = Fx[:,:,k] * IJ[:, 0, 0, np.newaxis] + Fy[:,:,k] * IJ[:, 0, 1, np.newaxis]
        Flux_eta[:,:,k] = Fx[:,:,k] * IJ[:, 1, 0, np.newaxis] + Fy[:,:,k] * IJ[:, 1, 1, np.newaxis]

    Flux_edge = np.zeros((3, Nt, Np, 3))

    """  # slower, but can modulate "a" and can handle vector fields changing in time
    for edgeTag, dic in edgesInInfo.items():  # loop over all the edges inside the domain
        elemIn, elemOut = dic["elem"]
        lIn, lOut = dic["number"]

        for nodeIn, nodeOut in zip(nodesIndices_fwd[lIn], nodesIndices_bwd[lOut]):
            normal_velocity = np.dot(velocity_local[:, elemIn, nodeIn], dic["normal"])

            avg = (phi[elemIn][nodeIn] + phi[elemOut][nodeOut]) * 0.5
            dif = (phi[elemIn][nodeIn] - phi[elemOut][nodeOut]) * 0.5 * np.sign(normal_velocity)
            Flux_edge[lIn][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * dic["length"]
            Flux_edge[lOut][elemOut][nodeOut] = -Flux_edge[lIn][elemIn][nodeIn]  # opposite flux for other element

    for edgeTag, dic in edgesBdInfo.items():  # loop over all the edges that lie on the boundary
        elemIn, = dic["elem"]
        l, = dic["number"]

        for nodeIn in nodesIndices_fwd[l]:
            normal_velocity = np.dot(velocity_local[:, elemIn, nodeIn], dic["normal"])

            avg = (phi[elemIn][nodeIn] + 0.) * 0.5
            dif = (phi[elemIn][nodeIn] - 0.) * 0.5 * np.sign(normal_velocity)
            Flux_edge[l][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * dic["length"]
    """

    for k in range(3):
        for i in range(Nt):
            for j in range(Np):
                Flux_edge[k, i, j, :] = Flux_edge_temp[k, i, j] @ phi[i,j,:]
    Flux_edge[idx[0]] += Flux_edge[idx[1]]
    Flux_edge[idx[1]] = -Flux_edge[idx[0]]
    
    # """

    sum_edge_fluxes = np.array([sum(np.dot(Flux_edge[i, :, :, j], ME[i]) for i in range(3)) for j in range(3)])
    sum_all_fluxes = np.array([np.dot(Flux_ksi[:,:,j], D[0].T) + np.dot(Flux_eta[:,:,j], D[1].T) - sum_edge_fluxes[j] for j in range(3)])

    temp = np.array([np.dot(sum_all_fluxes[j] / det[:, np.newaxis], M_inv) for j in range(3)])
    temp = np.swapaxes(temp, 0, 1)
    temp = np.swapaxes(temp, 1, 2)

    return temp


def fwd_euler(phi, dt, m, a, Nt, Np):
    for i in tqdm(range(m)):
        phi[i + 1] = phi[i] + dt * local_dot(phi[i], a, Nt, Np)

    return phi


def rk22(phi, dt, m, a, Nt, Np):
    for i in tqdm(range(m)):
        K1 = local_dot(phi[i], a, Nt, Np)
        K2 = local_dot(phi[i] + dt / 2. * K1, a, Nt, Np)
        phi[i + 1] = phi[i] + dt * K2

    return phi


def rk44(phi, dt, m, a, Nt, Np):
    for i in tqdm(range(m)):
        t = i * dt
        K1 = local_dot(phi[i], a, Nt, Np, t)
        K2 = local_dot(phi[i] + K1 * dt / 2., a, Nt, Np, t + dt / 2.)
        K3 = local_dot(phi[i] + K2 * dt / 2., a, Nt, Np, t + dt / 2.)
        K4 = local_dot(phi[i] + K3 * dt, a, Nt, Np, t + dt)
        phi[i + 1] = phi[i] + dt * (K1 + 2. * K2 + 2. * K3 + K4) / 6.

    return phi


def euler2d(meshfilename, dt, m, u0, v0, p_init, rktype="RK44", interactive=False,
                order=3, a=1., save=False, plotReturn=False, display=False, animation=False):
    global M_inv, D, ME, IJ, det, edgesInInfo, edgesBdInfo, velocity, nodesIndices_fwd, nodesIndices_bwd, Flux_edge_temp, idx
    global A1, A2, c0, rho

    gmsh.initialize()
    gmsh.open(meshfilename)
    gmsh.model.mesh.setOrder(order)  # this is an option

    c0 = 340
    rho = 1
    A1 = np.array([[u0, 0, 1], [0, u0, 0], [c0**2, 0, u0]])
    A2 = np.array([[v0, 0, 0], [0, v0, 1], [0, c0**2, v0]])

    Nt, Np, M, D, IJ, det, elementType, order = get_matrices()
    M_inv = np.linalg.inv(M)
    ME = get_matrices_edges(elementType, order)
    coordinates_matrix, edgesInInfo, edgesBdInfo, nodesIndices_fwd, nodesIndices_bwd = get_edges_mapping(order, Nt, Np)


    phi = np.zeros((m + 1, Nt * Np, 3))
    phi[0,:,-1] = p_init(coordinates_matrix)
    phi = phi.reshape((m + 1, Nt, Np, 3))

    Flux_edge_temp, idx = get_edge_flux_matrix(Nt, Np)

    if rktype == 'ForwardEuler':
        fwd_euler(phi, dt, m, a, Nt, Np)
    elif rktype == 'RK22':
        rk22(phi, dt, m, a, Nt, Np)
    elif rktype == 'RK44':
        rk44(phi, dt, m, a, Nt, Np)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    if display:
        static_plots(coordinates_matrix, phi[:,:,:,-1], m)
    if animation:
        anim_plots(coordinates_matrix, phi[:,:,:,-1], m)
    if interactive:
        anim_gmsh(elementType, phi[:,:,:,-1], m, dt, save)

    gmsh.finalize()

    if plotReturn:
        return phi, coordinates_matrix
    else:
        return phi[-1]


def static_plots(coords, phi, m):
    _, Nt, Np = phi.shape
    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]
    fig, axs = plt.subplots(3, 3, figsize=(14., 14.), constrained_layout=True, sharex="all", sharey="all")
    for i, ax in enumerate(axs.flatten()):
        # ax.tricontourf(coords[:, 0], coords[:, 1], phi[i].flatten(), cmap=plt.get_cmap('jet'))
        ax.tripcolor(coords[:, 0], coords[:, 1], phi[(i * m) // 8].flatten(), cmap=plt.get_cmap('jet'), vmin=0,
                     vmax=1)
        ax.triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        ax.set_aspect("equal")
    plt.show()


def anim_plots(coords, phi, m):
    def update(t):
        ax.clear()
        return ax.tripcolor(coords[:, 0], coords[:, 1], phi[t].flatten(), cmap=plt.get_cmap('jet'), vmin=0, vmax=1),

    _, Nt, Np = phi.shape
    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]
    fig, ax = plt.subplots(1, 1, figsize=(10., 8.))
    colormap = ax.tripcolor(coords[:, 0], coords[:, 1], phi[0].flatten(), cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
    _ = fig.colorbar(colormap)
    ax.set_aspect("equal")
    fig.tight_layout()

    _ = FuncAnimation(fig, update, m + 1, interval=20, repeat_delay=2000)
    plt.show()


def anim_gmsh(elementType, phi, m, dt, save=False):
    _, Nt, Np = phi.shape
    gmsh.fltk.initialize()
    viewTag = gmsh.view.add("phi")
    modelName = gmsh.model.list()[0]
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)

    if save:
        gmsh.option.set_number("General.Verbosity", 2)
        ratio = 5
        for t in tqdm(range(0, m + 1, ratio)):
            data = phi[t].reshape(Nt * Np, -1)
            gmsh.view.addModelData(viewTag, 0, modelName, "NodeData", elementNodeTags, data, numComponents=1, time=t * dt)

            gmsh.option.set_number("View.RangeType", 2)
            gmsh.option.set_number("View.CustomMin", -0.25)  # -0.025
            gmsh.option.set_number("View.CustomMax", 1.25)  # 0.825

            # gmsh.option.set_number("View.AutoPosition", 0)
            # gmsh.option.set_number("View.PositionX", 125)
            # gmsh.option.set_number("View.PositionY", 450)
            # gmsh.option.set_number("View.OffsetX", 400)
            # gmsh.option.set_number("View.OffsetY", 15)

            gmsh.write(f"./Animations/image_{t // ratio:04d}.jpg")

    else:
        for t in range(0, m + 1, 5):
            data = phi[t].reshape(Nt * Np, -1)
            gmsh.view.addModelData(viewTag, t, modelName, "NodeData", elementNodeTags, data, numComponents=1, time=t * dt)
        gmsh.view.combine("steps", "all")
        gmsh.fltk.run()


def my_initial_condition(x):
    xc = x[:, 0] - 0.5
    yc = x[:, 1] - 0.5
    l2 = xc ** 2 + yc ** 2
    return np.exp(-l2 * 10.)


def my_velocity_condition(x):
    return x * 0. + 1.


if __name__ == "__main__":

    global M_inv, D, ME, IJ, det, edgesInInfo, edgesBdInfo, velocity, \
        nodesIndices_fwd, nodesIndices_bwd, Flux_edge_temp, idx

    euler2d("./mesh/square_low.msh", 0.00002, 100, 200, 0, my_initial_condition, interactive=False,
                order=3, a=1., display=True, animation=False, save=False)



