# C:\Users\vince\anaconda3\share\doc\gmsh\tutorials\python
import numpy as np
import matplotlib.pyplot as plt
import gmsh
import sys
import os
from matplotlib.animation import FuncAnimation
from scipy.special import roots_legendre
from numpy import pi, sin, sqrt
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

    # print(edgeNodes)

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
    # print(numGaussPoints, order, uvw)

    # sf for shape function (probably)
    _, sf, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'Lagrange')
    sf = np.array(sf).reshape((numGaussPoints, -1))

    _, dsfdu, _ = gmsh.model.mesh.getBasisFunctions(elementType, uvw, 'GradLagrange')
    dsfdu = np.array(dsfdu).reshape((numGaussPoints, Np, 3))[:, :, :-1]

    M_matrix = np.einsum("k,ki,kj -> ij", weights, sf, sf)
    D_matrix = np.einsum("k,kil,kj ->lij", weights, dsfdu, sf)
    # np.savetxt("M.txt", M_matrix, fmt="%.5f")
    # np.savetxt("D.txt", D_matrix[0], fmt="%7.4f")

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


def get_edge_flux_matrix(Nt, Np):
    Flux_edge_temp = np.zeros((3, Nt, Np))
    idx = [[[], [], []], [[], [], []]]

    for edgeTag, dic in edgesInInfo.items():  # loop over all the edges
        elemIn, elemOut = dic["elem"]
        lIn, lOut = dic["number"]

        for nodeIn, nodeOut in zip(nodesIndices_fwd[lIn], nodesIndices_bwd[lOut]):
            normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])

            Flux_edge_temp[lIn][elemIn][nodeIn] = (normal_velocity > 0).astype(int) * normal_velocity * dic["length"]
            Flux_edge_temp[lOut][elemOut][nodeOut] = -(normal_velocity < 0).astype(int) * normal_velocity * dic["length"]

            if np.sign(normal_velocity) > 0:
                idx[0][0].append(lOut), idx[0][1].append(elemOut), idx[0][2].append(nodeOut)
                idx[1][0].append(lIn),  idx[1][1].append(elemIn),  idx[1][2].append(nodeIn)
            else:
                idx[0][0].append(lIn),  idx[0][1].append(elemIn),  idx[0][2].append(nodeIn)
                idx[1][0].append(lOut), idx[1][1].append(elemOut), idx[1][2].append(nodeOut)

    for edgeTag, dic in edgesBdInfo.items():
        elemIn, = dic["elem"]
        l, = dic["number"]

        for nodeIn in nodesIndices_fwd[l]:
            normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])
            Flux_edge_temp[l][elemIn][nodeIn] = (normal_velocity > 0).astype(int) * normal_velocity * dic["length"]

    idx[0], idx[1] = tuple(idx[0]), tuple(idx[1])
    return Flux_edge_temp, idx


def local_dot(phi, a, Nt, Np):
    Fx = velocity[0] * phi
    Fy = velocity[1] * phi
    Flux_ksi = Fx * IJ[:, 0, 0, np.newaxis] + Fy * IJ[:, 0, 1, np.newaxis]
    Flux_eta = Fx * IJ[:, 1, 0, np.newaxis] + Fy * IJ[:, 1, 1, np.newaxis]
    Flux_edge = np.zeros((3, Nt, Np))

    """ IDEA for improvement : needs 6 masks (l = 1,2,3 and 2 elems/side)
    avg = phi[mask_l_A] + phi[mask_l_B]
    dif = (phi[mask_l_A] - phi[mask_l_B]) * sign(normal_v)
    Flux_edge[l][mask_l_A] = 0.5 * (avg + a dif) * normal_v * length
    Flux_edge[l][mask_l_B] = -Flux_edge[l][mask_l_A]
    """

    """  # slower, but can modulate "a" and varying vector fields
    for edgeTag, dic in edgesInInfo.items():  # loop over all the edges inside the domain
        elemIn, elemOut = dic["elem"]
        lIn, lOut = dic["number"]

        for nodeIn, nodeOut in zip(nodesIndices_fwd[lIn], nodesIndices_bwd[lOut]):
            normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])

            avg = (phi[elemIn][nodeIn] + phi[elemOut][nodeOut]) * 0.5
            dif = (phi[elemIn][nodeIn] - phi[elemOut][nodeOut]) * 0.5 * np.sign(normal_velocity)
            Flux_edge[lIn][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * dic["length"]
            Flux_edge[lOut][elemOut][nodeOut] = -Flux_edge[lIn][elemIn][nodeIn]  # opposite flux for other element

    for edgeTag, dic in edgesBdInfo.items():  # loop over all the edges that lie on the boundary
        elemIn, = dic["elem"]
        l, = dic["number"]

        for nodeIn in nodesIndices_fwd[l]:
            normal_velocity = np.dot(velocity[:, elemIn, nodeIn], dic["normal"])

            avg = (phi[elemIn][nodeIn] + 0.) * 0.5
            dif = (phi[elemIn][nodeIn] - 0.) * 0.5 * np.sign(normal_velocity)
            Flux_edge[l][elemIn][nodeIn] = (avg + a * dif) * normal_velocity * dic["length"]
    """
    
    for i in range(3):
        Flux_edge[i] = Flux_edge_temp[i] * phi
    Flux_edge[idx[0]] = -Flux_edge[idx[1]]

    sum_edge_flux = sum(np.dot(Flux_edge[i], ME[i]) for i in range(3))
    sum_all_flux = np.dot(Flux_ksi, D[0].T) + np.dot(Flux_eta, D[1].T) - sum_edge_flux
    return np.dot(sum_all_flux / det[:, np.newaxis], M_inv)


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
        K1 = local_dot(phi[i], a, Nt, Np)
        K2 = local_dot(phi[i] + K1 * dt / 2., a, Nt, Np)
        K3 = local_dot(phi[i] + K2 * dt / 2., a, Nt, Np)
        K4 = local_dot(phi[i] + K3 * dt, a, Nt, Np)
        phi[i + 1] = phi[i] + dt * (K1 + 2. * K2 + 2. * K3 + K4) / 6.

    return phi


def advection2d(meshfilename, dt, m, f, u, order=3, rktype="RK44", a=1., display=False, animation=False, interactive=False):
    global M_inv, D, ME, IJ, det, edgesInInfo, edgesBdInfo, velocity, nodesIndices_fwd, nodesIndices_bwd, Flux_edge_temp, idx

    gmsh.initialize()
    gmsh.open(meshfilename)
    gmsh.model.mesh.setOrder(order)  # this is an option

    Nt, Np, M, D, IJ, det, elementType, order = get_matrices()
    M_inv = np.linalg.inv(M)
    ME = get_matrices_edges(elementType, order)
    coordinates_matrix, edgesInInfo, edgesBdInfo, nodesIndices_fwd, nodesIndices_bwd = get_edges_mapping(order, Nt, Np)

    velocity = np.array(u(coordinates_matrix)).reshape((Nt, Np, 2))
    velocity = np.swapaxes(velocity, 0, 2)
    velocity = np.swapaxes(velocity, 1, 2)

    phi = np.zeros((m + 1, Nt * Np))
    phi[0] = f(coordinates_matrix)
    phi = phi.reshape((m + 1, Nt, Np))

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
        static_plots(coordinates_matrix, phi, m)
    if animation:
        anim_plots(coordinates_matrix, phi, m)
    if interactive:
        anim_gmsh(elementType, phi, m, dt)

    gmsh.finalize()

    return


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


def anim_gmsh(elementType, phi, m, dt):
    _, Nt, Np = phi.shape
    print(phi.shape)
    gmsh.fltk.initialize()
    viewTag = gmsh.view.add("scalar_field")
    modelName = gmsh.model.list()[0]
    elementTags, elementNodeTags = gmsh.model.mesh.getElementsByType(elementType)

    for t in range(m+1):
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


def velocity_special(x):
    x, y = x[:, 0], x[:, 1]
    # u = -10. * x ** 2 * y * (1 - x) ** 2 * (1 - y)
    # v = +10. * x * y ** 2 * (1 - x) * (1 - y) ** 2
    # return np.c_[u, v]
    return np.c_[y, -x]


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
        phi_zero[i] = -np.tanh(alpha * d)

    return phi_zero


def velocity_Zalezak(x):
    x, y = x[:, 0], x[:, 1]
    u = pi / 314. * (50. - y)
    v = pi / 314. * (x - 50.)
    return np.c_[u, v]


def initial_Vortex(x):
    x, y = x[:, 0], x[:, 1]
    return (x - 0.5) ** 2 + (y - 0.15) ** 2 - 0.15 ** 2


def velocity_Vortex(x):
    x, y = x[:, 0], x[:, 1]
    u = +sin(2. * pi * y) * sin(pi * x) ** 2
    v = -sin(2. * pi * x) * sin(pi * y) ** 2
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
    # TODO: (1) vectorize "local_dot" a bit more because currently too slow
    # TODO: (2) adapt for vector fields with divergence

    global M_inv, D, ME, IJ, det, edgesInInfo, edgesBdInfo, velocity,\
        nodesIndices_fwd, nodesIndices_bwd, Flux_edge_temp, idx

    advection2d("./mesh/square.msh", 0.003, 300, my_initial_condition, my_velocity_condition,
                order=3, a=1., display=False, animation=False, interactive=True)

    # advection2d("./mesh/square.msh", 0.005, 300, initial_Vortex, velocity_Vortex,
    #             order=3, a=1., display=False, animation=False, interactive=True)

    # advection2d("./mesh/circle.msh", 0.5, 150, initial_Zalezak, velocity_Zalezak,
    #             order=3, a=1., display=False, animation=True, interactive=False)
