import numpy as np
import matplotlib.pyplot as plt
import gmsh
from advection2d import advection2d, initial_Zalezak, velocity_Zalezak, \
    initial_Vortex, velocity_Vortex, initial_VortexNew

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams['font.family'] = 'monospace'

H_ = lambda x: (x <= 0).astype(int)


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

    print("Area : {} -> % Area loss : {}%".format(Area, (582.2 - Area) / 582.2 * 100))
    return Area, L1


def plot_L1errors(hs, getFromTXT, save):
    Areas = np.zeros((len(hs), 5))
    L1 = np.zeros((len(hs), 5))
    if not getFromTXT:

        for i, h in enumerate(hs):
            for order in range(1, 6):
                meshfilename = "./mesh/circle_h" + str(h) + ".msh"
                phi, _ = advection2d(meshfilename, 0.25, 2512, initial_Zalezak, velocity_Zalezak,
                                     order=order, a=1., display=False, animation=False, interactive=False,
                                     plotReturn=True)

                gmsh.initialize()
                gmsh.open(meshfilename)
                gmsh.model.mesh.setOrder(order)

                area, l1 = get_Area_and_L1(phi[0], phi[-1], 144.29)
                Areas[i, order - 1] = area
                L1[i, order - 1] = l1
                gmsh.finalize()

        np.savetxt("./Figures/L1.txt", L1, fmt="%.5f")
        np.savetxt("./Figures/Area.txt", Areas, fmt="%.5f")

    else:
        L1 = np.loadtxt('./Figures/L1.txt')
        Areas = np.loadtxt('./Figures/Area.txt')

    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    for i, h in enumerate(hs):
        ax.plot(np.arange(5) + 1, L1[i], 'o-', label="$h = {}$".format(h))

    # print(Areas)
    # print(L1)
    ax.legend(fontsize=ftSz3)
    ax.grid(ls=':')
    ax.set_xlabel("order", fontsize=ftSz2)
    ax.set_ylabel("$L_1$ error", fontsize=ftSz2)
    ax.set_yscale("log")

    if save:
        fig.savefig("./Figures/L1_errors.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()


iso_zero = lambda x: (np.abs(x) <= 1e-3).astype(int)


def iso_zero_contour(filename, tend, n_times, orders, dt_list, init, velocity, level, save=False):

    meshfilename = f"./mesh/{filename}.msh"
    fig, axs = plt.subplots(len(orders), n_times, figsize=(10.5, 6.), constrained_layout=True, sharex="all", sharey="all")
    # n_times = len(axs[0])

    for i, (dt, order) in enumerate(zip(dt_list, orders)):
        m = int(np.ceil(tend / dt))
        phi, coords = advection2d(meshfilename, dt, m, init, velocity, interactive=False,
                                  order=order, a=1., display=False, animation=False, plotReturn=True)

        _, Nt, Np = phi.shape
        node_coords = np.empty((3 * Nt, 2))
        for j in range(Nt):
            node_coords[3 * j] = coords[Np * j]
            node_coords[3 * j + 1] = coords[Np * j + 1]
            node_coords[3 * j + 2] = coords[Np * j + 2]

        for j, ax in enumerate(axs[i, :]):
            ax.tricontour(coords[:, 0], coords[:, 1], phi[(j * m) // (n_times-1)].flatten(), [level], colors="mediumblue")
            ax.triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5, color='lightgrey')
            ax.set_aspect("equal")

    for j, ax in enumerate(axs[0, :]):
        ax.set_title("$t = {:.2f}$".format((j * tend) / (n_times - 1)), fontsize=ftSz2)
    for order, ax in zip(orders, axs[:, 0]):
        ax.set_ylabel(r"$p = {:d}$".format(order), fontsize=ftSz2)

    if save:
        fig.savefig(f"./Figures/{filename}.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()


def iso_contour_zalezak(names, tend, n_times, orders, dt_list, level, save=False):

    fig, axs = plt.subplots(len(orders), len(names), figsize=(11., 10.), sharex="all", sharey="all")
    fig.tight_layout()
    cs = [None for k in range(n_times + 1)]

    for i, name in enumerate(names):

        meshfilename = f"./mesh/{name}.msh"

        for j, (ax, dt, order) in enumerate(zip(axs[i, :], dt_list[i], orders)):
            m = int(np.ceil(tend / dt))

            # gmsh.option.set_number("General.Verbosity", 0)
            phi, coords = advection2d(meshfilename, dt, m, initial_Zalezak, velocity_Zalezak, interactive=False,
                                      order=order, a=1., display=False, animation=False, plotReturn=True)

            _, Nt, Np = phi.shape
            node_coords = np.empty((3 * Nt, 2))
            node_coords[0::3] = coords[0::Np]
            node_coords[1::3] = coords[1::Np]
            node_coords[2::3] = coords[2::Np]
            # for k in range(Nt):
            #     node_coords[3 * k] = coords[Np * k]
            #     node_coords[3 * k + 1] = coords[Np * k + 1]
            #     node_coords[3 * k + 2] = coords[Np * k + 2]

            for k in range(n_times+1):
                cs[k] = ax.tricontour(coords[:, 0], coords[:, 1], phi[(k * m) // n_times].flatten(),
                              [level], colors=f"C{k}")
                ax.triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5, color='lightgrey')
                ax.set_aspect("equal")

        for j, (ax, title) in enumerate(zip(axs[:, 0], ["Low", "Mid", "High"])):
            ax.set_ylabel(f"{title} quality", fontsize=ftSz2)  # .format((j * tend) / (n_times - 1))
        for order, ax in zip(orders, axs[0, :]):
            ax.set_title(r"$p = {:d}$".format(order), fontsize=ftSz2)

    # lines = [cs[k].collections[0] for k in range(n_times + 1)]
    # labels = [r'$t={:.1f}$'.format((k * tend) / n_times) for k in range(n_times + 1)]
    # _ = fig.legend(lines, labels, fontsize=ftSz3)

    if save:
        fig.savefig(f"./Figures/zalezak_overview.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    save_global = True
    plt.rcParams["text.usetex"] = save_global

    plot_L1errors([2, 4, 6, 8], getFromTXT=True, save=save_global)  # TAKES AN ETERNITY

    # iso_zero_contour("square_low", 4., n_times=5, orders=[1, 3, 5], dt_list=[0.025, 0.012, 0.0065],
    #                  init=initial_Vortex, velocity=velocity_Vortex, level=0., save=save_global)
    # iso_zero_contour("square_mid", 4., n_times=5, orders=[1, 3, 5], dt_list=[0.010, 0.006, 0.003],
    #                  init=initial_Vortex, velocity=velocity_Vortex, level=0., save=save_global)
    # iso_zero_contour("square_high", 4., n_times=5, orders=[1, 3, 5], dt_list=[0.008, 0.004, 0.002],
    #                  init=initial_Vortex, velocity=velocity_Vortex, level=0., save=save_global)

    # iso_contour_zalezak(["circle_h8", "circle_h6", "circle_h4"], 628., n_times=4, orders=[1, 3, 5], level=0.5,
    #                     dt_list=[[0.5, 0.25, 0.10], [0.5, 0.25, 0.10], [0.5, 0.25, 0.10]], save=save_global)
