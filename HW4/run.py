import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jv
import gmsh
from euler2d import euler2d, my_initial_condition

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams['font.family'] = 'monospace'


def init_centered_gaussian(x):
    xc = x[:, 0] - 0.5
    yc = x[:, 1] - 0.5
    l2 = xc ** 2 + yc ** 2
    alpha = 20
    eps = 1
    return eps * np.exp(-alpha * l2)


def solution_comparison(meshfilename, dt, m, u0, v0):
    phi, coords = euler2d(meshfilename, dt, m, u0, v0, init_centered_gaussian, c0=1, slip_walls=False, plotReturn=True)
    _, _, Nt, Np = phi.shape
    phi_analytical = np.zeros((9, Nt * Np))

    alpha = 20
    eps = 1
    integrand = lambda ksi, eta, t: eps / 2 / alpha * np.exp(-ksi ** 2 / 4 / alpha) * np.cos(ksi * t) * jv(0, ksi * eta) * ksi

    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]

    n = 5
    fig = plt.figure(figsize=(14., 10.), constrained_layout=True)
    (subfig1, subfig2) = fig.subfigures(2, 1)
    axs_ana = subfig1.subplots(1, n, sharey="all")
    axs_simu = subfig2.subplots(1, n, sharey="all")
    for i in range(n):
        t = (i * m) // n * dt
        for k in range(Nt * Np):
            phi_analytical[i, k] = quad(integrand, 0, 20, args=(
            np.sqrt((coords[k, 0] - 0.5 - u0 * t) ** 2 + (coords[k, 1] - 0.5 - v0 * t) ** 2), t))[0]

        axs_ana[i].tripcolor(coords[:, 0], coords[:, 1], phi_analytical[i], cmap=plt.get_cmap('jet'),
                             vmin=np.amin(phi), vmax=np.amax(phi))
        axs_ana[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_ana[i].set_aspect("equal")

        axs_simu[i].tripcolor(coords[:, 0], coords[:, 1], phi[(i * m) // 4, -1].flatten(), cmap=plt.get_cmap('jet'),
                              vmin=np.amin(phi), vmax=np.amax(phi))
        axs_simu[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_simu[i].set_aspect("equal")

    subfig1.suptitle('Analytical solution')
    subfig2.suptitle('Numerical result')
    plt.show()
    #fig.savefig("./figures/comparison.svg", format="svg", bbox_inches='tight')


def test():
    x = np.linspace(0, 15, 500)
    #plt.plot(x, jv(0, x))
    #plt.show()

    alpha = 10
    integrand = lambda ksi, eta, t: 0.5 / alpha * np.exp(-ksi ** 2 / 4 / alpha) * np.cos(ksi * t) * jv(0, ksi * eta) * ksi
    print(quad(integrand, 0, 100, args=(0.2, 0))[0])
    print(quad(integrand, 0, 20, args=(0.0196, 0))[0])
    print(quad(integrand, 0, 100, args=(0, 300 * 0.002))[0])
    print(quad(integrand, 0, 100, args=(0.5, 300 * 0.00002))[0])


if __name__ == "__main__":
    save_global = False
    plt.rcParams["text.usetex"] = save_global

    # test()
    solution_comparison("./mesh/square_mid.msh", 0.002, 300, 0, 0)
