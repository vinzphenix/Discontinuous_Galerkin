import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jv
from euler2d import euler2d

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'

alpha = 20.
eps = 1.


def centered_gaussian(x):
    xc = x[:, 0] - 0.
    yc = x[:, 1] - 0.
    l2 = xc ** 2 + yc ** 2
    return eps * np.exp(-alpha * l2)


def solution_comparison(meshfilename, dt, m, u0, v0):
    phi, coords = euler2d(meshfilename, dt, m, 3, u0, v0, centered_gaussian, c0=1, slip_walls=False, plotReturn=True)
    _, _, Nt, Np = phi.shape
    p_numerical = phi[:, 2]
    p_analytical = np.zeros((9, Nt * Np))
    p_min, p_max = np.amin(p_numerical), np.amax(p_numerical)

    integrand = lambda ksi, eta, tau: \
        0.5 * eps / alpha * np.exp(-ksi * ksi / (4 * alpha)) * np.cos(ksi * tau) * jv(0, ksi * eta) * ksi

    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]

    n = 5
    fig, axs = plt.subplots(2, n, figsize=(14., 6.), constrained_layout=True, sharey="all")
    axs_ana = axs[0, :]
    axs_simu = axs[1, :]

    for i in range(n):
        t = (i * m) // (n - 1) * dt
        print(t)
        for k in range(Nt * Np):
            p_analytical[i, k] = quad(integrand, 0., np.inf, args=(
                np.sqrt((coords[k, 0] - u0 * t) ** 2 + (coords[k, 1] - v0 * t) ** 2), t))[0]

        values = p_analytical[i]
        tc = axs_ana[i].tripcolor(coords[:, 0], coords[:, 1], values, cmap=plt.get_cmap('jet'), vmin=p_min, vmax=p_max)
        axs_ana[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_ana[i].set_aspect("equal")

        values = p_numerical[(i * m) // 4].flatten()
        axs_simu[i].tripcolor(coords[:, 0], coords[:, 1], values, cmap=plt.get_cmap('jet'), vmin=p_min, vmax=p_max)
        axs_simu[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_simu[i].set_aspect("equal")

        L = 0.5
        axs_simu[i].axis([-L, L, -L, L])
        axs_ana[i].axis([-L, L, -L, L])
        axs_ana[i].set_title(r"$t={:.3f}$".format(t), fontsize=ftSz2)

        if i == n - 1:
            cbar = fig.colorbar(tc, ax=axs[:, :], location='right', shrink=0.9)
            cbar.ax.set_ylabel(r"$\Delta p$ [Pa]", fontsize=ftSz2)

    fig.savefig("./figures/comparison.png", format="png", bbox_inches='tight')
    fig.savefig("./figures/comparison.svg", format="svg", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    save_global = False
    plt.rcParams["text.usetex"] = save_global

    solution_comparison("./mesh/square_compare.msh", 0.0025, 200, 0, 0)
