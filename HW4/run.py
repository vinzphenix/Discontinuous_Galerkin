import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import jv
import gmsh
from euler2d import euler2d, my_initial_condition

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams['font.family'] = 'monospace'



def solution_comparison(meshfilename, dt, m):

    phi, coords = euler2d(meshfilename, dt, m, 0, 0, my_initial_condition, c0=1, plotReturn=True)
    _, _, Nt, Np = phi.shape
    phi_analytical = np.zeros((9, Nt * Np))
    integrand = lambda ksi, eta, t: 0.05 * np.exp(-ksi ** 2 / 40) * np.cos(ksi * t) * jv(0, ksi * eta) * ksi

    node_coords = np.empty((3 * Nt, 2))
    for i in range(Nt):
        node_coords[3 * i] = coords[Np * i]
        node_coords[3 * i + 1] = coords[Np * i + 1]
        node_coords[3 * i + 2] = coords[Np * i + 2]


    fig = plt.figure(figsize=(14., 10.), constrained_layout=True)
    (subfig1, subfig2) = fig.subfigures(2, 1)
    axs_ana = subfig1.subplots(1, 4, sharey="all")
    axs_simu = subfig2.subplots(1, 4, sharey="all")
    for i in range(4):
        t = (i * m) // 4 * dt
        for k in range(Nt * Np):
            phi_analytical[i, k] = quad(integrand, 0, 10, args=(np.sqrt((coords[k, 0]-0.5)**2 + (coords[k, 1]-0.5)**2), t))[0]

        axs_ana[i].tripcolor(coords[:, 0], coords[:, 1], phi_analytical[i], cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        axs_ana[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_ana[i].set_aspect("equal")

        axs_simu[i].tripcolor(coords[:, 0], coords[:, 1], phi[(i * m) // 4, -1].flatten(), cmap=plt.get_cmap('jet'), vmin=0, vmax=1)
        axs_simu[i].triplot(node_coords[:, 0], node_coords[:, 1], lw=0.5)
        axs_simu[i].set_aspect("equal")

    subfig1.suptitle('Analytical solution')
    subfig2.suptitle('Numerical result')
    plt.show()


def test():
    x = np.linspace(0,15,500)
    plt.plot(x, jv(0, x))
    plt.show()

    integrand = lambda ksi, eta, t: 0.05*np.exp(-ksi ** 2 / 40) * np.cos(ksi * t) * jv(0, ksi * eta) * ksi
    print(quad(integrand, 0, 100, args=(0, 300*0.00002))[0])
    print(quad(integrand, 0, 100, args=(0.5, 300*0.00002))[0])



if __name__ == "__main__":
    save_global = False
    plt.rcParams["text.usetex"] = save_global

    # test()
    solution_comparison("./mesh/square_low.msh", 0.002, 300)


