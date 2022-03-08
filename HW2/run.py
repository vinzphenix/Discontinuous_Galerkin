import matplotlib.pyplot as plt
import numpy as np
from scipy.special import legendre
from maxwell1d import maxwell1d, table

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def compare_solutions(glass=False, bctype="periodic", name="", save=False):
    L, n, p = 3e8, 10, 3
    c = 1. / (np.sqrt(eps0) * np.sqrt(mu0))
    dt = 0.5 * table[p][3] / c * L / n
    dt = 0.005
    E0, H0 = E1_global, H1_global
    m = int(2. / dt)

    eps, mu = eps0 * np.ones(2 * n), mu0 * np.ones(2 * n)
    if glass:
        eps[n // 2:3 * n // 2] *= 5  # permittivity of glass differs, but the rest stays the same

    u = maxwell1d(L, E1_global, H1_global, n, eps, mu, dt, m, p, 'RK44', bctype=bctype, a=1., anim=False)

    n_plot = 25
    E = np.zeros((2 * n, m + 1, n_plot + 1))
    H = np.zeros((2 * n, m + 1, n_plot + 1))
    r = np.linspace(-1, 1, n_plot + 1)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    dx = L / n
    full_x = np.linspace(-L, L, 2 * n * n_plot + 1).flatten()
    full_sqrt_eps = np.sqrt(np.r_[eps[0], np.repeat(eps, n_plot)])
    full_sqrt_mu = np.sqrt(np.r_[mu[0], np.repeat(mu, n_plot)])

    for time in range(m + 1):
        for elem in range(2 * n):
            E[elem, time] = np.dot(psi, u[0, :, elem, time])
            H[elem, time] = np.dot(psi, u[1, :, elem, time])

    t_array = np.linspace(0, m * dt, m + 1)
    if not glass:
        times = [np.argmin(np.abs(t_array - t)) for t in [0.25, 0.75, 1., 1.25]]
    else:
        times = [np.argmin(np.abs(t_array - t)) for t in [0.25, 1., 1.5, 2.]]

    fig, axs = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True, sharex='all', sharey='col')

    for i, t_idx in enumerate(times):
        t = t_idx * dt
        v_1 = 0.5 / full_sqrt_mu * E0(full_x - c * t, L) + 0.5 / full_sqrt_eps * H0(full_x - c * t, L)
        v_2 = 0.5 / full_sqrt_mu * E0(full_x + c * t, L) - 0.5 / full_sqrt_eps * H0(full_x + c * t, L)

        ax1, ax2 = axs[i, 0], axs[i, 1]
        if not glass:
            ax1.plot(full_x / c, full_sqrt_mu * (v_1 + v_2), label='E Analytical', color='C1', lw=5, alpha=0.3)
            ax2.plot(full_x / c, full_sqrt_eps * (v_1 - v_2), label='H Analytical', color='C0', lw=5, alpha=0.3)

        for elem in range(2 * n):
            middle = dx * (elem + 1. / 2.) - L
            x = (middle + r * dx / 2) / c
            (labelE, labelH) = ('E numerical', 'H numerical') if elem == 0 else ('', '')
            ax1.plot(x, np.dot(psi, u[0, :, elem, t_idx]), color='C1', marker='.', markevery=[0, -1], label=labelE)
            ax2.plot(x, np.dot(psi, u[1, :, elem, t_idx]), color='C0', marker='.', markevery=[0, -1], label=labelH)

        ax1.grid(ls=':')
        ax2.grid(ls=':')
        ax1.set_ylabel("E [V/m]".format(dt * t_idx), fontsize=ftSz2)
        ax2.set_ylabel("H [A/m]", fontsize=ftSz2)
        ax1.text(0.03, 0.92, r"$t = {:.2f} \; s$".format(t_idx * dt), transform=ax1.transAxes, fontsize=ftSz2,
                 verticalalignment='top', bbox=dict(facecolor='lightgrey', alpha=0.35))

    limitEm, limitEp = np.amin([ax.get_ylim()[0] for ax in axs[:, 0]]), np.amax([ax.get_ylim()[1] for ax in axs[:, 0]])
    limitHm, limitHp = np.amin([ax.get_ylim()[0] for ax in axs[:, 1]]), np.amax([ax.get_ylim()[1] for ax in axs[:, 1]])
    if glass:
        for ax1, ax2 in zip(axs[:, 0], axs[:, 1]):
            ax1.set_ylim([limitEm, limitEp])
            ax2.set_ylim([limitHm, limitHp])
            ax1.add_patch(plt.Rectangle((-L / 2. / c, -Z), L / c, 2 * Z, facecolor="grey", alpha=0.25))
            ax2.add_patch(plt.Rectangle((-L / 2. / c, -1.), L / c, 2., facecolor="grey", alpha=0.25))

    axs[-1, 0].set_xlabel("x [light-second]", fontsize=ftSz2)
    axs[-1, 1].set_xlabel("x [light-second]", fontsize=ftSz2)
    lines_labels = [axs[0, i].get_legend_handles_labels() for i in range(2)]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lgd = fig.legend(lines, labels, labelspacing=2.5, bbox_to_anchor=(0.15, -1., 0.7, 1.), mode='expand',
                     ncol=4, facecolor='wheat', framealpha=0.25, fancybox=True, fontsize=ftSz3)

    if save:
        fig.savefig(f"./Figures/{name}.svg", format="svg", bbox_inches='tight', bbox_extra_artists=(lgd,))
    else:
        plt.show()


if __name__ == "__main__":
    save_global = False
    plt.rcParams["text.usetex"] = save_global

    eps0, mu0 = 8.8541878128e-12, 4.e-7 * np.pi
    Z = np.sqrt(mu0 / eps0)
    E1_global = lambda x, L: 0 * x
    H1_global = lambda x, L: np.exp(-(10 * x / L) ** 2)

    # compare_solutions(bctype="reflective", name="comparison_reflective", save=False)
    # compare_solutions(bctype="non-reflective", name="comparison_infinite", save=False)
    compare_solutions(glass=True, bctype="non-reflective", name="glass_block", save=False)
