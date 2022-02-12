import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.special import legendre
from advection1d import advection1d, table

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'


def compute_energy(L, p, n, m, u):
    w = [+0.5688888888888889, +0.4786286704993665, +0.4786286704993665, +0.2369268850561891, +0.2369268850561891]
    s = [+0.0000000000000000, -0.5384693101056831, +0.5384693101056831, -0.9061798459386640, +0.9061798459386640]
    dx = L / n
    psi = [legendre(i) for i in range(p + 1)]

    energy = np.zeros(m)

    for j in range(m):
        for k in range(n):
            for l in range(5):
                tmp = 0.
                for i in range(p + 1):
                    tmp += u[i, k, j] * psi[i](s[l])
                energy[j] += w[l] * dx / 2. * tmp ** 2

    return energy


def plot_energy(save=False):
    L, n, p, c, m = 1., 10, 3, 1., 1000
    dt = 0.15 * table[p][3] / c * L / n
    m = int(10. / dt)
    print(f"{dt:.3e}")

    f1 = lambda x: np.sin(2. * np.pi * x / L)
    f2 = lambda x: np.arctan(np.abs(np.tan(np.pi * x / L))) * 2. / np.pi
    f3 = lambda x: np.heaviside(np.fmod(np.fmod(x / L, 1.) + 1, 1.) - 0.5, 0.)
    f_list = [f1, f2, f3]

    t_array = np.linspace(0, m * dt, m)
    x_array = np.linspace(0, L, 500)

    fig, axs = plt.subplots(3, 2, figsize=(10, 8), constrained_layout=True)

    for i, f in enumerate(f_list):
        axs[i, 0].plot(x_array, f(x_array), color='black')
        # E_min, E_max = np.inf, -np.inf
        # for j, a in enumerate([-0.01, 0., 0.1, 0.5, 1.]):
        for j, a in enumerate([0., 0.1, 0.5, 1.]):
            U = advection1d(L=L, n=n, dt=dt, m=m, p=p, c=c, f=f, a=a, rktype='RK44', anim=False)
            E = compute_energy(L, p, n, m, U)
            axs[i, 1].plot(t_array, E, color='C'+str(j), label=r'$a={:.2f}$'.format(a))
            axs[i, 1].ticklabel_format(useOffset=False)
            # if j != 0:
            #     E_min, E_max = min(E_min, np.amin(E)), max(E_max, np.amax(E))

        # delta_E = E_max - E_min
        # axs[i, 1].set_ylim(E_min - delta_E / 10., E_max + 11. / 10. * delta_E)

    for ax in axs.flatten():
        ax.grid(ls=':')
    axs[-1, 0].set_xlabel(r"$x$", fontsize=ftSz2)
    for ax in axs[:, 0]:
        ax.set_ylabel(r"$u(x, t=0)$", fontsize=ftSz2)
    axs[-1, 1].set_xlabel(r"$t$", fontsize=ftSz2)
    for ax in axs[:, 1]:
        ax.set_ylabel("Energy", fontsize=ftSz2)
        ax.legend(fontsize=ftSz3)

    if save:
        fig.savefig("energy_evolution_10_new.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()


def plot_stable_time_step(save=False):
    df = pd.read_csv("stable_time_step.txt", delimiter=" ", names=["n", "p", "dt"], index_col=False)

    fig, ax = plt.subplots(1, 1, figsize=(10., 6.), constrained_layout=True)

    for p in df["p"].unique():
        sub_df = df[df['p'] == p]
        ax.loglog(1./sub_df["n"], sub_df["dt"], marker='o', label=r'$p={:d}$'.format(p))

        X = np.array(1./sub_df["n"]).reshape((-1, 1))
        Y = sub_df["dt"]
        model = LinearRegression(fit_intercept=False).fit(X, Y)
        intercept = model.intercept_
        order, = model.coef_
        # print("p = {:d} -> y = {:.3f} x  {:.3f}".format(p, np.exp(intercept), order))
        print("p = {:d} -> y = {:.3f} x + {:.3f}".format(p, order, intercept))

    ax.legend(fontsize=ftSz3)
    ax.grid(ls=':')
    ax.set_xlabel(r"$\Delta x \;\; [m]$", fontsize=ftSz2)
    ax.set_ylabel(r"$\Delta t \;\; [s]$", fontsize=ftSz2)

    if save:
        fig.savefig("stable_time_step.svg", format="svg", bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    save_global = False
    plt.rcParams["text.usetex"] = save_global

    # plot_energy(save=save_global)
    plot_stable_time_step(save=save_global)
