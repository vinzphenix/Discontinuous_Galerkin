import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre
from matplotlib.animation import FuncAnimation

ftSz1, ftSz2, ftSz3 = 20, 15, 12
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'monospace'

table = [
    [1.0000, 1.0000, 1.2564, 1.3926, 1.6085],
    [0, 0.3333, 0.4096, 0.4642, 0.5348],
    [0, 0, 0.2098, 0.2352, 0.2716],
    [0, 0, 0.1301, 0.1454, 0.1679],
    [0, 0, 0.0897, 0.1000, 0.1155],
    [0, 0, 0.0661, 0.0736, 0.0851],
    [0, 0, 0.0510, 0.0568, 0.0656],
    [0, 0, 0.0407, 0.0453, 0.0523],
    [0, 0, 0.0334, 0.0371, 0.0428],
    [0, 0, 0.0279, 0.0310, 0.0358],
    [0, 0, 0.0237, 0.0264, 0.0304]
]

global P_right  # list of (1.)
global P_left  # list of (-1)^i
global P_right2  # matrix version of P_right
global P_left2  # matrix version of P_left
global coef  # first row is 1/Z, second row is Z


def local_dot(n, Q, a, M_inv, D, u, bctype):
    # this "u" has shape (2, (p + 1), 2 * n)

    # u_left, u_right are the values of the fields on the left and right of each element
    u_right = np.sum(u, axis=1)  # sum of the coefficients
    u_left = np.sum(u[:, ::2], axis=1) - np.sum(u[:, 1::2], axis=1)  # alternating sum

    Z_avg = coef[:, :-1] + coef[:, 1:]  # 1/Z for the first row and Z for the second
    u_avg = (coef * u_right)[:, :-1] + (coef * u_left)[:, 1:]  # average of the discontinuity
    u_jump = u_right[:, :-1] - u_left[:, 1:]  # difference of the discontinuity

    # Numerical flux at right interface
    flux_right = np.empty((2, 2 * n))
    flux_right[:, :-1] = 1 / Z_avg * (u_avg + a * u_jump[::-1])  # u_jump[::-1] switches the fields E and H

    # Numerical flux at left interface
    flux_left = np.empty((2, 2 * n))
    flux_left[:, 1:] = 1 / Z_avg * (u_avg + a * u_jump[::-1])

    # handle boundary cases
    if bctype == "periodic":
        Z_avg = coef[:, 0] + coef[:, -1]
        u_avg = (coef * u_left)[:, 0] + (coef * u_right)[:, -1]
        flux_right[:, -1] = 1 / Z_avg * u_avg  # no jump
        flux_left[:, 0] = 1 / Z_avg * u_avg
    elif bctype == "reflective":
        flux_right[:, -1] = [1, -1] * u_right[:, -1]
        flux_left[:, 0] = [1, -1] * u_left[:, 0]
    elif bctype == "non-reflective":
        flux_right[:, -1] = [1, -1] * u_right[:, -1]
        flux_left[:, 0] = [1, -1] * u_left[:, 0]
    else:
        print("AIE : Unknown Boundary condition")
        raise ValueError

    F = np.zeros_like(u)  # result of the dot product
    F[0] = 1 / Q[0] * (M_inv * (D.dot(u[1]) - flux_right[1] * P_right2 + flux_left[1] * P_left2))
    F[1] = 1 / Q[1] * (M_inv * (D.dot(u[0]) - flux_right[0] * P_right2 + flux_left[0] * P_left2))

    return F


def fwd_euler(u, dt, m, n, Q, a, M_inv, D, bctype):
    for i in range(m):
        u[i + 1] = u[i] + dt * local_dot(n, Q, a, M_inv, D, u[i], bctype)

    return u


def rk22(u, dt, m, n, Q, a, M_inv, D, bctype):
    for i in range(m):
        K1 = local_dot(n, Q, a, M_inv, D, u[i], bctype)
        K2 = local_dot(n, Q, a, M_inv, D, u[i] + dt / 2. * K1, bctype)
        u[i + 1] = u[i] + dt * K2

    return u


def rk44(u, dt, m, n, Q, a, M_inv, D, bctype):
    for i in range(m):
        K1 = local_dot(n, Q, a, M_inv, D, u[i], bctype)
        K2 = local_dot(n, Q, a, M_inv, D, u[i] + K1 * dt / 2., bctype)
        K3 = local_dot(n, Q, a, M_inv, D, u[i] + K2 * dt / 2., bctype)
        K4 = local_dot(n, Q, a, M_inv, D, u[i] + K3 * dt, bctype)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.

    return u


def build_matrix(n, p, eps, mu):
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D])  # array of 0 for case p = 0

    global P_right, P_right2
    global P_left, P_left2
    P_right = np.ones(p + 1)
    P_right2 = P_right.reshape(-1, 1)
    P_left = np.array([(-1) ** i for i in range(p + 1)])
    P_left2 = P_left.reshape(-1, 1)

    global coef
    coef = np.zeros((2, 2 * n))
    coef[0] = np.sqrt(eps / mu)
    coef[1] = np.sqrt(mu / eps)

    return D


def compute_coefficients(f0_list, L, n, p):
    w = [+0.5688888888888889, +0.4786286704993665, +0.4786286704993665, +0.2369268850561891, +0.2369268850561891]
    s = [+0.0000000000000000, -0.5384693101056831, +0.5384693101056831, -0.9061798459386640, +0.9061798459386640]
    dx = L / n
    psi = [legendre(i) for i in range(p + 1)]

    u = np.zeros((2 * n, p + 1, 2))
    for j, f0 in enumerate(f0_list):
        for k in range(2 * n):
            middle = dx * (k + 1. / 2.) - L
            for i in range(p + 1):
                for l in range(5):
                    xsi = middle + s[l] * dx / 2.
                    u[k, i, j] += w[l] * 1. / 2. * f0(xsi, L) * psi[i](s[l])
                u[k, i, j] *= (2 * i + 1)

    return np.swapaxes(u, 0, 2)


def maxwell1d(L, E0, H0, n, eps, mu, dt, m, p, rktype, bctype, a=1., anim=False):
    M_inv = (n / L * np.linspace(1, 2 * p + 1, p + 1)).reshape(-1, 1)
    D = build_matrix(n, p, eps, mu)
    Q = np.array([eps, mu])

    u = np.zeros((m + 1, 2, (p + 1), 2 * n))
    u[0] = compute_coefficients(f0_list=[E0, H0], L=L, n=n, p=p)

    if rktype == 'ForwardEuler':
        fwd_euler(u, dt, m, n, Q, a, M_inv, D, bctype)
    elif rktype == 'RK22':
        rk22(u, dt, m, n, Q, a, M_inv, D, bctype)
    elif rktype == 'RK44':
        rk44(u, dt, m, n, Q, a, M_inv, D, bctype)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    u = np.swapaxes(u, 0, 3)
    u = np.swapaxes(u, 0, 2)
    u = np.swapaxes(u, 0, 1)
    if anim:
        plot_function(u, L=L, n=n, dt=dt, m=m, p=p, f0_list=[E0, H0])

    return u


def plot_function(u, L, n, dt, m, p, f0_list):
    def init():
        exact_E.set_data(full_x, E0(full_x, L))
        exact_H.set_data(full_x, H0(full_x, L))
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines_E):
            line.set_data(np.linspace(k * dx - L, (k + 1) * dx - L, n_plot + 1), E[k, 0])
        for k, line in enumerate(lines_H):
            line.set_data(np.linspace(k * dx - L, (k + 1) * dx - L, n_plot + 1), H[k, 0])

        return tuple([*lines_H, *lines_E, exact_H, exact_E, time_text])

    def animate(t_idx):
        t = t_idx * dt
        # mu0, eps0 should be functions of x to simulate different media
        v_1 = 0.5 / sqrt_mu0 * E0(full_x - c * t, L) + 0.5 / sqrt_eps0 * H0(full_x - c * t, L)
        v_2 = 0.5 / sqrt_mu0 * E0(full_x + c * t, L) - 0.5 / sqrt_eps0 * H0(full_x + c * t, L)
        exact_E.set_ydata(sqrt_mu0 * (v_1 + v_2))
        exact_H.set_ydata(sqrt_eps0 * (v_1 - v_2))

        time_text.set_text(time_template.format(t))
        for k, line in enumerate(lines_H):
            line.set_ydata(H[k, t_idx])
        for k, line in enumerate(lines_E):
            line.set_ydata(E[k, t_idx])

        return tuple([*lines_H, *lines_E, exact_H, exact_E, time_text])

    E0, H0 = f0_list

    n_plot = 25
    E = np.zeros((2 * n, m + 1, n_plot + 1))
    H = np.zeros((2 * n, m + 1, n_plot + 1))
    r = np.linspace(-1, 1, n_plot + 1)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    c = 1 / (sqrt_eps0 * sqrt_mu0)
    dx = L / n
    # Z = coef[0, 1]
    full_x = np.linspace(-L, L, 2 * n * n_plot + 1).flatten()

    for time in range(m + 1):
        for elem in range(2 * n):
            E[elem, time] = np.dot(psi, u[0, :, elem, time])
            H[elem, time] = np.dot(psi, u[1, :, elem, time])

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex='all')

    time_template = r'$t = {:.2f} [s]$'
    time_text = axs[0].text(0.85, 0.90, '', fontsize=17, transform=axs[0].transAxes)
    lines_E = [axs[0].plot([], [], color='C0', marker='.', markevery=[0, -1])[0] for _ in range(2 * n)]
    lines_H = [axs[1].plot([], [], color='C0', marker='.', markevery=[0, -1])[0] for _ in range(2 * n)]
    exact_E, = axs[0].plot(full_x, E0(full_x, L * np.ones_like(full_x)), color='C1', alpha=0.5, lw=5, zorder=0)
    exact_H, = axs[1].plot(full_x, H0(full_x, L * np.ones_like(full_x)), color='C1', alpha=0.5, lw=5, zorder=0)

    scale = 1.15
    axs[0].set_xlim(-L, L)
    axs[0].set_ylim(-scale * coef[1, 1], scale * coef[1, 1])
    axs[1].set_ylim(-scale, scale)

    axs[0].set_ylabel(r"$E(x,t)$ [V/m]", fontsize=ftSz2)
    axs[1].set_ylabel(r"$H(x,t)$ [A/m]", fontsize=ftSz2)
    axs[1].set_xlabel(r"L [m]", fontsize=ftSz2)
    axs[0].grid(ls=':')
    axs[1].grid(ls=':')

    # to animate
    _ = FuncAnimation(fig, animate, m + 1, interval=dt, blit=True, init_func=init, repeat_delay=3000)

    # to get only one frame at t = i
    # i = 0 ; init() ; animate(i)

    plt.show()


if __name__ == "__main__":
    sqrt_eps0, sqrt_mu0 = np.sqrt(8.8541878128e-12), np.sqrt(4.e-7 * np.pi)

    L_, n_, p_ = 3e8 / 2., 10, 3
    c_, m_ = 1 / (sqrt_eps0 * sqrt_mu0), 2000
    eps0 = 8.85e-12 * np.ones(2 * n_)
    mu0 = 4 * np.pi * 1e-7 * np.ones(2 * n_)

    dt_ = 0.5 * table[p_][3] / c_ * L_ / n_

    E1 = lambda x, L: 0 * x
    E2 = lambda x, L: -sqrt_mu0 / sqrt_eps0 * np.exp(-(10 * x / L) ** 2)
    H1 = lambda x, L: np.exp(-(10 * x / L) ** 2)
    H2 = lambda x, L: (0.5 * np.cos(np.pi * x / L) + 0.5) * np.where(np.abs(x) <= L, 1., 0.)
    H3 = lambda x, L: np.cos(2 * np.pi * 10. * x / L) * np.exp(-(10 * x / L) ** 2)  # needs 20 elems
    H4 = lambda x, L: np.sin(2 * 5 * np.pi * x / L) * np.where(np.abs(x) <= L / 5., 1., 0.)
    H5 = lambda x, L: np.sin(2 * np.pi * 5 * x / L) * np.where(np.abs(x - L / 2.) <= L / 10., 1., 0.)  # non symmetric

    res = maxwell1d(L_, E1, H5, n_, eps0, mu0, dt_, m_, p_, 'RK44', bctype='periodic', a=1., anim=True)
    # res = maxwell1d(L_, E1, H5, n_, eps0, mu0, dt_, m_, p_, 'RK44', bctype='reflective', a=1., anim=True)
    # res = maxwell1d(L_, E1, H1, n_, eps0, mu0, dt_, m_, p_, 'RK44', bctype='non-reflective', a=1., anim=True)
