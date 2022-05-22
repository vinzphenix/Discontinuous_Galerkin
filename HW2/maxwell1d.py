import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre, roots_legendre
from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm


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
global coef  # first row is sqrt(eps/mu), second row is sqrt(mu/eps)
global M_inv  # inverse of mass matrix
global Stiff  # stiff matrix


def local_dot(n, Q, a, u, bctype):
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
        u_jump = u_right[:, -1] - u_left[:, 0]
        flux_right[:, -1] = 1 / Z_avg * (u_avg + a * u_jump[::-1])
        flux_left[:, 0] = 1 / Z_avg * (u_avg + a * u_jump[::-1])
    elif bctype == "reflective":
        flux_right[:, -1] = [1, -1] * u_right[:, -1]
        flux_left[:, 0] = [1, -1] * u_left[:, 0]
    elif bctype == "non-reflective":
        flux_right[:, -1] = 0.5 * np.array([[1, coef[1, -1]], [coef[0, -1], 1]]) @ u_right[:, -1]
        flux_left[:, 0] = 0.5 * np.array([[1, -coef[1, 0]], [-coef[0, 0], 1]]) @ u_left[:, 0]
    else:
        print("AIE : Unknown Boundary condition")
        raise ValueError

    F = np.empty_like(u)  # result of the dot product
    F[0] = 1 / Q[0] * M_inv * (Stiff.dot(u[1]) - flux_right[1] * P_right + flux_left[1] * P_left)
    F[1] = 1 / Q[1] * M_inv * (Stiff.dot(u[0]) - flux_right[0] * P_right + flux_left[0] * P_left)

    return F


def fwd_euler(u, dt, m, n, Q, a, bctype):
    for i in range(m):
        u[i + 1] = u[i] + dt * local_dot(n, Q, a, u[i], bctype)

    return u


def rk22(u, dt, m, n, Q, a, bctype):
    for i in range(m):
        K1 = local_dot(n, Q, a, u[i], bctype)
        K2 = local_dot(n, Q, a, u[i] + dt / 2. * K1, bctype)
        u[i + 1] = u[i] + dt * K2

    return u


def rk44(u, dt, m, n, Q, a, bctype):
    for i in range(m):
        K1 = local_dot(n, Q, a, u[i], bctype)
        K2 = local_dot(n, Q, a, u[i] + K1 * dt / 2., bctype)
        K3 = local_dot(n, Q, a, u[i] + K2 * dt / 2., bctype)
        K4 = local_dot(n, Q, a, u[i] + K3 * dt, bctype)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.

    return u


def build_matrix(n, p, eps, mu, L):
    global M_inv
    M_inv = (n / L * np.linspace(1, 2 * p + 1, p + 1)).reshape(-1, 1)

    global Stiff
    diags_of_S = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_S = -np.arange(1, p + 1, 2)
    Stiff = sp.diags([np.zeros(p + 1)] + diags_of_S, np.r_[0, offsets_of_S])  # array of 0 for case p = 0

    global P_right
    global P_left
    P_right = (np.ones(p + 1)).reshape(-1, 1)
    P_left = np.array([(-1) ** i for i in range(p + 1)]).reshape(-1, 1)

    global coef
    coef = np.zeros((2, 2 * n))
    coef[0] = np.sqrt(eps / mu)
    coef[1] = np.sqrt(mu / eps)


def compute_coefficients(f0_list, L, n, p):
    quad_order = 2 * (p - 1)  # source to be added
    s, w = roots_legendre(quad_order)
    dx = L / n

    u = np.zeros((2, p + 1, 2 * n))
    m_inv = (2. * np.arange(p + 1) + 1.) / 2.

    # vectorized-form
    middle = np.tile(dx * np.arange(2 * n) + dx / 2. - L, (quad_order, 1))
    x_loc = middle + np.tile(s * dx / 2, (2 * n, 1)).T
    f_evals = [f(x_loc, L) for f in f0_list]  # 2 matrices of size (quad_order, 2n)
    psi = np.polynomial.legendre.legvander(s, p).T  # Vandermonde matrix for legendre polynomials
    u[0] = m_inv[:, np.newaxis] * np.dot(w[np.newaxis, :] * psi, f_evals[0])  # sum over the quadrature points
    u[1] = m_inv[:, np.newaxis] * np.dot(w[np.newaxis, :] * psi, f_evals[1])

    # readable-form
    # psi = [legendre(i) for i in range(p + 1)]
    # for k in range(2 * n):  # loop over elements
    #     middle = dx * (k + 1. / 2.) - L
    #     for l in range(quad_order):
    #         x_loc = middle + s[l] * dx / 2.
    #         for i in range(p + 1):
    #             u[0, i, k] += w[l] * f0_list[0](x_loc, L) * psi[i](s[l])
    #             u[1, i, k] += w[l] * f0_list[1](x_loc, L) * psi[i](s[l])
    #     u[0, :, k] *= m_inv
    #     u[1, :, k] *= m_inv

    return u


def maxwell1d(L, E0, H0, n, eps, mu, dt, m, p, rktype, bctype, a=1., anim=False, save=False):

    build_matrix(n, p, eps, mu, L)
    Q = np.array([eps, mu])

    u = np.zeros((m + 1, 2, (p + 1), 2 * n))
    u[0] = compute_coefficients(f0_list=[E0, H0], L=L, n=n, p=p)

    if rktype == 'ForwardEuler':
        fwd_euler(u, dt, m, n, Q, a, bctype)
    elif rktype == 'RK22':
        rk22(u, dt, m, n, Q, a, bctype)
    elif rktype == 'RK44':
        rk44(u, dt, m, n, Q, a, bctype)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    # set the time axis first: (3 permutations needed to roll the axes)
    u = np.swapaxes(u, 0, 3)
    u = np.swapaxes(u, 0, 2)
    u = np.swapaxes(u, 0, 1)
    if anim:
        plot_function(u, L=L, n=n, eps=eps, mu=mu, dt=dt, m=m, p=p, f0_list=[E0, H0], save=save, bctype=bctype)

    return u


def plot_function(u, L, n, eps, mu, dt, m, p, f0_list, save=False, bctype="Reflective"):
    def init():
        animated_items = []
        if alpha > 0.:
            exact_E.set_data(full_x_no_dim, E0(full_x, L))
            exact_H.set_data(full_x_no_dim, H0(full_x, L))
            animated_items += [exact_E, exact_H]
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines_E):
            line.set_data(np.linspace(k * dx - L, (k + 1) * dx - L, n_plot + 1) / c_, E[k, 0])
        for k, line in enumerate(lines_H):
            line.set_data(np.linspace(k * dx - L, (k + 1) * dx - L, n_plot + 1) / c_, H[k, 0])
        return animated_items + [*lines_H, *lines_E, time_text]

    def animate(t_idx):
        t_idx *= ratio
        t = t_idx * dt
        animated_items = []
        # mu0, eps0 should be functions of x to simulate different media
        if alpha > 0.:
            v_1 = 0.5 / full_sqrt_mu * E0(full_x - c * t, L) + 0.5 / full_sqrt_eps * H0(full_x - c * t, L)
            v_2 = 0.5 / full_sqrt_mu * E0(full_x + c * t, L) - 0.5 / full_sqrt_eps * H0(full_x + c * t, L)
            exact_E.set_ydata(full_sqrt_mu * (v_1 + v_2))
            exact_H.set_ydata(full_sqrt_eps * (v_1 - v_2))
            animated_items.append += [exact_E, exact_H]

        time_text.set_text(time_template.format(t))
        for k, line in enumerate(lines_H):
            line.set_ydata(H[k, t_idx])
        for k, line in enumerate(lines_E):
            line.set_ydata(E[k, t_idx])

        pbar.update(1)
        return animated_items + [*lines_H, *lines_E, time_text]

    E0, H0 = f0_list

    n_plot = 25
    E = np.zeros((2 * n, m + 1, n_plot + 1))
    H = np.zeros((2 * n, m + 1, n_plot + 1))
    r = np.linspace(-1, 1, n_plot + 1)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    dx = L / n
    full_x = np.linspace(-L, L, 2 * n * n_plot + 1).flatten()
    full_sqrt_eps = np.sqrt(np.r_[eps[0], np.repeat(eps, n_plot)])
    full_sqrt_mu = np.sqrt(np.r_[mu[0], np.repeat(mu, n_plot)])
    c = 1. / (np.sqrt(eps0) * np.sqrt(mu0))

    L_no_dim = L / c_
    full_x_no_dim = full_x / c_

    alpha = 0.5 if (bctype == "non-reflective") and (np.all(np.isclose(eps, eps[0], atol=1e-2 * eps0))) and (
        np.all(np.isclose(mu, mu[0], atol=1e-2 * mu0))) else 0.

    for time in range(m + 1):
        for elem in range(2 * n):
            E[elem, time] = np.dot(psi, u[0, :, elem, time])
            H[elem, time] = np.dot(psi, u[1, :, elem, time])

    fig, axs = plt.subplots(2, 1, figsize=(8., 5.), sharex='all')
    fig.tight_layout()

    time_template = r'$t = \mathtt{{{:.2f}}} \; [s]$'
    time_text = axs[0].text(0.815, 0.85, '', fontsize=ftSz1, transform=axs[0].transAxes)
    lines_E = [axs[0].plot([], [], color='C1', marker='.', markevery=[0, -1])[0] for _ in range(2 * n)]
    lines_H = [axs[1].plot([], [], color='C0', marker='.', markevery=[0, -1])[0] for _ in range(2 * n)]
    if alpha > 0.:
        exact_E, = axs[0].plot(full_x_no_dim, E0(full_x, L * np.ones_like(full_x)), color='C1', alpha=alpha, lw=5, zorder=0)
        exact_H, = axs[1].plot(full_x_no_dim, H0(full_x, L * np.ones_like(full_x)), color='C0', alpha=alpha, lw=5, zorder=0)

    if (eps[0] != eps[n]) or (mu[0] != mu[n]):
        axs[0].add_patch(plt.Rectangle((-L_no_dim / 2., -2. * coef[1, 1]),
                                       L_no_dim, 4. * coef[1, 1], facecolor="grey", alpha=0.35))
        axs[1].add_patch(plt.Rectangle((-L_no_dim / 2., -2.), L_no_dim, 4., facecolor="grey", alpha=0.35))

    scale = 1.15
    axs[0].set_xlim(-L_no_dim, L_no_dim)
    axs[0].set_ylim(-scale * coef[1, 1], scale * coef[1, 1])
    axs[1].set_ylim(-scale, scale)

    axs[0].set_ylabel(r"$E(x,t)$ [V/m]", fontsize=ftSz2)
    axs[1].set_ylabel(r"$H(x,t)$ [A/m]", fontsize=ftSz2)
    axs[1].set_xlabel(r"$L$ [light-second]", fontsize=ftSz2)
    axs[0].grid(ls=':')
    axs[1].grid(ls=':')
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.11, top=0.995)

    # to animate
    fps = 30
    t_anim = 6.
    ratio = max(m // int(fps * t_anim), 1)
    # ratio = 1
    nFrames = m//ratio + 1

    init()
    anim = FuncAnimation(fig, animate, nFrames, interval=50, blit=False, init_func=lambda: None, repeat_delay=3000)
    pbar = tqdm(total=nFrames)

    if save:
        writerMP4 = FFMpegWriter(fps=fps)
        anim.save(f"./figures/anim.mp4", writer=writerMP4)
    else:
        # to get only one frame at t = i
        # i = 0 ; init() ; animate(i)
        plt.show()

    pbar.close()
    return


ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'

if __name__ == "__main__":
    eps0, mu0 = 8.8541878128e-12, 4.e-7 * np.pi

    L_, n_, p_ = 3e8 / 2., 15, 3
    c_ = 1. / (np.sqrt(eps0) * np.sqrt(mu0))
    dt_ = 0.5 * table[p_][3] / c_ * L_ / n_
    m_ = int(np.ceil(1. / dt_))

    eps_, mu_ = eps0 * np.ones(2 * n_), mu0 * np.ones(2 * n_)
    # eps_[n_ // 2:3 * n_ // 2] *= 20  # permittivity of glass differs, but the rest stays the same
    # mu_[n_ // 2:3 * n_ // 2] *= 5  # hypothetical material

    E1 = lambda x, L: 0 * x
    E2 = lambda x, L: -np.sqrt(mu0) / np.sqrt(eps0) * np.exp(-(10 * x / L) ** 2)
    H1 = lambda x, L: np.exp(-(10 * x / L) ** 2)
    H2 = lambda x, L: (0.5 * np.cos(np.pi * x / L) + 0.5) * np.where(np.abs(x) <= L, 1., 0.)
    H3 = lambda x, L: np.cos(2 * np.pi * 10. * x / L) * np.exp(-(10 * x / L) ** 2)  # needs more precision
    H4 = lambda x, L: np.sin(2 * 5 * np.pi * x / L) * np.where(np.abs(x) <= L / 5., 1., 0.)
    H5 = lambda x, L: np.sin(2 * np.pi * 5 * x / L) * np.where(np.abs(x - L / 2.) <= L / 10., 1., 0.)  # non symmetric

    # res = maxwell1d(L_, E1, H1, n_, eps_, mu_, dt_, m_, p_, 'RK44', bctype='periodic', a=1., anim=True)
    res = maxwell1d(L_, E1, H4, n_, eps_, mu_, dt_, m_, p_, 'RK44', bctype='reflective', a=1., anim=True, save=False)
    # res = maxwell1d(L_, E1, H1, n_, eps_, mu_, dt_, m_, p_, 'RK44', bctype='non-reflective', a=1., anim=True, save=False)
