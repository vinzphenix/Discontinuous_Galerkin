import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.signal import square
from matplotlib.animation import FuncAnimation

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


# global P_right
# global P_left
# global P_right2
# global P_left2
# global coef

def local_dot(n, Q, a, M_inv, D, u):
    u_left = np.zeros((n,2))
    u_right = np.zeros((n,2))
    for j in range(2):
        u_left[:,j] = np.dot(u[:,:,j], P_left)
        u_right[:,j] = np.dot(u[:,:,j], P_right)

    F = np.zeros_like(u)
    for k in range(n):
        # Numerical flux at right interface
        if k != n-1:
            u_avg = coef[k] * u_right[k] + coef[k+1] * u_left[(k + 1)]
            u_jump = u_right[k] - u_left[(k + 1)]
            flux_r = 1 / (coef[k+1] + coef[k]) * (u_avg + a * u_jump[::-1])
        else:
            flux_r = np.zeros(2)

        # Numerical flux at left interface
        if k != 0:
            u_avg = coef[k] * u_left[k] + coef[k - 1] * u_right[k - 1]
            u_jump = -u_left[k] + u_right[k - 1]
            flux_l = 1 / (coef[k] + coef[k-1]) * (u_avg + a * u_jump[::-1])
        else:
            flux_l = np.zeros(2)
        F[k] = 1/Q[k] * (M_inv*(D @ u[k] - flux_r * P_right2 + flux_l * P_left2))[:,[1, 0]]
    return F



def fwd_euler(u, dt, m, n, Q, a, M_inv, D):
    for i in range(m - 1):
        u[i + 1] = u[i] + dt * local_dot(n, Q, a, M_inv, D, u[i])

    return u


def rk22(u, dt, m, n, Q, a, M_inv, D):
    for i in range(m - 1):
        u[i + 1] = u[i] + dt * local_dot(n, Q, a, M_inv, D, u[i] + dt / 2. * local_dot(n, Q, a, M_inv, D, u[i]))

    return u


def rk44(u, dt, m, n, Q, a, M_inv, D):
    for i in range(m - 1):
        K1 = local_dot(n, Q, a, M_inv, D, u[i])
        K2 = local_dot(n, Q, a, M_inv, D,u[i] + K1 * dt / 2.)
        K3 = local_dot(n, Q, a, M_inv, D,u[i] + K2 * dt / 2.)
        K4 = local_dot(n, Q, a, M_inv, D,u[i] + K3 * dt)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.

    return u


def build_matrix(n, p, a, eps, mu):
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D])  # array of 0 for case p = 0

    global P_right, P_right2
    global P_left, P_left2
    P_right = np.ones(p + 1)
    P_right2 = P_right.reshape(-1, 1)
    P_left = np.array([(-1)**i for i in range(p+1)])
    P_left2 = P_left.reshape(-1, 1)


    global coef
    coef = np.zeros((n,2))
    coef[:,0] = np.sqrt(eps/mu)
    coef[:,1] = np.sqrt(mu/eps)

    return D


def compute_coefficients(f, L, n, p):
    w = [+0.5688888888888889, +0.4786286704993665, +0.4786286704993665, +0.2369268850561891, +0.2369268850561891]
    s = [+0.0000000000000000, -0.5384693101056831, +0.5384693101056831, -0.9061798459386640, +0.9061798459386640]
    dx = L / n
    psi = [legendre(i) for i in range(p + 1)]

    u = np.zeros((n, p + 1, 2))
    for j in range(2):
        for k in range(n):
            middle = dx * (k + 1. / 2.) - L / 2
            for i in range(p + 1):
                for l in range(5):
                    xsi = middle + s[l] * dx / 2.
                    u[k, i, j] += w[l] * 1. / 2. * f[j](xsi) * psi[i](s[l])
                u[k, i, j] *= (2 * i + 1)

    return u


def maxwell1d(L, E0, H0, n, eps, mu, dt, m, p, a, rktype, anim=False):
    M_inv = (n / L * np.linspace(1, 2 * p + 1, p + 1)).reshape(-1, 1)
    D = build_matrix(n, p, a, eps, mu)
    Q = np.array([eps, mu]).T

    u = np.zeros((m, n, (p + 1), 2))
    u[0] = compute_coefficients(f=[E0, H0], L=L, n=n, p=p)

    if rktype == 'ForwardEuler':
        fwd_euler(u, dt, m, n, Q, a, M_inv, D)
    elif rktype == 'RK22':
        rk22(u, dt, m, n, Q, a, M_inv, D)
    elif rktype == 'RK44':
        rk44(u, dt, m, n, Q, a, M_inv, D)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    u = np.swapaxes(u, 0, 3)
    u = np.swapaxes(u, 1, 2)
    if anim:
        plot_function(u, L=L, n=n, dt=dt, m=m, p=p, f=[E0, H0])

    return u


def plot_function(u, L, n, dt, m, p, f):
    def init():
        #exact.set_data(full_x, f(full_x))
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines_H):
            line.set_data(np.linspace(k * dx - L / 2, (k + 1) * dx - L / 2, n_plot), H[k, 0])
        for k, line in enumerate(lines_E):
            line.set_data(np.linspace(k * dx - L / 2, (k + 1) * dx - L / 2, n_plot), E[k, 0])

        return tuple([*lines_H, *lines_E, exact, time_text])

    def animate(t):
        #exact.set_ydata(f(full_x - c * dt * t))
        time_text.set_text(time_template.format(dt * t))
        for k, line in enumerate(lines_H):
            line.set_ydata(H[k, t])
        for k, line in enumerate(lines_E):
            line.set_ydata(E[k, t])

        return tuple([*lines_H, *lines_E, exact, time_text])

    n_plot = 100
    E = np.zeros((n, m, n_plot))
    H = np.zeros((n, m, n_plot))
    r = np.linspace(-1, 1, n_plot)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    dx = L / n
    full_x = np.linspace(-L/2, L/2, n * n_plot)

    for time in range(m):
        for elem in range(n):
            H[elem, time] = np.dot(psi, u[1, :, elem, time])
            E[elem, time] = np.dot(psi, u[0, :, elem, time])


    # #test plot
    # fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True, sharex='all', sharey='all')
    # for elem in range(n):
    #     middle = dx * (elem + 1. / 2.) - L / 2
    #     ax.plot(middle + r * dx / 2, np.dot(psi, u[1,:, elem, 1]), color='C0')
    # plt.show()


    fig, ax = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True, sharex='all')
    ax[0].grid(ls=':')
    ax[1].grid(ls=':')


    time_template = r'$t = {:.2f} [s]$'
    time_text = ax[0].text(0.85, 0.92, '', fontsize=17, transform=ax[0].transAxes)
    lines_H = [ax[0].plot([], [], color='C0')[0] for _ in range(n)]
    lines_E = [ax[1].plot([], [], color='C0')[0] for _ in range(n)]
    exact, = ax[0].plot(full_x, f[1](full_x), color='C1', alpha=0.5, lw=5, zorder=0)

    ax[0].set_ylim(-1, 1)
    ax[0].set_xlim(-L/2, L/2)
    ax[1].set_ylim(-coef[0,1], coef[0,1])
    # to animate
    _ = FuncAnimation(fig, animate, m, interval=dt, blit=True, init_func=init, repeat_delay=3000)

    # to get only one frame at t = i
    # i = m //2 ; init() ; animate(i)

    plt.show()


if __name__ == "__main__":

    L_, n_, p_ = 3e8, 20, 3
    c_, m_ = 3e8, 2000
    eps0 = 8.85e-12 * np.ones(n_)
    mu0 = 4*np.pi *1e-7 *np.ones(n_)
    dt_ = 0.5 * table[p_][3] / c_ * L_ / n_

    E1 = lambda x: 0
    H1 = lambda x: np.exp(-(10*x/L_)**2)


    res = maxwell1d(L_, E1, H1, n_, eps0, mu0, dt_, m_, p_, a=1., rktype='RK44', anim=True)
