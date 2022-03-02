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


global P_right
global P_left

def local_dot(n, c, a, M_inv, D, u):

    u_left = np.dot(u, P_left)
    u_right = np.dot(u, P_right)

    F = np.zeros_like(u)

    for k in range(n):
        # Numerical flux at right interface
        n_out, n_in = -1, +1
        u_in = u_right[k]
        u_out = u_left[(k + 1) % n]
        u_avg = (u_in + u_out) / 2
        u_jump = n_in * u_in + n_out * u_out
        flux_r = c * u_avg + np.abs(c) * a * u_jump / 2

        # Numerical flux at left interface
        n_out, n_in = +1, -1
        u_in = u_left[k]
        u_out = u_right[(k - 1) % n]
        u_avg = (u_in + u_out) / 2
        u_jump = n_in * u_in + n_out * u_out
        flux_l = c * u_avg + np.abs(c) * a * u_jump / 2

        # Final computation
        F[k] = M_inv*(c * D @ u[k] - flux_r * P_right + flux_l * P_left)
    return F



def fwd_euler(u, dt, m, n, c, a, M_inv, D):
    for i in range(m - 1):
        u[i + 1] = u[i] + dt * local_dot(n, c, a, M_inv, D, u[i])

    return u


def rk22(u, dt, m, n, c, a, M_inv, D):
    for i in range(m - 1):
        u[i + 1] = u[i] + dt * local_dot(n, c, a, M_inv, D, u[i] + dt / 2. * local_dot(n, c, a, M_inv, D, u[i]))

    return u


def rk44(u, dt, m, n, c, a, M_inv, D):
    for i in range(m - 1):
        K1 = local_dot(n, c, a, M_inv, D, u[i])
        K2 = local_dot(n, c, a, M_inv, D,u[i] + K1 * dt / 2.)
        K3 = local_dot(n, c, a, M_inv, D,u[i] + K2 * dt / 2.)
        K4 = local_dot(n, c, a, M_inv, D,u[i] + K3 * dt)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.

    return u


def build_matrix(n, p, a):
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D])  # array of 0 for case p = 0

    global P_right
    global P_left
    P_right = np.ones(p + 1)
    P_left = np.array([(-1)**i for i in range(p+1)])

    return D


def compute_coefficients(f, L, n, p):
    w = [+0.5688888888888889, +0.4786286704993665, +0.4786286704993665, +0.2369268850561891, +0.2369268850561891]
    s = [+0.0000000000000000, -0.5384693101056831, +0.5384693101056831, -0.9061798459386640, +0.9061798459386640]
    dx = L / n
    psi = [legendre(i) for i in range(p + 1)]

    u = np.zeros((n, p + 1))
    for k in range(n):
        middle = dx * (k + 1. / 2.)
        for i in range(p + 1):
            for l in range(5):
                xsi = middle + s[l] * dx / 2.
                u[k, i] += w[l] * 1. / 2. * f(xsi) * psi[i](s[l])
            u[k, i] *= (2 * i + 1)

    return u


def advection1d(L, n, dt, m, p, c, f, a, rktype, anim=False):
    M_inv = n / L * np.linspace(1, 2 * p + 1, p + 1)
    D = build_matrix(n, p, a)

    u = np.zeros((m, n, (p + 1)))
    u[0] = compute_coefficients(f, L=L, n=n, p=p)

    if rktype == 'ForwardEuler':
        fwd_euler(u, dt, m, n, c, a, M_inv, D)
    elif rktype == 'RK22':
        rk22(u, dt, m, n, c, a, M_inv, D)
    elif rktype == 'RK44':
        rk44(u, dt, m, n, c, a, M_inv, D)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    u = np.swapaxes(u, 0, 2)
    if anim:
        plot_function(u, L=L, n=n, dt=dt, m=m, p=p, c=c, f=f)

    return u


def plot_function(u, L, n, dt, m, p, c, f):
    def init():
        exact.set_data(full_x, f(full_x))
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines):
            line.set_data(np.linspace(k * dx, (k + 1) * dx, n_plot), v[k, 0])

        return tuple([*lines, exact, time_text])

    def animate(t):
        exact.set_ydata(f(full_x - c * dt * t))
        time_text.set_text(time_template.format(dt * t))
        for k, line in enumerate(lines):
            line.set_ydata(v[k, t])

        return tuple([*lines, exact, time_text])

    n_plot = 100
    v = np.zeros((n, m, n_plot))
    r = np.linspace(-1, 1, n_plot)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    dx = L / n
    full_x = np.linspace(0, L, n * n_plot)

    for time in range(m):
        for elem in range(n):
            v[elem, time] = np.dot(psi, u[:, elem, time])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)
    ax.grid(ls=':')

    time_template = r'$t = {:.2f} [s]$'
    time_text = ax.text(0.85, 0.92, '', fontsize=17, transform=ax.transAxes)
    lines = [ax.plot([], [], color='C0')[0] for _ in range(n)]
    exact, = ax.plot(full_x, f(full_x), color='C1', alpha=0.5, lw=5, zorder=0)


    # to animate
    _ = FuncAnimation(fig, animate, m, interval=dt, blit=True, init_func=init, repeat_delay=3000)

    # to get only one frame at t = i
    # i = m //2 ; init() ; animate(i)

    plt.show()


if __name__ == "__main__":

    L_, n_, p_ = 1., 20, 3
    c_, m_ = 1., 2000
    dt_ = 0.5 * table[p_][3] / c_ * L_ / n_

    f1 = lambda x: np.sin(2 * np.pi * x / L_)
    f2 = lambda x: np.cos(2 * np.pi * x / L_) + 0.4 * np.cos(4 * np.pi * x / L_) + 0.1 * np.sin(6 * np.pi * x / L_)
    f3 = lambda x: np.arctan(np.abs(np.tan(np.pi * x / L_)))
    f4 = lambda x: np.heaviside(np.fmod(np.fmod(x / L_, 1.) + 1, 1.) - 0.5, 0.)
    f5 = lambda x: square(2 * np.pi * x / L_, 1/3)

    res = advection1d(L_, n_, dt_, m_, p_, c_, f=f5, a=1., rktype='RK44', anim=True)
