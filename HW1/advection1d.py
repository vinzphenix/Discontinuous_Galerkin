import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre
from matplotlib.animation import FuncAnimation


def fwd_euler(u, Q, dt, m):

    for i in range(m-1):
        u[i + 1] = u[i] + dt * Q.dot(u[i])

    return u


def rk22(u, Q, dt, m):

    for i in range(m-1):
        u[i + 1] = u[i] + dt * Q.dot(u[i] + dt / 2. * Q.dot(u[i]))

    return u


def rk44(u, Q, dt, m):

    for i in range(m-1):
        K1 = Q.dot(u[i])
        K2 = Q.dot(u[i] + K1 * dt / 2.)
        K3 = Q.dot(u[i] + K2 * dt / 2.)
        K4 = Q.dot(u[i] + K3 * dt)
        u[i + 1] = u[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6.

    return u


def build_matrix(n, p, a):
    diags_of_D = [2 * np.ones(p - 2 * i) for i in range((p + 1) // 2)]
    offsets_of_D = -np.arange(1, p + 1, 2)
    D = sp.diags([np.zeros(p + 1)] + diags_of_D, np.r_[0, offsets_of_D])  # array of 0 for case p = 0

    A = np.ones((p + 1, p + 1))
    A[1::2, ::2] = -1.
    A[::2, 1::2] = -1.
    A = sp.bsr_matrix(A)

    B = np.ones((p + 1, p + 1))
    B[:, 1::2] = -1.
    B = sp.bsr_matrix(B)

    I = np.ones((p + 1, p + 1))
    I = sp.bsr_matrix(I)

    mat_lft = -(1. + a) / 2. * B.T
    mat_rgt = +(1. - a) / 2. * B
    mat_ctr = D.T + (1. + a) / 2. * A - (1. - a) / 2. * I

    blocks = []
    for i in range(n):
        this_row = []
        for j in range(n):
            if (j == i - 1) or (i == 0 and j == n - 1):
                this_row.append(mat_lft)
            elif j == i:
                this_row.append(mat_ctr)
            elif (j == i + 1) or (i == n - 1 and j == 0):
                this_row.append(mat_rgt)
            else:
                this_row.append(None)
        blocks.append(this_row)

    L = sp.bmat(blocks, format='bsr')

    np.savetxt("matrix.txt", L.toarray(), fmt="%6.2f")
    return L


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
                u[k, i] += w[l] / 2. * f(xsi) * psi[i](s[l])
            u[k, i] *= (2 * i + 1)

    return u.reshape(n * (p + 1))


def advection1d(L, n, dt, m, p, c, f, a, rktype, anim=False):

    mass_matrix = sp.diags([np.tile(np.linspace(1, 2 * p + 1, p + 1), n)], [0], format='bsr')
    stiff_matrix = build_matrix(n, p, a)
    Q = -c * n / L * mass_matrix.dot(stiff_matrix)

    u = np.zeros((m, n * (p + 1)))
    u[0] = compute_coefficients(f, L=L, n=n, p=p)

    if rktype == 'ForwardEuler':
        fwd_euler(u, Q, dt, m)
    elif rktype == 'RK22':
        rk22(u, Q, dt, m)
    elif rktype == 'RK44':
        rk44(u, Q, dt, m)
    else:
        print("The integration method should be 'ForwardEuler', 'RK22', 'RK44'")
        raise ValueError

    u = u.T.reshape((n, p + 1, m))
    u = np.swapaxes(u, 0, 1)
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

    _ = FuncAnimation(fig, animate, m, interval=dt, blit=True, init_func=init, repeat_delay=3000)
    plt.show()


if __name__ == "__main__":

    f1 = lambda x: np.cos(2 * np.pi * x) + 0.4 * np.cos(4 * np.pi * x) + 0.1 * np.sin(6 * np.pi * x)
    f2 = lambda x: np.heaviside(np.fmod(np.fmod(x, 1.) + 1, 1.) - 0.5, 0.)
    f3 = lambda x: np.arctan(np.abs(np.tan(np.pi * x)))

    res = advection1d(L=1., n=20, dt=0.9 * 0.1454 * (1. / 20.), m=5000, p=3, c=1., f=f1, a=1., rktype='RK44', anim=True)
