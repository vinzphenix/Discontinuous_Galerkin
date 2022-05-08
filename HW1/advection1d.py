import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.special import legendre
from scipy.signal import square
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


def fwd_euler(u, Q, dt, m):
    for i in range(m):
        u[i + 1] = u[i] + dt * Q.dot(u[i])

    return u


def rk22(u, Q, dt, m):
    for i in range(m):
        u[i + 1] = u[i] + dt * Q.dot(u[i] + dt / 2. * Q.dot(u[i]))

    return u


def rk44(u, Q, dt, m):
    for i in range(m):
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
                u[k, i] += w[l] * 1. / 2. * f(xsi, L) * psi[i](s[l])
            u[k, i] *= (2 * i + 1)

    return u.reshape(n * (p + 1))


def advection1d(L, n, dt, m, p, c, f, a, rktype, anim=False, save=False, tend=0.):
    inv_mass_matrix = sp.diags([np.tile(np.linspace(1, 2 * p + 1, p + 1), n)], [0], format='bsr')
    stiff_matrix = build_matrix(n, p, a)
    Q = -c * n / L * inv_mass_matrix.dot(stiff_matrix)

    u = np.zeros((m + 1, n * (p + 1)))
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

    u = u.T.reshape((n, p + 1, m + 1))
    u = np.swapaxes(u, 0, 1)
    if anim:
        plot_function(u, L=L, n=n, dt=dt, m=m, p=p, c=c, f=f, save=save, tend=tend)

    return u


def plot_function(u, L, n, dt, m, p, c, f, save=False, tend=0.):
    def init():
        exact.set_data(full_x, f(full_x, L))
        time_text.set_text(time_template.format(0))
        for k, line in enumerate(lines):
            line.set_data(np.linspace(k * dx, (k + 1) * dx, n_plot + 1), v[k, 0])

        return tuple([*lines, exact, time_text])

    def animate(t):
        t *= ratio
        exact.set_ydata(f(full_x - c * dt * t, L))
        time_text.set_text(time_template.format(dt * t))
        for k, line in enumerate(lines):
            line.set_ydata(v[k, t])
        pbar.update(1)
        return tuple([*lines, exact, time_text])

    n_plot = 100
    v = np.zeros((n, m + 1, n_plot + 1))
    r = np.linspace(-1, 1, n_plot + 1)
    psi = np.array([legendre(i)(r) for i in range(p + 1)]).T
    dx = L / n
    full_x = np.linspace(0., L, n * n_plot + 1)

    for time in range(m + 1):
        for elem in range(n):
            v[elem, time] = np.dot(psi, u[:, elem, time])

    fig, ax = plt.subplots(1, 1, figsize=(8., 4.5))
    fig.tight_layout()
    ax.grid(ls=':')

    time_template = r'$t = \mathtt{{{:.2f}}} \;[s]$'
    time_text = ax.text(0.815, 0.92, '', fontsize=ftSz1, transform=ax.transAxes)

    lines = [ax.plot([], [], color='C0')[0] for _ in range(n)]
    exact, = ax.plot(full_x, f(full_x, L), color='C1', alpha=0.5, lw=5, zorder=0)
    ax.set_ylim(1.25 * np.array(ax.get_ylim()))
    ax.set_xlabel(r"$x$", fontsize=ftSz2)
    ax.set_ylabel(r"$u$", fontsize=ftSz2)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.11, top=0.995)

    # to animate
    fps = m / tend
    # ratio = max(m // int(fps * tend), 1)
    ratio = 1
    pbar = tqdm(total=m // ratio + 1)
    init()
    anim = FuncAnimation(fig, animate, m // ratio + 1, interval=50, blit=False,
                         init_func=lambda: None, repeat=False, repeat_delay=3000)

    if save:
        writerMP4 = FFMpegWriter(fps=fps)
        anim.save(f"./Figures/anim.mp4", writer=writerMP4)
    else:
        # to get only one frame at t = i
        # i = m ; init() ; animate(i)
        plt.show()

    pbar.close()
    return


ftSz1, ftSz2, ftSz3 = 20, 17, 14
plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'serif'


if __name__ == "__main__":

    L_, n_, p_ = 1., 4, 3
    c_ = 0.4
    dt_ = 0.5 * table[p_][3] / c_ * L_ / n_
    # dt_ = 0.002
    T_end = 5.
    m_ = int(T_end / dt_)

    f1 = lambda x, L: np.sin(2 * 3 * np.pi * x / L)
    f2 = lambda x, L: np.cos(2 * np.pi * x / L) + 0.4 * np.cos(4 * np.pi * x / L) + 0.1 * np.sin(6 * np.pi * x / L)
    f3 = lambda x, L: np.arctan(np.abs(np.tan(np.pi * x / L)))
    f4 = lambda x, L: np.heaviside(np.fmod(np.fmod(x / L, 1.) + 1, 1.) - 0.5, 0.)
    f5 = lambda x, L: square(2 * np.pi * x / L, 1/3)

    res = advection1d(L_, n_, dt_, m_, p_, c_, f=f2, a=1., rktype='RK44', anim=True, save=False, tend=T_end)
