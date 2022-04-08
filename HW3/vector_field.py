import numpy as np
import matplotlib.pyplot as plt


def plot_field():
    x = np.linspace(0, 100, 20)
    y = np.linspace(0, 100, 20)
    x, y = np.meshgrid(x, y)

    u = np.pi / 314 * (50 - y)
    v = np.pi / 314 * (x - 50)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    ax.quiver(x, y, u, v, np.hypot(u, v))
    plt.show()


def plot_field2():
    x = np.linspace(0, 1, 20)
    y = np.linspace(0, 1, 20)
    x, y = np.meshgrid(x, y)

    # u = -x ** 2 * y * (2 - x) ** 2 * (2 - y)
    # v = +x * y ** 2 * (2 - x) * (2 - y) ** 2
    u = -x ** 2 * y * (1 - x) ** 2 * (1 - y)
    v = +x * y ** 2 * (1 - x) * (1 - y) ** 2

    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)
    ax.quiver(x, y, u, v, np.hypot(u, v))
    plt.show()


if __name__ == "__main__":
    plot_field2()
