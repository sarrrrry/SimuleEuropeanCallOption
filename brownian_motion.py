import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set(style='darkgrid', context='paper')
seed = 0
np.random.seed(0)


def generate_brownian_motion(capital_t=1, n_split=100, x0: float=0.):
    var = capital_t / n_split
    noise = np.random.normal(0, var, n_split)
    noise[0] = x0
    brownian_motion = np.cumsum(noise)
    return brownian_motion


def q3_2(n_sample=100, capital_t=1, n_split=100):
    min_ = 999
    max_ = -999
    for _ in range(n_sample):
        brownian_motion = generate_brownian_motion(capital_t=capital_t, n_split=n_split)
        x_axis = np.linspace(0, capital_t, n_split)

        # plot
        bm = brownian_motion
        if min_ > bm.min(): min_ = bm.min()
        if max_ < bm.max(): max_ = bm.max()
        plt.plot(x_axis, brownian_motion)
    optional_caption = f"simulation times: {n_sample}\n(min={min_:.4f}, max={max_:.4f})"
    plt.title(r"Q.3(2): B.M. $B_{T}$ ($T=\{0=t_0,...,t_n=T\}$)" + optional_caption)
    plt.ylabel(r"$B_{t_i}$")
    plt.xlabel(r"$t_i$")

    plt.savefig(f'img3-2.png')
    plt.show()


def q3_1():
    capital_t = 1
    n_split = 100
    brownian_motion = generate_brownian_motion(capital_t=capital_t, n_split=n_split)
    x_axis = np.linspace(0, capital_t, n_split)

    # plot
    min_ = min(brownian_motion)
    max_ = max(brownian_motion)
    plt.plot(x_axis, brownian_motion)
    optional_caption = f"\n(min={min_:.4f}, max={max_:.4f})"
    plt.title(r"Q.3(1): B.M. $B_{T}$ ($T=\{0=t_0,...,t_n=T\}$)" + optional_caption)
    plt.ylabel(r"$B_{t_i}$")
    plt.xlabel(r"$t_i$")

    plt.savefig(f'img3-1.png')
    plt.show()


if __name__ == '__main__':
    q3_1()
    q3_2()