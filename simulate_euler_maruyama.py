from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from tqdm import tqdm

sns.set(context="paper", style="darkgrid")
np.random.seed(0)


def generate_brownian_motion(capital_t=1, n_split=100, x0: float = 0.):
    var = capital_t / n_split
    noise = np.random.normal(0, var, n_split)
    noise[0] = x0
    brownian_motion = np.cumsum(noise)
    return brownian_motion


class SDE:
    """ stochastic differential equation """

    def __init__(self, drift: float, volatility: float):
        self.drift = drift
        self.volatility = volatility

    def __call__(self, dt, X_t, dB_next):
        """
        Args:
            dt: t_{i+h} - t_i
            X_t:
            dB_next: dB_(t+h) - dB_(t)

        Returns:

        """
        dX_t = (self.drift * X_t * dt) + (self.volatility * X_t * dB_next)
        return dX_t


class EuropeanCallOption:
    def __init__(self, r, T, K):
        self.r = r
        self.T = T
        self.K = K

    def __call__(self, X_T):
        r = self.r
        T = self.T
        K = self.K
        f_X = np.exp(-r * T) * np.max((0, (X_T - K)))
        return f_X


class EulerMaruyamaSimulation:
    def __init__(self, sde: SDE, call_option: EuropeanCallOption, T: int, x0: float):
        self.sde = sde
        self.call_option = call_option
        self.T = T
        self.x0 = x0

    def simulate(self, dt: float, N_sim: int, N_split: int, ax: Optional[plt.Axes] = None):
        # N = 100
        T = self.T
        x0 = self.x0

        fX_t_list = []
        X_T_list = []
        for _ in tqdm(range(N_sim)):
            bm = generate_brownian_motion(capital_t=T, n_split=N_split, x0=x0)

            # Euler-Maruyama approximation
            X = [x0, ]
            for i in range(N_split - 1):
                dB_next = bm[i + 1] - bm[i]
                X_next = self.euler_maruyama_approx(dt, X[i], dB_next)
                X.append(X_next)

            if ax:
                t_ = np.linspace(0, T, N_split)
                ax.plot(t_, X)
            X_T = X[-1]
            f_X_t = self.call_option(X_T)

            fX_t_list.append(f_X_t)
            X_T_list.append(X_T)

        return fX_t_list, X_T_list

    def euler_maruyama_approx(self, dt, X_t, dB_next):
        dX_t = self.sde(dt, X_t, dB_next)
        X_next = X_t + dX_t
        return X_next


# -------------------------
# parameters
# -------------------------

# parameters
t0 = 0
T = 1

x0 = 100
# N_sim = 10
# N_sim = 100
N_sim = 100000
K = 100
r = 0.01
mu = 0.01
# mu = 0.1
sigma = 0.4

# -------------------------
# main process
# -------------------------

sde = SDE(drift=mu, volatility=sigma)
call_option = EuropeanCallOption(r=r, T=T, K=K)
simulator = EulerMaruyamaSimulation(
    sde=sde, call_option=call_option, T=T, x0=x0
)

N_split_list = []
mean_list = []
# for N_split in np.arange(100, 1001, 100):
# for N_split in [10, 50, 100, 300]:
for N_split in [100, ]:
    dt = float(T - t0) / N_split

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4), sharey='row',
        gridspec_kw=dict(
            width_ratios=[4, 1],
            wspace=0.1
        )
    )
    fX_t_list, X_T_list = simulator.simulate(
        dt=dt, N_sim=N_sim, N_split=N_split, ax=ax1
    )
    print(f"N_sim: {N_sim}")
    print(f"N_split: {N_split}")
    print(f"mean: {np.mean(fX_t_list)}")
    print(f"(std: {np.std(fX_t_list)})")
    N_split_list.append(N_split)
    mean_list.append(np.mean(fX_t_list))

    # -------------------------
    # plot
    # -------------------------

    eq = f"dX_t = {mu}X_t dt + {sigma}X_t dB_t"
    title = r"simulate: $" + eq + r"$" + f"\nN_sim: {N_sim}"
    ax1.set_title(title)
    ax1.set_xlabel(r"$t$")
    ax1.set_ylabel(r"$dX_t$")

    x_norm = np.linspace(np.min(X_T_list), np.max(X_T_list), 100)
    ax2.scatter(norm.pdf(x_norm, loc=np.mean(X_T_list), scale=np.std(X_T_list)), x_norm)
    # plt.savefig("data/em_sim.eps", dpi=300, bbox_inches="tight")
    plt.show()

    # plt.plot(fX_t_list)
    # plt.show()

    sns.kdeplot(fX_t_list)
    plt.axvline(np.mean(fX_t_list), color="red", ls="--", label="mean")
    ticks = [np.min(fX_t_list), np.mean(fX_t_list), np.max(fX_t_list)]
    plt.xticks(ticks, color="red")
    for fX in fX_t_list:
        plt.axvline(fX, ymax=0.02)
    plt.legend()
    # plt.savefig("data/call_option.eps", dpi=300, bbox_inches="tight")
    plt.show()

df = pd.DataFrame([N_split_list, mean_list], index=["N_split", "mean"]).T
# df.to_csv("./data/european_call_option_single.csv", index=False)
