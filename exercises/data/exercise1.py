# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: neuralNetworks (3.10.9)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### Exercise 1 - Point Clouds: Geometry and Spread in 2D

# %% [markdown]
# A - Generate the Clouds

# %%
# --8<-- [start:itemA]
import numpy as np
import matplotlib.pyplot as plt

params = {
    0: {"mean": (2, 3), "std": (0.8, 2.5)},
    1: {"mean": (5, 6), "std": (1.2, 1.9)},
    2: {"mean": (8, 1), "std": (0.9, 0.9)},
    3: {"mean": (15, 4), "std": (0.5, 2.0)},
}

n_per_class = 100
rng = np.random.default_rng(seed=42)

X_list = []
y_list = []

for label, p in params.items():
    mean = p["mean"]
    std = p["std"]
    points = rng.normal(loc=mean, scale=std, size=(n_per_class, 2))
    X_list.append(points)
    y_list.append(np.full(n_per_class, label))

X = np.vstack(X_list)   
y = np.concatenate(y_list)  

colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
plt.figure(figsize=(7, 6))
 
for label, p in params.items():
    mask = y == label
    plt.scatter(X[mask, 0], X[mask, 1], s=25, alpha=0.7,
                color=colors[label], label=f"Class {label}")
    plt.scatter(*p["mean"], color=colors[label], edgecolor="black",
                marker="X", s=150, linewidths=1.5, zorder=5)
 
plt.title("Figure 1: Synthetic 2D Gaussian Dataset (4 classes)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/figure1.png", dpi=150)
# --8<-- [end:itemA]

# %% [markdown]
# B - More or Less Spread Out

# %%
# ---8<-- [start:itemB-dataset]
# ==== Generate 4 datasets, one per scale ====

from itertools import combinations

params = {
    0: {"mean": (2, 3), "std": (0.8, 2.5)},
    1: {"mean": (5, 6), "std": (1.2, 1.9)},
    2: {"mean": (8, 1), "std": (0.9, 0.9)},
    3: {"mean": (15, 4), "std": (0.5, 2.0)},
}

n_per_class = 100
scales = [0.5, 1.0, 2.0, 4.0]
rng = np.random.default_rng(seed=42)

datasets = {}
for s in scales:
    X_list = []
    y_list = []
    for label, p in params.items():
        mean = p["mean"]
        std = tuple(s * np.array(p["std"]))
        points = rng.normal(loc=mean, scale=std, size=(n_per_class, 2))
        X_list.append(points)
        y_list.append(np.full(n_per_class, label))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    datasets[s] = (X, y)


# ==== Shared axis limits ====

X_max_s, _ = datasets[max(scales)]
pad = 2
xlim = (X_max_s[:, 0].min() - pad, X_max_s[:, 0].max() + pad)
ylim = (X_max_s[:, 1].min() - pad, X_max_s[:, 1].max() + pad)

# ==== Figure 2: 4 subplots, one per scale ====

fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
for ax, s in zip(axes.flat, scales):
    X, y = datasets[s]
    for label, p in params.items():
        mask = y == label
        ax.scatter(X[mask, 0], X[mask, 1], s=18, alpha=0.7,
                   color=colors[label], label=f"Class {label}")
        ax.scatter(*p["mean"], color=colors[label], edgecolor="black",
                   marker="X", s=150, linewidths=1.3, zorder=5)
    ax.set_title(f"scale factor s = {s}")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(alpha=0.3)
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Figure 2: Same 4 classes at increasing std scale factors", fontsize=13)
fig.tight_layout()
fig.savefig("./figures/figure2.png", dpi=150)
# ---8<-- [end:itemB-dataset]


# %%
# ---8<-- [start:itemB-separation]
# ==== Separation ratio at s=1 ====

def eff_std(p):
    return (p["std"][0] + p["std"][1]) / 2

s_ref = 1
pairs = list(combinations(range(4), 2))
rows = []
for i, j in pairs:
    mean_i = np.array(params[i]["mean"])
    mean_j = np.array(params[j]["mean"])
    dist = np.linalg.norm(mean_i - mean_j)
    std_sum = (eff_std(params[i]) + eff_std(params[j])) * s_ref
    ratio = dist / std_sum
    rows.append((i, j, dist, std_sum, ratio))
 
print("Separation ratio table:")
print(f"{'pair':>7} | {'dist':>7} | {'std_i+std_j':>12} | {'ratio':>7}")
for i, j, dist, std_sum, ratio in rows:
    print(f"({i},{j}):  | {dist:7.3f} | {std_sum:12.3f} | {ratio:7.3f}")
 
min_row = min(rows, key=lambda r: r[4])
print(f"\nSmallest ratio: pair ({min_row[0]},{min_row[1]}) = {min_row[4]:.3f} at s=1")
s_target = 2
extrapolated = min_row[4] * s_ref / s_target
print(f"At s={s_target}, that smallest ratio becomes: {min_row[4]:.3f} / {s_target} = {extrapolated:.3f}")
# ---8<-- [end:itemB-separation]

# %% [markdown]
# Separation Ratio Table
#
# | pair | dist | std_i+std_j | ratio |
# |------|------|-----------|-------|
# | (0,1):  |  4.243  |   3.200  |  1.326 |
# | (0,2):  |  6.325  |   2.550  |  2.480 |
# | (0,3):  |  13.038 |   2.900  |  4.496 |
# | (1,2):  |  5.831  |   2.450  |  2.379 |
# | (1,3):  |  10.198 |   2.800  |  3.642 |
# | (2,3):  |  7.616  |   2.150  |  3.542 |
#
# Smallest ratio: pair (0,1) = 1.326 at s=1
#
# At s=2, that smallest ratio becomes: 1.326 / 2 = 0.663
#

# %%
# ---8<-- [start:itemB-mixing]
# ===== Mixing rate calculation =====

class_labels = np.array(sorted(params.keys()))

class_means = np.array([
    params[label]["mean"]
    for label in class_labels
])

mixing_rates = []

for s in scales:
    X, y = datasets[s]
    distances = np.linalg.norm(
        X[:, None, :] - class_means[None, :, :],
        axis=2
    )

    nearest_mean_indices = np.argmin(distances, axis=1)
    nearest_labels = class_labels[nearest_mean_indices]

    mixing_rate = np.mean(nearest_labels != y)
    mixing_rates.append(mixing_rate)

    print(
        f"Mixing rate at s = {s:.1f}: "
        f"{mixing_rate:.3f} ({mixing_rate * 100:.1f}%)"
    )
# ---8<-- [end:itemB-mixing]

# %% [markdown]
# Mixing rate at s=0.5: 0.003 (0.3%) <br>
# Mixing rate at s=1.0: 0.072 (7.2%) <br>
# Mixing rate at s=2.0: 0.193 (19.3%) <br>
# Mixing rate at s=4.0: 0.482 (48.2%)

# %%
# ---8<-- [start:itemB-plot]
# ===== Figure 3: mixing rate vs scale factor =====

plt.figure(figsize=(8, 7))
plt.plot(scales, mixing_rates, marker="o", linewidth=2, color="#8e44ad")
for s, mr in zip(scales, mixing_rates):
    plt.annotate(f"{mr:.2f}", (s, mr), textcoords="offset points", xytext=(0, 8), ha="center")
plt.xlabel("scale factor s")
plt.ylabel("mixing rate")
plt.title("Figure 3: Mixing rate vs. std scale factor")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("./figures/figure3.png", dpi=150)
# ---8<-- [end:itemB-plot]

# %% [markdown]
# C - Analysis
#
# 1. Describe the overlap of the four classes in the original dataset (s=1). Could a single linear boundary separate all classes? What about a set of linear boundaries?
#
#    **A:** In the original dataset (s=1), the four classes form distinguishable clusters, although some overlap occurs, especially between the closer classes 0 and 1. A single linear boundary cannot separate four classes because it divides the plane into only two regions. A set of linear boundaries could create one decision region for each class and achieve good separation, but perfect classification would still be unlikely in the overlapping areas.
#
# 2. Sketch on Figure 1 the decision boundaries you think a trained neural network might learn.
#
#    **A:** A trained neural network would likely learn several piecewise-linear decision boundaries that divide the plane into four regions, one for each class. These boundaries would generally lie between neighboring clusters and adjust to their different positions and spreads. The boundary between classes 0 and 1 would be the most difficult to place because these classes have the greatest relative overlap.
#
# 3. Relate your sketch to item B: the more spread out the clouds are, what happens to the region where the network necessarily makes mistakes?
#
#    **A:** As the clouds become more spread out, the overlap between classes increases. Consequently, the regions where points from different classes are mixed become larger. In these regions, even an optimal neural network cannot classify every point correctly because similar points may belong to different classes. Therefore, increasing the standard deviation enlarges the unavoidable error regions and increases the expected misclassification rate.
