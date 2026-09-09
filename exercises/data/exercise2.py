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
# ### Exercise 2 - Non-Linearity in Higher Dimensions

# %% [markdown]
# A - Dataset I: Shifted Gaussians

# %%
# ---8<-- [start:itemA]
import numpy as np

params = {
    "A": {
        "mean": np.array([0, 0, 0, 0, 0]),
        "cov": np.array([
            [1.0, 0.8, 0.1, 0.0, 0.0],
            [0.8, 1.0, 0.3, 0.0, 0.0],
            [0.1, 0.3, 1.0, 0.5, 0.0],
            [0.0, 0.0, 0.5, 1.0, 0.2],
            [0.0, 0.0, 0.0, 0.2, 1.0],
        ]),
    },
    "B": {
        "mean": np.array([1.5, 1.5, 1.5, 1.5, 1.5]),
        "cov": np.array([
            [1.5, -0.7, 0.2, 0.0, 0.0],
            [-0.7, 1.5, 0.4, 0.0, 0.0],
            [0.2, 0.4, 1.5, 0.6, 0.0],
            [0.0, 0.0, 0.6, 1.5, 0.3],
            [0.0, 0.0, 0.0, 0.3, 1.5],
        ]),
    },
}

n_per_class = 500
rng = np.random.default_rng(seed=42)

samples_A = rng.multivariate_normal(
    mean=params["A"]["mean"],
    cov=params["A"]["cov"],
    size=n_per_class,
)

samples_B = rng.multivariate_normal(
    mean=params["B"]["mean"],
    cov=params["B"]["cov"],
    size=n_per_class,
)

# ==== Dataset I: Shifted Gaussians ====

X_I = np.vstack([samples_A, samples_B])
y_I = np.concatenate([
    np.full(n_per_class, "A"),
    np.full(n_per_class, "B"),
])

print("X_I shape:", X_I.shape)
print("y_I shape:", y_I.shape)
# ---8<-- [end:itemA]

# %% [markdown]
# B - Dataset II: Concentric Shells

# %%
# ---8<-- [start:itemB]
n_dimensions = 5
n_per_class = 500

def generate_shell(n_samples, mean_radius, radius_std, rng):
    v = rng.normal(
        loc=0,
        scale=1,
        size=(n_samples, n_dimensions)
    )

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    u = v / norms

    rho = rng.normal(
        loc=mean_radius,
        scale=radius_std,
        size=n_samples
    )

    X = rho[:, np.newaxis] * u

    return X


samples_C = generate_shell(
    n_samples=n_per_class,
    mean_radius=2.0,
    radius_std=0.4,
    rng=rng
)

samples_D = generate_shell(
    n_samples=n_per_class,
    mean_radius=5.0,
    radius_std=0.4,
    rng=rng
)

# ==== Dataset II: Concentric Shells ====

X_II = np.vstack([samples_C, samples_D])
y_II = np.concatenate([
    np.full(n_per_class, "C"),
    np.full(n_per_class, "D"),
])

print("X_II shape:", X_II.shape)
print("y_II shape:", y_II.shape)
# ---8<-- [end:itemB]

# %% [markdown]
# C - Visualize and Compare

# %%
# ---8<-- [start:itemC-pca]
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca_I = PCA(n_components=2)
X_I_pca = pca_I.fit_transform(X_I)

pca_II = PCA(n_components=2)
X_II_pca = pca_II.fit_transform(X_II)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Dataset I
for label, color in [("A", "tab:blue"), ("B", "tab:orange")]:
    mask = y_I == label

    axes[0].scatter(
        X_I_pca[mask, 0],
        X_I_pca[mask, 1],
        color=color,
        label=f"Class {label}",
        s=20,
        alpha=0.6,
    )

axes[0].set_title("Dataset I: shifted Gaussians")
axes[0].set_xlabel("Principal component 1")
axes[0].set_ylabel("Principal component 2")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Dataset II
for label, color in [("C", "tab:green"), ("D", "tab:red")]:
    mask = y_II == label

    axes[1].scatter(
        X_II_pca[mask, 0],
        X_II_pca[mask, 1],
        color=color,
        label=f"Class {label}",
        s=20,
        alpha=0.6,
    )

axes[1].set_title("Dataset II: concentric shells")
axes[1].set_xlabel("Principal component 1")
axes[1].set_ylabel("Principal component 2")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.suptitle("Figure 4: Two-dimensional PCA projections")
fig.tight_layout()
fig.savefig("./figures/figure4.png", dpi=150)
plt.show()
# ---8<-- [end:itemC-pca]


# %%
# ---8<-- [start:itemC-explained_variance]
def report_explained_variance(name, pca):
    ratios = pca.explained_variance_ratio_

    print(name)
    print(f"PC1: {ratios[0]:.4f} ({100 * ratios[0]:.2f}%)")
    print(f"PC2: {ratios[1]:.4f} ({100 * ratios[1]:.2f}%)")
    print(f"Total: {ratios.sum():.4f} ({100 * ratios.sum():.2f}%)")
    print()


report_explained_variance("Dataset I", pca_I)
report_explained_variance("Dataset II", pca_II)
# ---8<-- [end:itemC-explained_variance]


# %% [markdown]
# Explained Variance
#
# Dataset I: Shifted Gaussians <br>
# PC1: 0.5004 (50.04%)<br>
# PC2: 0.1593 (15.93%)<br>
# Total: 0.6597 (65.97%)
#
# Dataset II: Concentric Shells<br>
# PC1: 0.2159 (21.59%)<br>
# PC2: 0.2132 (21.32%)<br>
# Total: 0.4291 (42.91%)
#
# **A**: Dataset I better preserves the information relevant for classification in the 2D projection.

# %%
# ---8<-- [start:itemC-distance]
def center_distance(class_1, class_2):
    center_1 = class_1.mean(axis=0)
    center_2 = class_2.mean(axis=0)

    distance = np.linalg.norm(center_1 - center_2)

    return center_1, center_2, distance


center_A, center_B, distance_I = center_distance(
    samples_A,
    samples_B
)

center_C, center_D, distance_II = center_distance(
    samples_C,
    samples_D
)

print("Dataset I")
print("Center A:", np.round(center_A, 3))
print("Center B:", np.round(center_B, 3))
print(f"Distance: {distance_I:.3f}")

print("\nDataset II")
print("Center C:", np.round(center_C, 3))
print("Center D:", np.round(center_D, 3))
print(f"Distance: {distance_II:.3f}")
# ---8<-- [end:itemC-distance]

# %% [markdown]
# Distance Between the Class Centers
#
# Dataset I: Shifted Gaussians<br>
# Center A: [0.026, 0.085, 0.037, 0.019, 0.035]<br>
# Center B: [1.489, 1.499, 1.45, 1.458, 1.524]<br>
# Distance: 3.228
#
# Dataset II: Concentric Shells<br>
# Center C: [ 0.062,  0.017, -0.034, -0.001,  0.043]<br>
# Center D: [-0.108, -0.107, -0.142,  0.014,  0.166]<br>
# Distance: 0.266

# %%
# ---8<-- [start:itemC-radii]
radii_A = np.linalg.norm(samples_A, axis=1)
radii_B = np.linalg.norm(samples_B, axis=1)

radii_C = np.linalg.norm(samples_C, axis=1)
radii_D = np.linalg.norm(samples_D, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Dataset I
bins_I = np.linspace(
    min(radii_A.min(), radii_B.min()),
    max(radii_A.max(), radii_B.max()),
    30,
)

axes[0].hist(
    radii_A,
    bins=bins_I,
    alpha=0.6,
    color="tab:blue",
    label="Class A",
)

axes[0].hist(
    radii_B,
    bins=bins_I,
    alpha=0.6,
    color="tab:orange",
    label="Class B",
)

axes[0].set_title("Dataset I: shifted Gaussians")
axes[0].set_xlabel(r"Radius $\|x\|$")
axes[0].set_ylabel("Number of points")
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

# Dataset II
bins_II = np.linspace(
    min(radii_C.min(), radii_D.min()),
    max(radii_C.max(), radii_D.max()),
    30,
)

axes[1].hist(
    radii_C,
    bins=bins_II,
    alpha=0.6,
    color="tab:green",
    label="Class C",
)

axes[1].hist(
    radii_D,
    bins=bins_II,
    alpha=0.6,
    color="tab:red",
    label="Class D",
)

axes[1].set_title("Dataset II: concentric shells")
axes[1].set_xlabel(r"Radius $\|x\|$")
axes[1].set_ylabel("Number of points")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Figure 5: Distribution of the 5D radius")
fig.tight_layout()
fig.savefig("./figures/figure5.png", dpi=150)
plt.show()
# ---8<-- [end:itemC-radii]


# %% [markdown]
# D - Analysis
#
# 1. In Dataset II the distance between the centers is close to zero, yet the radius histograms are well separated. What does that combination tell you about the possibility of separating the classes with a hyperplane?
#
#     **A:** This combination shows that the difference between the classes is not their location but their distance from the origin. Class C forms an inner region, while Class D forms an outer shell surrounding it. A hyperplane separates points according to their position along a particular direction, so it cannot isolate the inner class from a class that surrounds it in every direction. Therefore, the classes are not linearly separable, despite their well-separated radii.
#
# 2. Explain why the structure of Dataset II cannot be solved by a linear boundary, no matter how much data is collected.
#
#     **A:** Dataset II has a concentric structure: Class C is concentrated around radius 2, and Class D is concentrated around radius 5. A linear boundary divides the space into two half-spaces, but separating these classes requires a closed boundary around the inner class. Collecting more data makes the concentric structure clearer, but it does not change its geometry. Therefore, no single linear boundary can correctly separate the classes. A nonlinear boundary, such as a hypersphere, is required.
#
# 3. PCA is a linear transformation. Discuss: does a 2D projection in which the classes look mixed prove that they are inseparable in the original space?
#
#     **A:** No. PCA projects the data onto a lower-dimensional linear subspace, so information contained in the discarded dimensions can be lost. In Dataset II, the first two principal components preserve only about 43% of the total variance. Furthermore, because the shell structure extends across all five dimensions, points with a large 5D radius may appear close to the origin after projection if much of their magnitude lies in the three discarded dimensions. Thus, the classes may look mixed in the PCA plot even though their 5D radii are well separated.
#
#
#     The radius histograms show that Dataset II can be separated using the nonlinear function
#
#     $$
#     f(x) = \lVert x \rVert_2
#         = \sqrt{x_1^2 + x_2^2 + x_3^2 + x_4^2 + x_5^2}.
#     $$
#
#     Since the classes have mean radii 2 and 5, a reasonable threshold is their midpoint:
#
#     $$
#     \hat{y}(x) =
#     \begin{cases}
#     C, & \text{if } \lVert x \rVert_2 < 3.5, \\
#     D, & \text{if } \lVert x \rVert_2 \geq 3.5.
#     \end{cases}
#     $$

# %%
def classify_dataset_II(X, threshold=3.5):
    radii = np.linalg.norm(X, axis=1)
    return np.where(radii < threshold, "C", "D")
