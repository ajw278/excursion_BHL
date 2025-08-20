#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt


# ---------- helpers (same as I shared before) ----------
def sph_to_cart_positions(r, theta, phi):
    Nr, Npix = len(r), len(theta)
    ux = np.sin(theta) * np.cos(phi)
    uy = np.sin(theta) * np.sin(phi)
    uz = np.cos(theta)
    R_flat = np.repeat(r, Npix)
    return np.column_stack([R_flat*np.tile(ux, Nr),
                            R_flat*np.tile(uy, Nr),
                            R_flat*np.tile(uz, Nr)])

def label_jeans_regions(X, unstable_mask, link_radius):
    idx_u = np.where(unstable_mask)[0]
    labels_full = np.full(X.shape[0], -1, dtype=int)
    if idx_u.size == 0:
        return labels_full, 0
    Xu = X[idx_u]
    tree = cKDTree(Xu)
    nbrs = tree.query_ball_point(Xu, r=link_radius)
    rows, cols = [], []
    for i, js in enumerate(nbrs):
        rows.extend([i]*len(js))
        cols.extend(js)
    G = csr_matrix((np.ones(len(rows), dtype=np.uint8),
                   (rows, cols)), shape=(Xu.shape[0], Xu.shape[0]))
    n_comp, labels_u = connected_components(G, directed=False)
    labels_full[idx_u] = labels_u
    return labels_full, n_comp

def plot_jeans_regions(X, labels, s=1.0, alpha=0.6, elev=20, azim=45,
                       title=None, max_points_per_region=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)

    stable = (labels < 0)
    if stable.any():
        idx = np.where(stable)[0]
        if max_points_per_region and idx.size > max_points_per_region:
            idx = np.random.choice(idx, max_points_per_region, replace=False)
        ax.scatter(X[idx,0], X[idx,1], X[idx,2], s=s, alpha=0.12, c="#bbbbbb", linewidths=0)

    n_regions = int(labels.max()) + 1
    base = plt.get_cmap("tab20").colors
    colors = np.array(base * ((n_regions + 20)//20 + 1))
    for rid in range(n_regions):
        idx = np.where(labels == rid)[0]
        if idx.size == 0:
            continue
        if max_points_per_region and idx.size > max_points_per_region:
            idx = np.random.choice(idx, max_points_per_region, replace=False)
        ax.scatter(X[idx,0], X[idx,1], X[idx,2],
                   s=s, alpha=alpha, c=[colors[rid]], linewidths=0, label=f"region {rid}")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("y [cm]"); ax.set_zlabel("z [cm]")
    if title: ax.set_title(title)
    if n_regions <= 12: ax.legend(loc="upper right", fontsize=8, frameon=False)
    return fig, ax


def plot_from_sph(seeds, r, theta, phi):
    # 3) 3D positions for ALL cells
    X = sph_to_cart_positions(r, theta, phi)
    tree_all = cKDTree(X)

    # 4) build union-of-balls mask for **unstable** seeds
    #    cells are marked unstable if inside ANY sphere from a seed with M/MJ > thresh
    thresh = 1.0  # Jeans-unstable threshold
    bad = np.where(seeds["M_over_MJ"].values > thresh)[0]
    unstable_mask = np.zeros(X.shape[0], dtype=bool)
    if bad.size > 0:
        for k in bad:
            ir = int(seeds.iloc[k]["ir"]); ip = int(seeds.iloc[k]["ipix"])
            idx = ir * Npix + ip
            Rk  = float(seeds.iloc[k]["R_kernel_cm"])
            nbrs = tree_all.query_ball_point(X[idx], r=Rk)
            unstable_mask[nbrs] = True
        print(f"Cells flagged unstable: {unstable_mask.sum()} of {X.shape[0]}")
    else:
        print("No seeds exceed the Jeans threshold; nothing to plot as unstable.")
        # Optionally loosen threshold for visualization only:
        # thresh = 0.5

    # 5) label connected unstable regions (linking length can be ~ few cell widths)
    #    A pragmatic default is link_radius = min(Rk over unstable seeds) or ~0.3*median(Rk)
    if bad.size > 0:
        link_radius = 0.3 * np.median(seeds.iloc[bad]["R_kernel_cm"].values)
    else:
        link_radius = 0.0  # unused if no regions
    labels, nreg = label_jeans_regions(X, unstable_mask, link_radius=link_radius)
    print(f"Connected Jeans-unstable regions: {nreg}")

    # 6) plot
    if nreg > 0:
        fig, ax = plot_jeans_regions(
            X, labels, s=0.6, alpha=0.7,
            title=f"Jeans-unstable regions (N={nreg})",
            max_points_per_region=20000  # limit per-region for speed/clarity
        )
        plt.tight_layout()
        plt.savefig("jeans_regions_3d.png", dpi=180)
        print("Saved figure:eans_regions_3d.png")
        plt.show()


# ---------- main: run seeds -> union of unstable spheres -> label -> plot ----------
if __name__ == "__main__":
        
    # --- import the pipeline to get seeds (Step 1 & 2) ---
    from core_analysis import run_pipeline
    
    # 1) run the analysis (adjust args as you like)
    seeds = run_pipeline(
        grid_path="/mnt/data/sph_coords_equalarea.npz",
        rho_path="/mnt/data/snapshot_0000_lnrho_eq.npy",
        out_csv="/mnt/data/core_candidates_from_phi_minima.csv",
        lmax=12,
        T=10.0, mu=2.33,
        f_kernel=0.5,
        max_eval=5000,
        fallback_if_no_minima=True
    )
    # 2) load grid again (for positions)
    grid = np.load("/mnt/data/sph_coords_equalarea.npz", allow_pickle=True)
    r = grid["r"]; theta = grid["theta"]; phi = grid["phi"]
    Nr, Npix = len(r), len(theta)

    # 3) 3D positions for ALL cells
    X = sph_to_cart_positions(r, theta, phi)
    tree_all = cKDTree(X)

    # 4) build union-of-balls mask for **unstable** seeds
    #    cells are marked unstable if inside ANY sphere from a seed with M/MJ > thresh
    thresh = 1.0  # Jeans-unstable threshold
    bad = np.where(seeds["M_over_MJ"].values > thresh)[0]
    unstable_mask = np.zeros(X.shape[0], dtype=bool)
    if bad.size > 0:
        for k in bad:
            ir = int(seeds.iloc[k]["ir"]); ip = int(seeds.iloc[k]["ipix"])
            idx = ir * Npix + ip
            Rk  = float(seeds.iloc[k]["R_kernel_cm"])
            nbrs = tree_all.query_ball_point(X[idx], r=Rk)
            unstable_mask[nbrs] = True
        print(f"Cells flagged unstable: {unstable_mask.sum()} of {X.shape[0]}")
    else:
        print("No seeds exceed the Jeans threshold; nothing to plot as unstable.")
        # Optionally loosen threshold for visualization only:
        # thresh = 0.5

    # 5) label connected unstable regions (linking length can be ~ few cell widths)
    #    A pragmatic default is link_radius = min(Rk over unstable seeds) or ~0.3*median(Rk)
    if bad.size > 0:
        link_radius = 0.3 * np.median(seeds.iloc[bad]["R_kernel_cm"].values)
    else:
        link_radius = 0.0  # unused if no regions
    labels, nreg = label_jeans_regions(X, unstable_mask, link_radius=link_radius)
    print(f"Connected Jeans-unstable regions: {nreg}")

    # 6) plot
    if nreg > 0:
        fig, ax = plot_jeans_regions(
            X, labels, s=0.6, alpha=0.7,
            title=f"Jeans-unstable regions (N={nreg})",
            max_points_per_region=20000  # limit per-region for speed/clarity
        )
        plt.tight_layout()
        plt.savefig("/mnt/data/jeans_regions_3d.png", dpi=180)
        print("Saved figure: /mnt/data/jeans_regions_3d.png")
        # plt.show()
