"""
Plot stitched density evolution for the *iterative* hierarchical manager
-----------------------------------------------------------------------

This script runs the IterativeHierarchicalManager for N realisations,
then stitches the parent-linked node snapshots into a single time series
per run, and finally produces:
  • an overlay of the initial density ρ(R, t=0) across runs
  • a per-run animation of ρ(R, t)

Requirements:
  - iterative_hier_manager.py (the iterative manager module provided earlier)
  - generate_cloudrho0.build_cloud_baseline_fn (to build a root baseline)
  - excursion.py, consts_defaults.py

Usage:
    python plot_iterative_snapshots.py \
        --n 3 --target_pc 1.0 \
        --Tend_myr 5.0 --dt_myr 0.02 --dt_snap_myr 0.1 \
        --out out_iter --writer auto --fps 12
"""
from __future__ import annotations

import os
import re
import json
import glob
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FFMpegWriter

from consts_defaults import pc2cm, Myr2s
import generate_cloudrho0 as gcr
import iterative_hier_manager as ihm

# -------------------------- helpers: picking level -------------------------- #

def pick_level_for_radius(grid, R_target_cm: float) -> int:
    r = np.asarray(grid.rlevels, float)
    return int(np.argmin(np.abs(r - float(R_target_cm))))

# -------------------------- helpers: stitching ----------------------------- #

META_FILE = "meta.json"
SNAP_RE = re.compile(r"snap_k(\d{5})\.npz$")

@dataclass
class ChainNode:
    dirpath: str
    node_id: str
    parent_id: Optional[str]
    inherit_until_k: Optional[int]  # how many snapshots to keep from THIS node (as specified by its CHILD)


def _read_meta(dirpath: str) -> Dict:
    mpath = os.path.join(dirpath, META_FILE)
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing {META_FILE} in {dirpath}")
    with open(mpath, "r") as f:
        return json.load(f)


def _list_snaps(dirpath: str) -> List[Tuple[int, str]]:
    files = sorted(glob.glob(os.path.join(dirpath, "snap_k*.npz")))
    out: List[Tuple[int, str]] = []
    for fp in files:
        m = SNAP_RE.search(os.path.basename(fp))
        if not m:
            continue
        k = int(m.group(1))
        out.append((k, fp))
    return out


def build_chain_from_leaf(leaf_dir: str) -> List[ChainNode]:
    """Return nodes from ROOT→…→LEAF with per-node inherit limits.

    The inherit limit for a parent node is stored in the CHILD's meta.json
    under `inherit_until_k`.
    """
    # Walk upward gathering (dir, meta) starting at leaf
    ladder: List[Tuple[str, Dict]] = []
    cur = leaf_dir
    while True:
        meta = _read_meta(cur)
        ladder.append((cur, meta))
        pid = meta.get("parent_id")
        if not pid:
            break
        # parent dir is sibling of current; search by node_id match
        parent_dir = None
        root = os.path.dirname(cur)
        for cand in glob.glob(os.path.join(root, "node_*")) + [os.path.join(root, "node_root")]:
            try:
                m2 = _read_meta(cand)
                if m2.get("node_id") == pid:
                    parent_dir = cand
                    break
            except Exception:
                continue
        if parent_dir is None:
            raise FileNotFoundError(f"Cannot find parent dir for node_id={pid} under {root}")
        cur = parent_dir

    # Ladder is LEAF→…→ROOT; reverse to ROOT→…→LEAF
    ladder = list(reversed(ladder))

    # Build ChainNode list with per-node inherit_limit (from child meta)
    chain: List[ChainNode] = []
    for i, (d, m) in enumerate(ladder):
        node_id = m.get("node_id")
        parent_id = m.get("parent_id")
        inherit_limit = None
        if i < len(ladder)-1:
            # child's meta tells how many frames to keep from THIS node
            child_meta = ladder[i+1][1]
            ilim = child_meta.get("inherit_until_k")
            inherit_limit = None if ilim is None else int(ilim)
        chain.append(ChainNode(dirpath=d, node_id=node_id, parent_id=parent_id, inherit_until_k=inherit_limit))
    return chain


def stitch_timeseries_from_leaf(leaf_dir: str) -> Dict[str, np.ndarray]:
    """Load parent-linked snapshots into one time series.

    Returns dict with keys:
      t_s: (Nt,)
      r_cm: (Nr,)
      rho_cm3: (Nt,Nr)
      Lcut: (Nt,)
      level_active: (Nt,)
    """
    chain = build_chain_from_leaf(leaf_dir)
    t_list: List[float] = []
    rho_rows: List[np.ndarray] = []
    Lcut_list: List[float] = []
    level_list: List[int] = []

    r_ref, Nr = None, None

    rho_crit = None

    for idx, node in enumerate(chain):
        snaps = _list_snaps(node.dirpath)
        if not snaps:
            continue
        kmax = node.inherit_until_k if node.inherit_until_k is not None else snaps[-1][0]
        for k, fp in snaps:
            if k > kmax:
                break
            d = np.load(fp)
            if rho_crit is None:
                rho_crit = d['rho_crit']
            r = d['rlevels_cm'].astype(float)
            rho = d['rho_filtered'].astype(float)
            t = float(d['time_s'][0])
            Lcut = float(d.get('baseline_Lcut', np.array([np.inf]))[0])
            lvl = int(d.get('level_active', np.array([0]))[0])

            if r_ref is None:
                r_ref = r.copy(); Nr = r_ref.size
            else:
                if r.size != Nr or not np.allclose(r, r_ref):
                    raise ValueError("Inconsistent rlevels across nodes; ensure a single grid is used.")

            t_list.append(t)
            rho_rows.append(rho)
            Lcut_list.append(Lcut)
            level_list.append(lvl)


    # sort by time
    order = np.argsort(t_list)
    t = np.array([t_list[i] for i in order], float)
    rho_ts = np.vstack([rho_rows[i][None, :] for i in order])
    Lcut = np.array([Lcut_list[i] for i in order], float)
    level = np.array([level_list[i] for i in order], int)

    return dict(t_s=t, r_cm=r_ref, rho_cm3=rho_ts, Lcut=Lcut, level_active=level, rho_crit=rho_crit)

# ------------------------------- plotting --------------------------------- #

def plot_initial_densities(leaf_dirs: List[str], out_png: str, r_unit: float = pc2cm) -> None:
    plt.figure(figsize=(7.2, 5.2))
    for i, d in enumerate(leaf_dirs):
        S = stitch_timeseries_from_leaf(d)
        r = S['r_cm'] / r_unit
        rho0 = S['rho_cm3'][0]
        plt.plot(r, rho0, lw=1.6, label=f"run {i+1}")
    plt.xscale('log'); plt.yscale('log')
    plt.xlabel('Scale R [pc]')
    plt.ylabel(r'$\rho(R, t{=}0)$ [g cm$^{-3}$]')
    if len(leaf_dirs) <= 12:
        plt.legend(ncol=2, fontsize=8)
    plt.grid(True, ls=':', alpha=0.4)
    os.makedirs(os.path.dirname(out_png) or '.', exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()
    print(f"Saved {out_png}")


def make_density_video(leaf_dir: str, out_path: str, fps: int = 12, writer: str = 'auto', r_unit: float = pc2cm) -> str:
    S = stitch_timeseries_from_leaf(leaf_dir)
    t = S['t_s']; r = S['r_cm']/r_unit; rho = S['rho_cm3'] ; rho_crit = S['rho_crit']
    Nt, Nr = rho.shape

    # Choose writer
    if writer == 'ffmpeg':
        W = FFMpegWriter(fps=fps, metadata={"title": "rho evolution"}); ext = 'mp4'
    elif writer == 'pillow':
        W = PillowWriter(fps=fps); ext = 'gif'
    else:
        try:
            W = FFMpegWriter(fps=fps, metadata={"title": "rho evolution"}); ext = 'mp4'
        except Exception:
            W = PillowWriter(fps=fps); ext = 'gif'

    if not out_path.endswith(('.mp4', '.gif')):
        out_path = out_path + ('.' + ext)

    # Axis limits from first/last frames
    rmin, rmax = np.nanmin(r), np.nanmax(r)
    pos = np.nan_to_num(np.concatenate([rho[0], rho[-1]]), nan=0.0)
    pos = pos[pos > 0]
    ymin, ymax = (1e-24, 1e-18)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(rmin, rmax); ax.set_ylim(ymin, ymax)
    ax.set_xlabel('Scale R [pc]')
    ax.set_ylabel(r'$\rho(R,t)$ [g cm$^{-3}$]')
    ax.grid(True, ls=':', alpha=0.4)
    line, = ax.plot([], [], lw=2.0)
    title = ax.set_title("")
    ax.plot(r, rho_crit, color='r')
    with W.saving(fig, out_path, dpi=180):
        for i in range(Nt):
            line.set_data(r, rho[i])
            title.set_text(f"t = {t[i]/Myr2s:.3f} Myr")
            W.grab_frame()
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path

# ------------------------------ end-to-end -------------------------------- #

def run_and_plot(
    n: int,
    target_radius_cm: float,
    outdir: str,
    Tend_myr: float,
    dt_myr: float,
    dt_snap_myr: float,
    dt_factor: float = 0.1,
    seed: Optional[int] = 1234,
    writer: str = 'auto',
    fps: int = 12,
):
    os.makedirs(outdir, exist_ok=True)

    leaf_dirs: List[str] = []
    for i in range(n):
        # Build root baseline + grid
        bfn_root, info = gcr.build_cloud_baseline_fn(target_radius_cm=target_radius_cm, drfact=(0.5)**(1./3.))
        grid = info['grid']
        root_level = pick_level_for_radius(grid, target_radius_cm)

        # Make a run folder; each run gets its own root node folder
        run_dir = os.path.join(outdir, f"run{i+1}")
        os.makedirs(run_dir, exist_ok=True)
        root_dir = os.path.join(run_dir, "node_root")

        mgr = ihm.IterativeHierarchicalManager(
            grid=grid,
            dt_factor=dt_factor,
            root_level=root_level,
            root_baseline_fn=bfn_root,
            save_root=root_dir,
            seed=None if seed is None else int(seed) + i,
        )

        mgr.evolve(Tend_s=Tend_myr*Myr2s,
                   dt_s=dt_myr*Myr2s,
                   dt_snap_s=dt_snap_myr*Myr2s,
                   verbose=False)

        leaf_dirs.append(mgr.leaf.shard.dirpath)

    # Plot initial densities
    plot_initial_densities(leaf_dirs, os.path.join(outdir, 'initial_densities.png'))

    # Videos
    vdir = os.path.join(outdir, 'videos'); os.makedirs(vdir, exist_ok=True)
    for i, d in enumerate(leaf_dirs):
        make_density_video(d, os.path.join(vdir, f"run{i+1}"), fps=fps, writer=writer)


# -------------------------------- CLI ------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--target_pc', type=float, default=1.0)
    ap.add_argument('--out', type=str, default='out_iter')
    ap.add_argument('--Tend_myr', type=float, default=5.0)
    ap.add_argument('--dt_myr', type=float, default=0.01)
    ap.add_argument('--dt_snap_myr', type=float, default=0.1)
    ap.add_argument('--dt_factor', type=float, default=0.1)
    ap.add_argument('--writer', type=str, default='auto', choices=['auto','ffmpeg','pillow'])
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--seed', type=int, default=1234)
    args = ap.parse_args()

    run_and_plot(
        n=args.n,
        target_radius_cm=args.target_pc*pc2cm,
        outdir=args.out,
        Tend_myr=args.Tend_myr,
        dt_myr=args.dt_myr,
        dt_snap_myr=args.dt_snap_myr,
        dt_factor=args.dt_factor,
        seed=args.seed,
        writer=args.writer,
        fps=args.fps,
    )

if __name__ == '__main__':
    main()
