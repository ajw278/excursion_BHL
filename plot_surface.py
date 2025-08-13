import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import os
import argparse

def load_field(base_path, snapshot_idx, radius_pc, field):
    fname_raw = f"snapshot_{snapshot_idx:04d}_r{radius_pc:.2f}_{field}_healpix.npy"
    print(f"Loading field '{field}' for snapshot {snapshot_idx} at radius {radius_pc} pc: {fname_raw}")
    fname = os.path.join(base_path, fname_raw)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing file: {fname}")
    return np.load(fname)

def plot_single_panel(data, title, unit, vmin=None, vmax=None, cmap='viridis'):
    hp.mollview(data, title=title, unit=unit, cmap=cmap, min=vmin, max=vmax)
    hp.graticule()
    plt.show()

def plot_multipanel(base_path, snapshot_idx, radius_pc, cmap='viridis', vlims=None, save=None):
    fig = plt.figure(figsize=(12, 10))
    fields = ['rho', 'vx', 'vy', 'vz']
    titles = [r'log $\rho$', r'$v_x$', r'$v_y$', r'$v_z$']

    for i, (field, title) in enumerate(zip(fields, titles), 1):
        data = load_field(base_path, snapshot_idx, radius_pc, field)
        if field == 'rho':
            data = np.log10(data)
            unit = "g cm$^{-3}$"
        else:
            unit = r"km/s"  
            data /= 1e5

        print(data)

        print(data.shape, data.dtype, np.nanmin(data), np.nanmax(data))

        vmin, vmax = None, None
        if vlims and field in vlims:
            vmin, vmax = vlims[field]
        elif field == 'rho':
            vmin, vmax = np.percentile(data[~np.isnan(data)], [10, 90])

        ax = plt.subplot(2, 2, i)
        hp.mollview(data, fig=fig.number, sub=(2, 2, i), title=title,
                    unit=unit, min=vmin, max=vmax, cmap=cmap, hold=True)
        hp.graticule()

    fig.suptitle(f"Snapshot {snapshot_idx:04d} — Radius = {radius_pc:.2f} pc", fontsize=16)
    if save:
        plt.savefig(save, bbox_inches='tight', dpi=200)
        print(f"Saved figure to {save}")
    else:
        plt.show()

def plot_shell_field(base_path, snapshot_idx, radius_pc, field='rho', cmap='viridis', vmin=None, vmax=None):
    data = load_field(base_path, snapshot_idx, radius_pc, field)
    title = f"{field.upper()} — snapshot {snapshot_idx:04d} @ {radius_pc:.2f} pc"
    if field == 'rho':
        data = np.log10(data)
        unit = r"log $N_{\rm H}$"
        if vmin is None or vmax is None:
            vmin, vmax = np.percentile(data[~np.isnan(data)], [10, 90])
    else:
        unit = 'km/s'
        data /= 1e5
    plot_single_panel(data, title, unit, vmin=vmin, vmax=vmax, cmap=cmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HEALPix shell data from simulation.")
    parser.add_argument("--base_path", type=str, default="healpix_shells", help="Directory with HEALPix .npy files")
    parser.add_argument("--snapshot", type=int, default=0, help="Snapshot index (default: 0)")
    parser.add_argument("--radius", type=float, default=1.0, help="Radius of shell in parsecs (default: 1.0)")
    parser.add_argument("--field", type=str, default=None, help="Field to plot (rho, vx, vy, vz). If not given, plots all in 2x2 panel.")
    parser.add_argument("--vmin", type=float, default=None, help="Minimum colorbar value")
    parser.add_argument("--vmax", type=float, default=None, help="Maximum colorbar value")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap name")
    parser.add_argument("--save", type=str, default=None, help="Output filename to save the plot (e.g., output.png)")

    args = parser.parse_args()

    if args.field:
        # Single panel mode
        plot_shell_field(
            base_path=args.base_path,
            snapshot_idx=args.snapshot,
            radius_pc=args.radius,
            field=args.field,
            cmap=args.cmap,
            vmin=args.vmin,
            vmax=args.vmax
        )
    else:
        # Multi-panel mode
        plot_multipanel(
            base_path=args.base_path,
            snapshot_idx=args.snapshot,
            radius_pc=args.radius,
            cmap=args.cmap,
            save=args.save
        )
