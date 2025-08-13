import numpy as np
import os
import healpy as hp
from scipy.interpolate import RegularGridInterpolator
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import os



def interpolate_field_on_shell(field, x, y, z, r_cm, nside):
    """
    Interpolates a 3D scalar or vector field onto a HEALPix shell at radius r_cm.

    Args:
        field (np.ndarray): 3D array (scalar) or 4D array (vector, last axis = 3) of the field.
        x, y, z (np.ndarray): 1D arrays defining the grid (must match field shape).
        r_cm (float): Radius of the spherical shell (in cm).
        nside (int): HEALPix nside resolution.

    Returns:
        dict: Dictionary with keys ['rho'] for scalar fields, or ['vx', 'vy', 'vz'] for vector fields.
              Each value is a 1D np.ndarray of length `npix`.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Cartesian coordinates of shell surface
    xs = r_cm * np.sin(theta) * np.cos(phi)
    ys = r_cm * np.sin(theta) * np.sin(phi)
    zs = r_cm * np.cos(theta)
    points = np.stack([xs, ys, zs], axis=-1)

    if field.ndim == 3:  # Scalar field
        
        interp = RegularGridInterpolator((x, y, z), field, bounds_error=True, fill_value=np.nan)
        return {'rho': interp(points)}

    elif field.ndim == 4 and field.shape[-1] == 3:  # Vector field
        result = {}
        for i, name in enumerate(['vx', 'vy', 'vz']):
            interp = RegularGridInterpolator((x, y, z), field[..., i], bounds_error=True, fill_value=np.nan)
            result[name] = interp(points)
        return result

    else:
        raise ValueError("Field must be a 3D (scalar) or 4D (vector) array.")


def load_grid_coords(snapshot_dir, expected_shape=None):
    """
    Load the 1D grid coordinates saved by the simulation as `snapshot_coords.npy`.

    Args:
        snapshot_dir (str): Directory containing snapshots and snapshot_coords.npy
        expected_shape (tuple or None): Expected (nx, ny, nz) to validate length.

    Returns:
        np.ndarray: 1D coordinate array of length nx, monotonically increasing.
    """
    coords_path = os.path.join(snapshot_dir, "snapshot_coords.npy")
    if not os.path.exists(coords_path):
        raise FileNotFoundError(
            f"Could not find '{coords_path}'. "
            "Make sure your simulation saved the coordinate file."
        )

    coords = np.load(coords_path)
    if coords.ndim != 1:
        raise ValueError(f"'snapshot_coords.npy' must be 1D, got shape {coords.shape}")

    if expected_shape is not None:
        nx, ny, nz = expected_shape[:3]
        if not (coords.size == nx == ny == nz):
            raise ValueError(
                "Coordinate length does not match snapshot grid shape: "
                f"len(coords)={coords.size}, grid={expected_shape}"
            )

    if not np.all(np.diff(coords) > 0):
        raise ValueError("Coordinates must be strictly increasing for RegularGridInterpolator.")

    return coords

def process_snapshots(snapshot_dir, output_dir, radii_pc, nside=32, correct_star_motion=False, ref_radius_pc=1e-2):
    os.makedirs(output_dir, exist_ok=True)
    pc2cm = 3.086e18
    radii_cm = np.array(radii_pc) * pc2cm

    # Load snapshot list
    density_files = sorted(glob(os.path.join(snapshot_dir, "snapshot_*_density.npy")))

    # Load grid coordinates once (and validate per-snapshot)
    coords = load_grid_coords(snapshot_dir)  # from earlier helper
    nx_coords = coords.size

    x = y = z = coords  # uniform cubic grid
    for density_file in tqdm(density_files):
        snapshot_idx = int(os.path.basename(density_file).split("_")[1])
        velocity_file = density_file.replace("density", "velocity")

        rho = np.load(density_file)
        v = np.load(velocity_file)
        # Define Cartesian grid (assume cube centered at 0)

        # Compute and save v_ref
        if correct_star_motion:
            r0_cm = ref_radius_pc * pc2cm
            v_ref = compute_reference_velocity(rho, v, coords, r0_cm)
            print(f"Computed reference velocity for snapshot {snapshot_idx}: {v_ref}")

            # Subtract globally before any interpolation
            v = v - v_ref  # broadcasts over (nx,ny,nz,3)
        else:
            v_ref = np.zeros(3, dtype=v.dtype)

        # Save v_ref for this snapshot
        vref_path = os.path.join(output_dir, f"snapshot_{snapshot_idx:04d}_vref.npy")
        np.save(vref_path, v_ref)


        for r_cm in radii_cm:
            # Interpolate scalar
            rho_interp = interpolate_field_on_shell(rho, x, y, z, r_cm, nside)
            for key, val in rho_interp.items():
                np.save(os.path.join(output_dir, f"snapshot_{snapshot_idx:04d}_r{r_cm/pc2cm:.2f}_{key}_healpix.npy"), val)

            # Interpolate vector
            v_interp = interpolate_field_on_shell(v, x, y, z, r_cm, nside)
            for key, val in v_interp.items():
                np.save(os.path.join(output_dir, f"snapshot_{snapshot_idx:04d}_r{r_cm/pc2cm:.2f}_{key}_healpix.npy"), val)

def compute_reference_velocity(rho, v, coords, r0_cm):
    """
    Density-weighted mean velocity inside a sphere of radius r0_cm.

    Args:
        rho (ndarray): 3D density array, shape (nx, ny, nz)
        v   (ndarray): 4D velocity array, shape (nx, ny, nz, 3)
        coords (ndarray): 1D coordinates (cm), length nx (same for x,y,z)
        r0_cm (float): reference radius in cm

    Returns:
        np.ndarray: shape (3,), the density-weighted mean velocity vector
    """
    x = coords[:, None, None]
    y = coords[None, :, None]
    z = coords[None, None, :]

    r = np.sqrt(x*x + y*y + z*z)
    mask = r <= r0_cm

    if not np.any(mask):
        # No cells within radius; return zero vector to be safe
        return np.zeros(3, dtype=v.dtype)

    w = rho[mask]  # weights
    wsum = w.sum()
    if wsum == 0.0 or not np.isfinite(wsum):
        return np.zeros(3, dtype=v.dtype)

    v_ref = np.array([
        np.sum(w * v[..., 0][mask]) / wsum,
        np.sum(w * v[..., 1][mask]) / wsum,
        np.sum(w * v[..., 2][mask]) / wsum,
    ], dtype=v.dtype)

    return v_ref


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize HEALPix field stored as .npy array.")
    parser.add_argument("--process", action=argparse.BooleanOptionalAction, default=False,
                        help="Process to healpix")
    parser.add_argument("--correct-star-motion", action=argparse.BooleanOptionalAction, default=False,
                        help="Subtract density-weighted mean velocity within ref radius")
    parser.add_argument("--ref-radius-pc", type=float, default=1e-2,
                        help="Reference radius in parsec for v_ref (default 1e-2 pc)")

    args = parser.parse_args()

    if args.process:
        process_snapshots(
            snapshot_dir="snapshots",
            output_dir="healpix_shells",
            radii_pc=[0.1, 0.2, 0.5],
            nside=32,
            correct_star_motion=args.correct_star_motion,
            ref_radius_pc=args.ref_radius_pc
        )