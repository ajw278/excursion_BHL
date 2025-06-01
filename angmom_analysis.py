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
        interp = RegularGridInterpolator((x, y, z), field, bounds_error=False, fill_value=np.nan)
        return {'rho': interp(points)}

    elif field.ndim == 4 and field.shape[-1] == 3:  # Vector field
        result = {}
        for i, name in enumerate(['vx', 'vy', 'vz']):
            interp = RegularGridInterpolator((x, y, z), field[..., i], bounds_error=False, fill_value=np.nan)
            result[name] = interp(points)
        return result

    else:
        raise ValueError("Field must be a 3D (scalar) or 4D (vector) array.")


def process_snapshots(snapshot_dir, output_dir, radii_pc, nside=32):
    os.makedirs(output_dir, exist_ok=True)
    pc2cm = 3.086e18
    radii_cm = np.array(radii_pc) * pc2cm

    # Load snapshot list
    density_files = sorted(glob(os.path.join(snapshot_dir, "snapshot_*_density.npy")))

    for density_file in tqdm(density_files):
        snapshot_idx = int(os.path.basename(density_file).split("_")[1])
        velocity_file = density_file.replace("density", "velocity")

        rho = np.load(density_file)
        v = np.load(velocity_file)

        # Define Cartesian grid (assume cube centered at 0)
        nx = rho.shape[0]
        L = 0.5 * pc2cm  # Adjust if needed
        x = y = z = np.linspace(-L / 2, L / 2, nx)

        for r_cm in radii_cm:
            # Interpolate scalar
            rho_interp = interpolate_field_on_shell(rho, x, y, z, r_cm, nside)
            for key, val in rho_interp.items():
                np.save(os.path.join(output_dir, f"snapshot_{snapshot_idx:04d}_r{r_cm/pc2cm:.2f}_{key}_healpix.npy"), val)

            # Interpolate vector
            v_interp = interpolate_field_on_shell(v, x, y, z, r_cm, nside)
            for key, val in v_interp.items():
                np.save(os.path.join(output_dir, f"snapshot_{snapshot_idx:04d}_r{r_cm/pc2cm:.2f}_{key}_healpix.npy"), val)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Visualize HEALPix field stored as .npy array.")
	parser.add_argument("--process",  action=argparse.BooleanOptionalAction, default=False, help="Process to healpix")

	args = parser.parse_args()

	if args.process:
		process_snapshots(
			snapshot_dir="snapshots",
			output_dir="healpix_shells",
			radii_pc=[0.1, 0.2, 0.5],
			nside=32
		)
