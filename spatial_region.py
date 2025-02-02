import numpy as np
from turbfuncs import *
from consts_defaults import *
import matplotlib.pyplot as plt
import excursion as exc
import cloud as cl
from scipy.ndimage import zoom
from scipy.special import erfinv
import pickle
import os

import matplotlib.animation as animation

plt.rc('text', usetex=True)

class MultiResolutionArray:
	def __init__(self, grid, snapshot_dir='snapshots'):
		"""
		Initialize the MultiResolutionArray. If a file with the given filename exists,
		load the object from the file. Otherwise, initialize the object and save it.

		Args:
			grid (trajectory_spatial_grid): The grid object containing rlevels and physical properties.
			filename (str): File path to save or load the object.

		Returns:
			None
		"""
		self.grid = grid
		self.resolutions = self.generate_resolutions()
		self.snapshot_dir = snapshot_dir

		# Create snapshot directory if it doesn't exist
		os.makedirs(self.snapshot_dir, exist_ok=True)

		# Try loading the latest snapshot
		latest_snapshot = self.get_latest_snapshot_index()
		if latest_snapshot is not None:
			print(f"Loading latest snapshot: {latest_snapshot}")
			self.load_snapshot(latest_snapshot)
		else:
			print("No previous snapshots found. Starting fresh.")

	def generate_resolutions(self):
		"""
		Generate arrays corresponding to each resolution level.

		Returns:
			list of np.ndarray: Arrays for each level, covering the same spatial area at appropriate resolutions.
		"""
		resolutions = []
		# Get the finest resolution (spanning the domain with rmax as the length scale)
		#finest_resolution_size = int(self.grid.rmax / self.grid.dr[-1])

		for ir, r in enumerate(self.grid.rlevels):
			# Define the resolution for the current level based on the length scale
			resolution_size = int(self.grid.rmax / self.grid.rlevels[ir])
			
			# Create and initialize the grid for this resolution level
			level_grid = self.initialize_resolution(ir, resolution_size)
	
			resolutions.append(level_grid)

		return resolutions
	
	def initialize_resolution(self, ilevel, resolution_size):
		"""
		Initialize the values of a single resolution grid.

		Args:
			ilevel (int): The level index for the resolution.
			resolution_size (int): The size of the resolution grid.

		Returns:
			np.ndarray: The initialized resolution grid.
		"""
		# Retrieve Delta_S for this level
		DS = self.grid.Delta_S[ilevel]

		# Create a uniform random array
		u_delta = np.random.uniform(size=(resolution_size, resolution_size, resolution_size))

		# Transform the uniform distribution using the inverse error function
		initialized_grid = np.sqrt(2.0 * DS) * erfinv(2.0 * u_delta - 1.0)

		return initialized_grid

	def evaluate_at_finest(self):
		"""
		Computes the sum of all levels at the finest resolution.

		Returns:
			np.ndarray: A 3D array with the sum of all resolutions, mapped to the finest grid.
		"""
		# Initialize the result array at the finest resolution
		total = np.zeros(self.resolutions[-1].shape, dtype=np.float64)

		for ir, level_array in enumerate(self.resolutions):
			# Compute the scaling factor for this level relative to the finest level
			factor = np.array(total.shape) / np.array(level_array.shape)

			# Upsample the current array to the finest resolution
			upsampled_array = zoom(level_array, factor, order=1)  # Linear interpolation

			# Add the upsampled array to the total
			total += upsampled_array

		return total
	
	def update_resolutions(self, dt):
		"""
		Update the resolutions for a given time-step.

		Args:
			dt (float): Time-step value.

		Returns:
			None: Updates self.resolutions in place.
		"""
		for ilevel in range(len(self.resolutions)):
			# Compute exponential decay term
			exp_decay = np.exp(-dt / self.grid.tau_R[ilevel])

			# Update the current resolution level
			Ddelta_new = self.resolutions[ilevel] * exp_decay
			random_term = np.random.normal(
				loc=0.0,
				scale=1.0,
				size=self.resolutions[ilevel].shape
			)
			Ddelta_new += random_term * np.sqrt(
				self.grid.Delta_S[ilevel] * (1.0 - np.exp(-2.0 * dt / self.grid.tau_R[ilevel]))
			)

			# Update the resolution level in place
			self.resolutions[ilevel] = Ddelta_new

	def evolve(self, Tend, fraction_of_tau=0.1, dt_snap=0.1):
		"""
		Perform time-steps up to a maximum time Tend and store snapshots every dt_snap.
		Restart from the last saved snapshot if it exists.

		Args:
			Tend (float): Maximum time in Myr.
			fraction_of_tau (float): Fraction of the minimum tau_R to use as the time-step size.
			dt_snap (float): Interval at which snapshots are stored, in Myr.

		Returns:
			None
		"""
		# Convert Tend and dt_snap from Myr to seconds
		Tend_sec = Tend * 1e6 * year2s
		dt_snap_sec = dt_snap * 1e6 * year2s

		# Determine the time-step size (fraction of the minimum tau_R)
		dt = fraction_of_tau * np.min(self.grid.tau_R)

		# Ensure dt is reasonable for the simulation duration
		if dt > Tend_sec:
			raise ValueError("Time-step size exceeds the total evolution time. Adjust fraction_of_tau or Tend.")

		#  Start evolution from latest snapshot if available
		latest_snapshot = self.get_latest_snapshot_index()
		if latest_snapshot is not None:
			t = latest_snapshot * dt_snap_sec
			print(f"Resuming from snapshot {latest_snapshot} at {t / year2s / 1e6:.2f} Myr.")
		else:
			t = 0.0
			latest_snapshot = -1
			print("Starting evolution from the beginning.")

		# Time evolution loop
		snapshot_idx = latest_snapshot + 1
		next_snapshot_time = t + dt_snap_sec
		while t < Tend_sec:
			self.update_resolutions(dt)
			t += dt

			if t >= next_snapshot_time:
				self.save_snapshot(snapshot_idx)
				snapshot_idx += 1
				next_snapshot_time += dt_snap_sec
		
		print(f"Evolution completed: Total time = {Tend} Myr")

	def save_snapshot(self, snapshot_idx):
		""" Save the current resolutions to disk as separate .npy files. """
		for level, array in enumerate(self.resolutions):
			filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_level_{level}.npy")
			np.save(filename, array)
		print(f"Snapshot {snapshot_idx} saved.")

	def load_snapshot(self, snapshot_idx):
		""" Load a snapshot from .npy files. """
		self.resolutions = []
		for level in range(len(self.grid.rlevels)):
			filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_level_{level}.npy")
			if os.path.exists(filename):
				self.resolutions.append(np.load(filename))
			else:
				print(f"Warning: Missing file {filename}. Snapshot might be incomplete.")
		print(f"Snapshot {snapshot_idx} loaded.")

	def get_latest_snapshot_index(self):
		""" Find the latest snapshot index from the saved files. """
		snapshot_files = [f for f in os.listdir(self.snapshot_dir) if f.startswith("snapshot_")]
		if not snapshot_files:
			return None
		snapshot_indices = sorted(set(int(f.split("_")[1]) for f in snapshot_files))
		return snapshot_indices[-1] if snapshot_indices else None


	def create_video(self, output_filename="simulation.mp4", fps=10):
		"""
		Create an MP4 video using precomputed volume densities.

		- If the volume densities are missing, they will be precomputed first.
		- The function loads precomputed 3D volume density `.npy` files and 
			computes surface densities on-the-fly.

		Args:
			output_filename (str): The name of the output MP4 file.
			fps (int): Frames per second for the video.

		Returns:
			None
		"""
		# Check if precomputed volume densities exist
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_volume.npy")
		))

		if not snapshot_indices:
			print("No precomputed volume densities found. Running precompute_volume_densities()...")
			self.precompute_volume_densities()
			snapshot_indices = sorted(set(
				int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_volume.npy")
			))

		if not snapshot_indices:
			print("No volume densities available. Exiting video creation.")
			return

		print(f"Creating video from {len(snapshot_indices)} precomputed volume densities...")

		# Load first snapshot to initialize the figure
		first_snapshot_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_indices[0]:04d}_volume.npy")
		first_volume_density = np.load(first_snapshot_filename)*self.grid.rho0
		first_surface_density = self.compute_surface_density(first_volume_density)
		first_surface_density= first_volume_density[:,:,0]

		# Setup figure
		fig, ax = plt.subplots(figsize=(8, 6))
		extent = [0, self.grid.rmax / pc2cm, 0, self.grid.rmax / pc2cm]
		im = ax.imshow(np.log10(first_surface_density.T), extent=extent, origin="lower", aspect="auto", cmap="hot", vmin=-24.0, vmax=-20.0)
		ax.set_xlabel(r"$x \, [\mathrm{pc}]$")
		ax.set_ylabel(r"$y \, [\mathrm{pc}]$")
		cbar = plt.colorbar(im, ax=ax, label=r"log Density [$\mathrm{g \, cm^{-3}}$]")

		# Function to update the animation
		def update(frame_idx):
			snapshot_idx = snapshot_indices[frame_idx]
			volume_density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_volume.npy")
			volume_density = np.load(volume_density_filename)
			volume_density *= self.grid.rho0
			surface_density = self.compute_surface_density(volume_density)
			surface_density= volume_density[:,:,0]

			im.set_array(np.log10(surface_density.T))
			#ax.set_title(f"Time = {snapshot_idx} Myr")
			return [im]

		# Create the animation
		ani = animation.FuncAnimation(fig, update, frames=len(snapshot_indices), blit=False)

		# Save the animation as MP4
		ani.save(output_filename, writer="ffmpeg", fps=fps)
		plt.close(fig)
		print(f"Video saved as {output_filename}.")

	def compute_volume_density(self, snapshot):
		"""
		Compute the volume density from a snapshot efficiently at the highest resolution.

		Args:
			snapshot (list of np.ndarray): A list of resolution arrays for one snapshot.

		Returns:
			np.ndarray: The volume density array at the finest resolution.
		"""
		total_linear_density = None

		for ir, level_array in enumerate(snapshot[::-1]):
			if total_linear_density is None:
				total_linear_density = level_array
			else:
				factor = np.array(total_linear_density.shape) / np.array(level_array.shape)
				upsampled_array = zoom(level_array, factor, order=1)
				total_linear_density += upsampled_array

		return np.exp(total_linear_density)
	

	def precompute_volume_densities(self):
		"""
		Precompute and save volume density maps for all existing snapshots at the finest resolution.

		This function loads all available snapshot levels, sums the density contributions to 
		the most refined grid, and saves the final 3D volume density as a `.npy` file.
		"""
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.startswith("snapshot_")
		))

		if not snapshot_indices:
			print("No snapshot files found.")
			return

		print(f"Precomputing volume densities for {len(snapshot_indices)} snapshots...")

		for snapshot_idx in snapshot_indices:
			volume_density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_volume.npy")

			# Skip if already precomputed
			if os.path.exists(volume_density_filename):
				continue

			# Load snapshot data
			snapshot = []
			for level in range(len(self.grid.rlevels)):
				filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_level_{level}.npy")
				if os.path.exists(filename):
					snapshot.append(np.load(filename))

			if not snapshot:
				print(f"Warning: No data found for snapshot {snapshot_idx}, skipping.")
				continue

			# Compute volume density at the finest resolution
			volume_density = self.compute_volume_density(snapshot)

			# Save the precomputed volume density
			np.save(volume_density_filename, volume_density)
			print(f"Saved precomputed volume density: {volume_density_filename}")
	
	def compute_surface_density(self, volume_density):
		"""
		Compute the surface density from a precomputed volume density.

		Args:
			volume_density (np.ndarray): The 3D volume density array.

		Returns:
			np.ndarray: The 2D surface density array.
		"""
		dz = self.grid.rmax / volume_density.shape[2]  # Cell depth
		surface_density = np.sum(self.grid.rho0*volume_density, axis=2) * dz
		return surface_density

# Assuming trajectory_spatial_grid is properly defined
pc2cm = 3.086e18  # Example constant
year2s = 3.154e7  # Seconds in a year
grid = exc.trajectory_grid(rmax=30.0 * pc2cm, rmin=0.2 * pc2cm, drfact=0.9)


# Initialize the MultiResolutionArray§
mra = MultiResolutionArray(grid)

# Create or load MultiResolutionArray
mra = MultiResolutionArray(grid)

# Evolve for 10 Myr, storing snapshots every 1 Myr
mra.evolve(Tend=10.0, fraction_of_tau=0.1, dt_snap=0.2)

# Create an MP4 video from the snapshots
mra.create_video(output_filename="evolution.mp4", fps=10)

			
