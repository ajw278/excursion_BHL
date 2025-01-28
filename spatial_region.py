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
	def __init__(self, grid, filename='spatial_evolve'):
		"""
		Initialize the MultiResolutionArray. If a file with the given filename exists,
		load the object from the file. Otherwise, initialize the object and save it.

		Args:
			grid (trajectory_spatial_grid): The grid object containing rlevels and physical properties.
			filename (str): File path to save or load the object.

		Returns:
			None
		"""
		self.filename = filename

		# Check if the file exists
		if os.path.exists(self.filename):
			print(f"File {self.filename} exists. Loading MultiResolutionArray from file.")
			loaded_obj = self.load(self.filename)
			self.grid = loaded_obj.grid
			self.resolutions = loaded_obj.resolutions
			self.snapshots = loaded_obj.snapshots
		else:
			print(f"File {self.filename} does not exist. Initializing a new MultiResolutionArray.")
			self.grid = grid
			self.resolutions = self.generate_resolutions()
			self.snapshots = []
			self.save(self.filename)

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

		# Determine the restart point
		if self.snapshots:
			last_snapshot_time = len(self.snapshots) * dt_snap_sec
			print(f"Restarting from snapshot at time {last_snapshot_time / year2s / 1e6:.2f} Myr.")
		else:
			last_snapshot_time = 0.0
			print("Starting evolution from the beginning.")

		# Time evolution loop
		t = last_snapshot_time  # Start time in seconds
		next_snapshot_time = t + dt_snap_sec  # Time at which to store the next snapshot
		while t < Tend_sec:
			self.update_resolutions(dt)
			t += dt

			# Check if it's time to store a snapshot
			if t >= next_snapshot_time:
				snapshot = [np.copy(res) for res in self.resolutions]
				self.snapshots.append(snapshot)
				self.save(self.filename)  # Save the class state to the file
				print(f"Snapshot stored and saved at time {t / year2s / 1e6:.2f} Myr.")
				next_snapshot_time += dt_snap_sec

		print(f"Evolution completed: Total time = {Tend} Myr")

	def save(self, filename):
		"""
		Save the MultiResolutionArray object to a file.

		Args:
			filename (str): File path to save the object.

		Returns:
			None
		"""
		with open(filename, 'wb') as f:
			pickle.dump(self, f)
		print(f"MultiResolutionArray saved to {filename}.")

	@staticmethod
	def load(filename):
		"""
		Load a MultiResolutionArray object from a file.

		Args:
			filename (str): File path from which to load the object.

		Returns:
			MultiResolutionArray: The loaded object.
		"""
		with open(filename, 'rb') as f:
			obj = pickle.load(f)
		print(f"MultiResolutionArray loaded from {filename}.")
		return obj

	def create_video(self, output_filename="simulation.mp4", fps=10, downsample_factor=1):
		"""
		Create an MP4 video from the saved snapshots.

		Args:
			output_filename (str): The name of the output MP4 file.
			fps (int): Frames per second for the video.

		Returns:
			None
		"""
		if not self.snapshots:
			print("No snapshots available to create a video.")
			return

		print(f"Creating video from {len(self.snapshots)} snapshots...")

		# Set up the figure for plotting
		fig, ax = plt.subplots(figsize=(8, 6))
		im = None

		# Function to initialize the plot
		def init():
			nonlocal im
			surface_density = self.calculate_surface_density(self.snapshots[0])
			# Downsample the surface density
			surface_density = surface_density[::downsample_factor, ::downsample_factor]
			im = ax.imshow(
				surface_density.T,
				extent=[0, self.grid.rmax / pc2cm, 0, self.grid.rmax / pc2cm],
				origin="lower",
				aspect="auto",
				cmap="viridis",
				vmin=np.min(surface_density),
				vmax=np.percentile(surface_density, 90.0),
			)
			ax.set_title("Surface Density Evolution")
			ax.set_xlabel(r"$x \, [\mathrm{pc}]$")
			ax.set_ylabel(r"$y \, [\mathrm{pc}]$")
			fig.colorbar(im, ax=ax, label=r"Surface Density [$\mathrm{g \, cm^{-2}}$]")
			return [im]

		# Function to update the plot for each frame
		def update(frame_idx):
			print('Calculating SD...')
			surface_density = self.calculate_surface_density(self.snapshots[frame_idx])
			print('Downsampling...')
			surface_density = surface_density[::downsample_factor, ::downsample_factor]
			im.set_array(surface_density.T)
			ax.set_title(f"Snapshot {frame_idx + 1}/{len(self.snapshots)}")
			print(frame_idx)
			return [im]

		# Create the animation
		ani = animation.FuncAnimation(
			fig, update, frames=len(self.snapshots), init_func=init, blit=True
		)

		# Save the animation as an MP4 file
		ani.save(output_filename, writer="ffmpeg", fps=fps)
		plt.close(fig)
		print(f"Video saved as {output_filename}.")
	
	def calculate_surface_density(self, snapshot):
		"""
		Calculate the surface density from a snapshot.

		Args:
			snapshot (list of np.ndarray): A list of resolution arrays for one snapshot.

		Returns:
			np.ndarray: The surface density array.
		"""
		# Combine all levels into the finest resolution
		total = np.zeros(snapshot[-1].shape, dtype=np.float64)
		for ir, level_array in enumerate(snapshot):
			factor = np.array(total.shape) / np.array(level_array.shape)
			upsampled_array = zoom(level_array, factor, order=1)
			total += upsampled_array

		# Convert log density to linear density
		linear_density = np.exp(total)

		# Integrate over the z-axis to compute surface density
		dz = self.grid.rmax / total.shape[2]  # Assume uniform spacing in z
		surface_density = np.sum(linear_density, axis=2) * dz

		return surface_density

	def plot_surface_density(self):
		"""
		Compute and plot the surface density.

		- Converts the 3D log density into linear density.
		- Integrates the density along the z-axis to calculate surface density.
		- Plots the surface density.

		Returns:
			None
		"""
		# Evaluate the total perturbation at the finest resolution
		total = self.evaluate_at_finest()

		# Convert from log density to linear density
		linear_density = np.exp(total)

		# Compute the z-axis depth of each cell
		dz = self.grid.rmax / total.shape[2]  # Assume uniform spacing in z

		# Integrate over the z-axis to compute surface density
		surface_density = np.sum(linear_density, axis=2) * dz

		# Plot the surface density
		plt.figure(figsize=(8, 6))
		extent = [0, self.grid.rmax / pc2cm, 0, self.grid.rmax / pc2cm]  # Convert rmax to pc for axes
		plt.imshow(surface_density.T, extent=extent, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=np.percentile(surface_density, 80.0))
		plt.colorbar(label=r"Surface Density [$\mathrm{g \, cm^{-2}}$]")
		plt.xlabel(r"$x \, [\mathrm{pc}]$")
		plt.ylabel(r"$y \, [\mathrm{pc}]$")
		plt.title("Surface Density")
		plt.show()

# Assuming trajectory_spatial_grid is properly defined
pc2cm = 3.086e18  # Example constant
year2s = 3.154e7  # Seconds in a year
grid = exc.trajectory_grid(rmax=100.0 * pc2cm, rmin=0.5 * pc2cm, drfact=0.5)

# Initialize the MultiResolutionArray
mra = MultiResolutionArray(grid)

# Create or load MultiResolutionArray
filename = "multi_resolution_array.pkl"
mra = MultiResolutionArray(grid, filename)

# Evolve for 10 Myr, storing snapshots every 1 Myr
mra.evolve(Tend=20.0, fraction_of_tau=0.1, dt_snap=0.2)

# Create an MP4 video from the snapshots
mra.create_video(output_filename="evolution.mp4", fps=10)

# Evaluate the sum at the finest resolution
#summed_array = mra.evaluate_at_finest()
mra.plot_surface_density()

# Check the result
print("Summed array shape:", summed_array.shape)
			
