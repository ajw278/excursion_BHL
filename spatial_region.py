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
	def __init__(self, grid, snapshot_dir='snapshots', imaxcoll=None):
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
		if imaxcoll is None:
			iall= np.arange(len(self.resolutions))
			imaxcoll = int(np.percentile(iall, 85.0))
		
		self.imaxcoll = imaxcoll
		self.t = 0.0

		self.clouds =  []
		self.icollapsed = [np.zeros(self.resolutions[ir].shape) for ir in range(len(self.resolutions[:imaxcoll]))]

		# Create snapshot directory if it doesn't exist
		os.makedirs(self.snapshot_dir, exist_ok=True)


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
			upsampled_array = zoom(level_array, factor, order=0)  

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

	def evolve(self, Tend, fraction_of_tau=0.1, dt_snap=0.2):
		"""
		Evolve the system until Tend, saving snapshots at dt_snap intervals.

		If a saved snapshot already extends to Tend, skip the run.

		Args:
			Tend (float): Maximum time in Myr.
			fraction_of_tau (float): Fraction of the minimum tau_R used for timestep.
			dt_snap (float): Time interval between snapshots (Myr).

		Returns:
			None
		"""
		# Convert to seconds
		Tend_sec = Tend * 1e6 * year2s
		dt_snap_sec = dt_snap * 1e6 * year2s
		dt = fraction_of_tau * np.min(self.grid.tau_R)

		# Check if we already have snapshots up to Tend
		snapshot_times_file = os.path.join(self.snapshot_dir, "snapshot_times.npy")
		if os.path.exists(snapshot_times_file):
			snapshot_times = np.load(snapshot_times_file)
			if len(snapshot_times) > 0 and snapshot_times[-1] >= Tend_sec:
				print(f"Snapshots already exist up to {Tend} Myr. Skipping evolution.")
				return
			
			last_snapshot_idx = len(snapshot_times) - 1
			t = snapshot_times[-1]  # Resume from last snapshot
			print(f"Resuming from {t / year2s / 1e6:.2f} Myr.")
		else:
			snapshot_times = []
			t = 0.0
			last_snapshot_idx = -1
			print("Starting evolution from the beginning.")

		snapshot_idx = last_snapshot_idx + 1
		next_snapshot_time = t + dt_snap_sec

		while t < Tend_sec:
			self.update_resolutions(dt)
			t += dt
			self.t = t

			# Evolve clouds
			for cloud in self.clouds:
				if not cloud.dispersed:
					cloud.evolve(t, rejuvinate=False)

			self.find_collapse()

			if t >= next_snapshot_time:
				self.save_snapshot(snapshot_idx, t)
				snapshot_times.append(t)
				np.save(snapshot_times_file, np.array(snapshot_times))  # Update time tracking
				snapshot_idx += 1
				next_snapshot_time += dt_snap_sec

		print(f"Evolution completed: Total time = {Tend} Myr")

	def save_snapshot(self, snapshot_idx, time):
		"""
		Save the finest resolution density and cloud properties.

		Args:
			snapshot_idx (int): Index of the snapshot.
			time (float): Time of the snapshot (seconds).

		Returns:
			None
		"""
		density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
		clouds_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

		# Compute finest density
		finest_density = self.evaluate_at_finest()

		# Save density
		np.save(density_filename, finest_density)

		# Save cloud properties
		if self.clouds:
			cloud_data = np.zeros((8, len(self.clouds)))  # 4 properties: (pos, size, vel, density)
			for i, cloud in enumerate(self.clouds):
				cloud_data[:3, i] = cloud.r  # Position
				cloud_data[3, i] = cloud.R    # Size
				cloud_data[4:7, i] = cloud.v
				cloud_data[7, i] = cloud.rho_med

			np.save(clouds_filename, cloud_data)

		print(f"Snapshot {snapshot_idx} saved at time {time / year2s / 1e6:.2f} Myr")


	def create_video(self, output_filename="simulation.mp4", fps=10):
		"""
		Create an MP4 video using precomputed density snapshots.

		- Loads saved `snapshot_times.npy` for time labels.
		- Loads the finest-resolution density for each snapshot.
		- Optionally overlays cloud positions as circles.
		
		Args:
			output_filename (str): The name of the output MP4 file.
			fps (int): Frames per second for the video.
		
		Returns:
			None
		"""
		# Load snapshot indices
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_density.npy")
		))

		if not snapshot_indices:
			print("No precomputed densities found. Exiting video creation.")
			return

		# Load time array
		snapshot_times_file = os.path.join(self.snapshot_dir, "snapshot_times.npy")
		if os.path.exists(snapshot_times_file):
			snapshot_times = np.load(snapshot_times_file)
		else:
			snapshot_times = np.array([i for i in range(len(snapshot_indices))])  # Dummy time array

		print(f"Creating video from {len(snapshot_indices)} density snapshots...")

		# Load first snapshot to initialize figure
		first_density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_indices[0]:04d}_density.npy")
		delta = np.load(first_density_filename)
		first_density= np.exp(delta)
		print(delta, self.grid.rho0, self.grid.rmax)

		# Select slice for visualization (midplane cut in z)
		first_surface_density =  self.compute_surface_density(first_density)

		# Compute 10th-90th percentile for color normalization
		vmin, vmax = np.percentile(first_surface_density, [10, 90])

		# Setup figure
		fig, ax = plt.subplots(figsize=(8, 6))
		extent = [0, self.grid.rmax / pc2cm, 0, self.grid.rmax / pc2cm]
		im = ax.imshow(np.log10(first_surface_density.T), extent=extent, origin="lower", aspect="auto", cmap="hot", vmin=np.log10(vmin), vmax=np.log10(vmax))

		# Add colorbar
		cbar = plt.colorbar(im, ax=ax, label=r"log Density [$\mathrm{g \, cm^{-3}}$]")

		# Set labels
		ax.set_xlabel(r"$x \, [\mathrm{pc}]$")
		ax.set_ylabel(r"$y \, [\mathrm{pc}]$")
		time_text = ax.text(0.05, 0.9, "", transform=ax.transAxes, color="white", fontsize=12)
		plt.show()

		# Function to update the animation
		def update(frame_idx):
			snapshot_idx = snapshot_indices[frame_idx]
			density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
			cloud_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

			# Load density
			density =  np.exp(np.load(density_filename))
			surface_density = self.compute_surface_density(density)  # Midplane slice


			im.set_array(np.log10(surface_density.T))

			# Clear previous circles
			for coll in ax.collections:
				coll.remove()
			[p.remove() for p in reversed(ax.patches)]

			# Load clouds for this snapshot (if available)
			if os.path.exists(cloud_filename):
				cloud_data = np.load(cloud_filename)
				positions = cloud_data[:3, :].T
				sizes = cloud_data[3, :]

				# Convert to PC units
				pos_pc = positions / pc2cm
				sizes_pc = sizes / pc2cm


				# Overlay clouds
				for pos, size in zip(pos_pc, sizes_pc):
					circle = plt.Circle((pos[0], pos[1]), size, color="cyan", fill=False, linewidth=1)
					ax.add_patch(circle)

			# Update time label
			time_text.set_text(f"Time = {snapshot_times[frame_idx] / 1e6 / year2s:.2f} Myr")
			print(time_text)
			return [im, time_text]

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
				upsampled_array = zoom(level_array, factor, order=0)
				total_linear_density += upsampled_array

		return np.exp(total_linear_density)

	
	def get_spatial_coordinates(self, ilevel, index):
		"""
		Compute the spatial coordinates of cell centers given their indices at a specific resolution level.

		Args:
			ilevel (int): The resolution level index.
			index (tuple or np.ndarray): A tuple (i, j, k) for a single point or an array of shape (N, 3) for multiple points.

		Returns:
			np.ndarray: A tuple (x, y, z) for a single index or an array of shape (N, 3) for multiple indices.
		"""
		# Compute grid spacing at this level
		N = int(self.grid.rmax / self.grid.rlevels[ilevel])  # Number of cells along each dimension
		dx = 2 * self.grid.rmax / N  # Grid spacing at this level

		# Convert to NumPy array if needed
		index = np.asarray(index)

		# Compute spatial coordinates using vectorized operations
		coords = (index + 0.5) * dx - grid.rmax

		return coords

	def find_collapse(self, **kwargs):
		"""
		"""
		delta_sum = None

		if delta_sum is None:
			delta_sum = np.zeros(self.resolutions[0].shape)

		for ir, level_array in enumerate(self.resolutions[:self.imaxcoll]):
			if ir>0:
				delta_c = self.grid.delta_c[ir]
				factor =  np.array(level_array.shape)/np.array(delta_sum.shape)
				upsampled_array = zoom(delta_sum, factor, order=0, grid_mode=False)
				delta_sum = upsampled_array+level_array

				#Check for any collapse on larger scales
				self.icollapsed[ir][delta_sum>1e5] = 1

				#Identify new collapses
				icollapse = (delta_sum>delta_c)&(self.icollapsed[ir]==0)

				for coll_ind in np.swapaxes(np.where(icollapse),0,1):
					print('New collapse:', coll_ind, ir, self.grid.rlevels[ir]/pc2cm)

					position  = self.get_spatial_coordinates(ir, coll_ind)

					self.clouds.append(cl.bound_clump(**kwargs))

					#Placeholder velocity vector 
					velocity = np.random.normal(size=3)*km2cm

					tform = self.t

					self.clouds[-1].form_nontraj(ir, self.grid, position, velocity, tform)

				self.icollapsed[ir][icollapse] = 1

				#Add large number to collapsed delta in order to flag that smaller scale cells 
				#have already collapsed
				delta_sum[icollapse] = 1e9

	def precompute_volume_densities(self):
		"""
		Precompute and save volume densities, overriding grid cells where clouds exist.
		"""
		snapshot_indices = sorted(set(
			int(f.split("_")[1]) for f in os.listdir(self.snapshot_dir) if f.endswith("_density.npy")
		))

		if not snapshot_indices:
			print("No snapshot files found.")
			return

		print(f"Precomputing volume densities for {len(snapshot_indices)} snapshots...")

		for snapshot_idx in snapshot_indices:
			density_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_density.npy")
			clouds_filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_clouds.npy")

			# Load density
			volume_density = self.grid.rho0*np.exp(np.load(density_filename))

			# Load clouds
			if os.path.exists(clouds_filename):
				cloud_data = np.load(clouds_filename)
				positions = cloud_data[:3, :].T
				radii = cloud_data[3, :]
				#velocities = cloud_data[4:7, :].T
				densities = cloud_data[7, :]

				# Create grid arrays
				grid_shape = volume_density.shape
				dx_finest = 2 * self.grid.rmax / grid_shape[0]
				x_vals = np.linspace(-self.grid.rmax + dx_finest / 2, self.grid.rmax - dx_finest / 2, grid_shape[0])
				y_vals = np.linspace(-self.grid.rmax + dx_finest / 2, self.grid.rmax - dx_finest / 2, grid_shape[1])
				z_vals = np.linspace(-self.grid.rmax + dx_finest / 2, self.grid.rmax - dx_finest / 2, grid_shape[2])
				X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')

				# Override density where clouds exist
				for pos, R, rho in zip(positions, radii, densities):
					dist_squared = (X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2
					volume_density[dist_squared <= R**2] = rho

			# Save updated density
			np.save(density_filename, volume_density)
			print(f"Updated volume density: {density_filename}")


	"""
	def precompute_volume_densities(self):
		#""
		#Precompute and save volume density maps for all existing snapshots at the finest resolution.

		#This function loads all available snapshot levels, sums the density contributions to 
		#the most refined grid, and saves the final 3D volume density as a `.npy` file.
		#""
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
			print(f"Saved precomputed volume density: {volume_density_filename}")"""
	
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
grid = exc.trajectory_grid(rmax=30.0 * pc2cm, rmin=0.2 * pc2cm, drfact=0.5)


# Initialize the MultiResolutionArray§
mra = MultiResolutionArray(grid)

# Create or load MultiResolutionArray
mra = MultiResolutionArray(grid)

# Evolve for 10 Myr, storing snapshots every 1 Myr
mra.evolve(Tend=5.2, fraction_of_tau=0.1, dt_snap=0.2)

# Create an MP4 video from the snapshots
mra.create_video(output_filename="evolution.mp4", fps=10)

			
