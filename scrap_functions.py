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
			print(f"Saved precomputed volume density: {volume_density_filename}")"
			

	def save_snapshot(self, snapshot_idx):
		""" Save the current resolutions to disk as separate .npy files. """
		for level, array in enumerate(self.resolutions):
			filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_level_{level}.npy")
			np.save(filename, array)
		# Save clouds as well
		self.save_clouds()
		print(f"Snapshot {snapshot_idx} saved.")

	def save_clouds(self):
		"""Save all clouds as individual pickle files in the snapshot directory."""
		for idx, cloud in enumerate(self.clouds):
			filename = os.path.join(self.snapshot_dir, f"cloud_{idx:04d}.pkl")
			with open(filename, "wb") as f:
				pickle.dump(cloud, f)
		print(f"Saved {len(self.clouds)} clouds.")


	def load_snapshot(self, snapshot_idx):
		""" Load a snapshot from .npy files. """
		self.resolutions = []
		for level in range(len(self.grid.rlevels)):
			filename = os.path.join(self.snapshot_dir, f"snapshot_{snapshot_idx:04d}_level_{level}.npy")
			if os.path.exists(filename):
				self.resolutions.append(np.load(filename))
			else:
				print(f"Warning: Missing file {filename}. Snapshot might be incomplete.")
		# Load clouds
		self.load_clouds()
		print(f"Snapshot {snapshot_idx} loaded.")

	def load_clouds(self):
		"""Load all clouds from pickle files in the snapshot directory."""
		cloud_files = [f for f in os.listdir(self.snapshot_dir) if f.startswith("cloud_") and f.endswith(".pkl")]

		if not cloud_files:
			print("No previous clouds found. Starting fresh.")
			return

		self.clouds = []
		for filename in sorted(cloud_files):
			filepath = os.path.join(self.snapshot_dir, filename)
			with open(filepath, "rb") as f:
				cloud = pickle.load(f)
				self.clouds.append(cloud)

		print(f"Loaded {len(self.clouds)} clouds from snapshots.")

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

