# excursion_BHL

The code required to make all of the figures for the paper Winter et al. 2024 are in the file 'paper_script.py' - simply run:

```console
python paper_script.py
```

The main script files are as follows:

- excursion.py: the class/functions that compute the trajectories for density/velocity perturbations following Hopkins 2012.
- cloud.py: class that computes collapse and star formation in a gravitationally unstable cloud undergoing free-fall, following Girichidis et al. 2014.
- star_forming_region.py: class that wraps around cloud.py and orbit.py to produce a sample of stellar trajectories through the ISM
- sfr_database.py: creates a database for the star forming regions produced by star_forming_region.py and computes disc evolution for a representative sample. Also contains several plotting functions.

