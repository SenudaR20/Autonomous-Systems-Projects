
import numpy as np
from particle_filter import ParticleFilter
from ScannableMap import ScanableMap

np.random.seed(42)

# Load the map
scanner = ScanableMap(
    map_dimensions=(12, 25),
    fov=180,
    max_distance=100,
    num_rays=21,
    scale=10
)
scanner.load_map("maps/demoMap")

# Initialize particles around a test pose
pf = ParticleFilter(num_particles=500)
pf.initialize(x=2.5, y=2.5, theta=0.0)

# Generate a test observation at the chosen pose.
# This tests scan weighting with a map-generated scan;
# it is not yet a real robot lidar observation.
observed_distances, _ = scanner.scan(
    2.5 * 10,
    2.5 * 10,
    0.0
)

# Update particle weights
pf.update_weights(
    scanner,
    observed_distances,
    scale=10,
    measurement_sigma=5.0
)

print("Estimated pose after scan weighting:")
print(pf.estimate())

print("\nEffective sample size after scan weighting:")
print(pf.effective_sample_size())

print("\nSum of particle weights:")
print(np.sum(pf.particles[:, 3]))

# Resample when effective sample size is low
if pf.effective_sample_size() < pf.num_particles / 2:
    pf.resample()

print("\nEstimated pose after resampling:")
print(pf.estimate())

print("\nEffective sample size after resampling:")
print(pf.effective_sample_size())

print("\nSum of weights after resampling:")
print(np.sum(pf.particles[:, 3]))