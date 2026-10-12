
import numpy as np
from pathlib import Path
from particle_filter import ParticleFilter
from ScannableMap import ScanableMap

# Make the test repeatable
np.random.seed(42)

# Create and initialize the filter
pf = ParticleFilter(num_particles=500)
pf.initialize(x=2.0, y=2.0, theta=0.0)

print("Initial estimate:")
print(pf.estimate())

print("\nInitial effective sample size:")
print(pf.effective_sample_size())

# Simulate a small forward movement
pf.predict(delta_x=0.1, delta_y=0.0, delta_theta=0.0)

print("\nEstimate after prediction:")
print(pf.estimate())

print("\nEffective sample size after prediction:")
print(pf.effective_sample_size())

# Test resampling
pf.resample()

print("\nEstimate after resampling:")
print(pf.estimate())

print("\nEffective sample size after resampling:")
print(pf.effective_sample_size())


def make_project_scanner():
	scanner = ScanableMap(
		map_dimensions=(12, 25),
		fov=180,
		max_distance=100,
		num_rays=21,
		scale=10,
		endpoint=True,
	)
	scanner.load_map(str(Path(__file__).resolve().parent / "maps" / "demoMap"))
	return scanner


def test_prediction_uses_odometry_heading_when_pf_heading_differs():
	pf = ParticleFilter(num_particles=20)
	pf.initialize(0, 0, 0, position_noise=0, heading_noise=0)
	pf.particles[:, 2] = np.pi / 2

	pf.predict(1.0, 0.0, 0.0, position_noise=0, heading_noise=0)

	assert np.allclose(pf.particles[:, 0], 0.0, atol=1e-12)
	assert np.allclose(pf.particles[:, 1], 1.0, atol=1e-12)


def test_heading_estimate_handles_wraparound():
	pf = ParticleFilter(num_particles=2)
	pf.particles[:, 2] = [np.pi - 0.05, -np.pi + 0.05]
	pf.particles[:, 3] = 0.5

	_, _, heading = pf.estimate()

	assert abs(abs(heading) - np.pi) < 0.01


def test_prediction_wraps_particle_headings():
	pf = ParticleFilter(num_particles=20)
	pf.initialize(0, 0, np.pi - 0.01, position_noise=0, heading_noise=0)

	pf.predict(0, 0, 0.05, position_noise=0, heading_noise=0)

	assert np.all(pf.particles[:, 2] <= np.pi)
	assert np.all(pf.particles[:, 2] >= -np.pi)
	assert np.isclose(pf.estimate()[2], -np.pi + 0.04)


def test_initialization_places_particles_in_free_map_cells():
	scanner = make_project_scanner()
	pf = ParticleFilter(num_particles=500)
	pf.initialize(0, 0, 0, scanner=scanner, scale=10)

	assert np.all(
		scanner.is_free(pf.particles[:, 0] * 10, pf.particles[:, 1] * 10)
	)


def test_scan_update_normalizes_weights_and_zero_weights_recover():
	scanner = make_project_scanner()
	pf = ParticleFilter(num_particles=100)
	pf.initialize(3.5, 3.5, 0, position_noise=0.01,
				  heading_noise=0.01, scanner=scanner, scale=10)
	observed, _ = scanner.scan(35, 35, 0)

	pf.update_weights(scanner, observed, scale=10)

	assert np.all(np.isfinite(pf.particles[:, 3]))
	assert np.isclose(np.sum(pf.particles[:, 3]), 1.0)
	assert 0 < pf.effective_sample_size() <= pf.num_particles

	pf.particles[:, 3] = 0.0
	assert pf.effective_sample_size() == 0.0
	pf.resample()
	assert np.isclose(np.sum(pf.particles[:, 3]), 1.0)
	assert np.isclose(pf.effective_sample_size(), pf.num_particles)


def test_resampling_roughens_particles_without_leaving_free_space():
	scanner = make_project_scanner()
	pf = ParticleFilter(num_particles=200)
	pf.initialize(
		3.5,
		3.5,
		0,
		position_noise=0,
		heading_noise=0,
		scanner=scanner,
		scale=10,
	)
	pf.particles[:, 3] = 0.0
	pf.particles[0, 3] = 1.0

	pf.resample()

	assert np.var(pf.particles[:, 0]) > 0
	assert np.var(pf.particles[:, 2]) > 0
	assert np.all(
		scanner.is_free(pf.particles[:, 0] * 10, pf.particles[:, 1] * 10)
	)
	assert np.isclose(np.sum(pf.particles[:, 3]), 1.0)