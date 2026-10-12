
import numpy as np

FILTER_VERSION = 2


class ParticleFilter:
    def __init__(self, num_particles=500):
        self.num_particles = num_particles
        self.scanner = None
        self.map_scale = 10
        self.resample_position_jitter = 0.12
        self.resample_heading_jitter = 0.05

        # Each particle stores: x, y, heading, weight
        self.particles = np.zeros((num_particles, 4))
        self.odom_theta = 0.0

    def initialize(self, x, y, theta, position_noise=0.2,
                   heading_noise=0.2, scanner=None, scale=10):
        """Initialize particles around an approximate starting pose."""
        self.particles[:, 0] = np.random.normal(
            x, position_noise, self.num_particles
        )
        self.particles[:, 1] = np.random.normal(
            y, position_noise, self.num_particles
        )
        self.particles[:, 2] = np.random.normal(
            theta, heading_noise, self.num_particles
        )
        self.particles[:, 2] = (
            self.particles[:, 2] + np.pi
        ) % (2 * np.pi) - np.pi
        self.particles[:, 3] = 1.0 / self.num_particles

        if scanner is not None:
            if not scanner.is_free(x * scale, y * scale):
                raise ValueError("The initial pose must be in a free map cell")

            valid = scanner.is_free(
                self.particles[:, 0] * scale,
                self.particles[:, 1] * scale,
            )
            for _ in range(100):
                if np.all(valid):
                    break
                invalid = ~valid
                self.particles[invalid, 0] = np.random.normal(
                    x, position_noise, np.count_nonzero(invalid)
                )
                self.particles[invalid, 1] = np.random.normal(
                    y, position_noise, np.count_nonzero(invalid)
                )
                valid[invalid] = scanner.is_free(
                    self.particles[invalid, 0] * scale,
                    self.particles[invalid, 1] * scale,
                )
            if not np.all(valid):
                raise RuntimeError("Could not place all particles in free map cells")

        self.odom_theta = theta
        self.scanner = scanner
        self.map_scale = scale

    def predict(self, delta_x, delta_y, delta_theta,
                position_noise=None, heading_noise=None):
        """
        Move particles using the odometry change.

        delta_x and delta_y are the change in the odometry's
        world-frame position, in metres.
        delta_theta is the heading change, in radians.
        """
        # Convert world-frame odometry displacement into the
        # previous robot-local frame.
        # This assumes odometry changes are small between updates.
        theta_odom = self.odom_theta

        local_dx = (
            np.cos(theta_odom) * delta_x
            + np.sin(theta_odom) * delta_y
        )
        local_dy = (
            -np.sin(theta_odom) * delta_x
            + np.cos(theta_odom) * delta_y
        )
        distance = np.hypot(local_dx, local_dy)
        if position_noise is None:
            position_noise = 0.10 * np.sqrt(distance)
        if heading_noise is None:
            heading_noise = (
                0.25 * np.sqrt(abs(delta_theta))
                + 0.05 * np.sqrt(distance)
            )

        # Apply local motion using each particle's own heading.
        particle_theta = self.particles[:, 2].copy()

        self.particles[:, 0] += (
            np.cos(particle_theta) * local_dx
            - np.sin(particle_theta) * local_dy
            + np.random.normal(0, position_noise, self.num_particles)
        )

        self.particles[:, 1] += (
            np.sin(particle_theta) * local_dx
            + np.cos(particle_theta) * local_dy
            + np.random.normal(0, position_noise, self.num_particles)
        )

        self.particles[:, 2] += (
            delta_theta
            + np.random.normal(0, heading_noise, self.num_particles)
        )

        # Keep headings within -pi to pi
        self.particles[:, 2] = (
            self.particles[:, 2] + np.pi
        ) % (2 * np.pi) - np.pi
        self.odom_theta = (
            self.odom_theta + delta_theta + np.pi
        ) % (2 * np.pi) - np.pi

    def update_weights(self, scanner, observed_distances,
                       scale=10, measurement_sigma=5.0):
        """
        Compare observed lidar distances with map-predicted distances.

        observed_distances must use the same ray angles, ray order,
        and distance units as scanner.scan().
        """
        observed_distances = np.asarray(observed_distances, dtype=float)
        if observed_distances.ndim != 1 or len(observed_distances) != scanner.num_rays:
            raise ValueError("Observed lidar distances must match the scanner ray count")
        if not np.all(np.isfinite(observed_distances)) or np.any(observed_distances < 0):
            raise ValueError("Observed lidar distances must be finite and non-negative")

        log_weights = np.zeros(self.num_particles)

        for i, particle in enumerate(self.particles):
            x, y, theta = particle[:3]

            # Particle position is in metres; scanner uses grid units.
            predicted_distances, _ = scanner.scan(
                x * scale,
                y * scale,
                theta
            )

            errors = observed_distances - predicted_distances

            # Gaussian sensor model, calculated in log space.
            log_weights[i] = -0.5 * np.sum(
                (errors / measurement_sigma) ** 2
            )

        valid_particles = scanner.is_free(
            self.particles[:, 0] * scale,
            self.particles[:, 1] * scale,
        )
        log_weights[~valid_particles] = -np.inf

        # Normalize without numerical underflow.
        finite_weights = np.isfinite(log_weights)
        if not np.any(finite_weights):
            self.particles[:, 3] = 1.0 / self.num_particles
            return

        log_weights[finite_weights] -= np.max(log_weights[finite_weights])
        weights = np.exp(log_weights)
        total = np.sum(weights)

        if total > 0 and np.isfinite(total):
            self.particles[:, 3] = weights / total
        else:
            self.particles[:, 3] = 1.0 / self.num_particles

    def resample(self):
        """Systematic resampling based on particle weights."""
        weights = np.where(
            np.isfinite(self.particles[:, 3]) & (self.particles[:, 3] > 0),
            self.particles[:, 3],
            0.0,
        )
        total = np.sum(weights)

        if total <= 0 or not np.isfinite(total):
            self.particles[:, 3] = 1.0 / self.num_particles
            return

        weights = weights / total

        positions = (
            np.random.random() + np.arange(self.num_particles)
        ) / self.num_particles

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.0

        indices = np.searchsorted(cumulative_sum, positions)

        resampled = self.particles[indices].copy()
        parent_poses = resampled[:, :3].copy()
        resampled[:, 0] += np.random.normal(
            0, self.resample_position_jitter, self.num_particles
        )
        resampled[:, 1] += np.random.normal(
            0, self.resample_position_jitter, self.num_particles
        )
        resampled[:, 2] += np.random.normal(
            0, self.resample_heading_jitter, self.num_particles
        )
        resampled[:, 2] = (
            resampled[:, 2] + np.pi
        ) % (2 * np.pi) - np.pi

        if self.scanner is not None:
            valid = self.scanner.is_free(
                resampled[:, 0] * self.map_scale,
                resampled[:, 1] * self.map_scale,
            )
            resampled[~valid, :3] = parent_poses[~valid]

        self.particles = resampled
        self.particles[:, 3] = 1.0 / self.num_particles

    def effective_sample_size(self):
        """Estimate how many particles effectively contribute."""
        weights = np.where(
            np.isfinite(self.particles[:, 3]) & (self.particles[:, 3] > 0),
            self.particles[:, 3],
            0.0,
        )
        total = np.sum(weights)

        if total <= 0 or not np.isfinite(total):
            return 0.0

        weights = weights / total
        return 1.0 / np.sum(weights ** 2)

    def estimate(self):
        """Return the weighted mean pose."""
        weights = np.where(
            np.isfinite(self.particles[:, 3]) & (self.particles[:, 3] > 0),
            self.particles[:, 3],
            0.0,
        )
        total = np.sum(weights)

        if total <= 0 or not np.isfinite(total):
            weights = np.ones(self.num_particles) / self.num_particles
        else:
            weights = weights / total

        x = np.sum(self.particles[:, 0] * weights)
        y = np.sum(self.particles[:, 1] * weights)

        # Circular mean for heading, avoiding angle wraparound issues.
        sin_theta = np.sum(np.sin(self.particles[:, 2]) * weights)
        cos_theta = np.sum(np.cos(self.particles[:, 2]) * weights)
        theta = np.arctan2(sin_theta, cos_theta)

        return x, y, theta