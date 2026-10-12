
import os
import pybullet as p
import numpy as np
import utils.world as W
import utils.demo_robot as R
from demo_odometry import DifferentialDriveOdometry
from ScannableMap import ScanableMap
from particle_filter import FILTER_VERSION, ParticleFilter
from utils.particle_display import ParticleDisplay

# The world and texture loaders use paths relative to the project directory.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)

# Create the world
wld = W.BaseWorld()

# Load the map
wld.map_from_file("maps/demoMap")
wld.texture_walls()

# Create the robot
r = R.DemoRobot(x=0, y=0)

# Create odometry
odom = DifferentialDriveOdometry(
    wheel_radius=0.15,
    wheel_base=0.60,
    left_wheel_joints=[1, 3],
    right_wheel_joints=[0, 2],
    distance_scale=0.979,
    rotation_scale=0.1098,
    use_simulator_pose=False
)

# Print odometry and simulator ground truth
def print_odometry():
    x, y, theta = odom.get_pose()

    pos, orn = p.getBasePositionAndOrientation(r.agv_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]

    print(
        f"Odom: x={x:.3f}, y={y:.3f}, "
        f"theta={np.degrees(theta):.1f}° | "
        f"True: x={pos[0]:.3f}, y={pos[1]:.3f}, "
        f"theta={np.degrees(true_theta):.1f}°"
    )

# Create the map scanner
scanner = ScanableMap(
    map_dimensions=(12, 25),
    fov=180,
    max_distance=100,
    num_rays=21,
    scale=10,
    endpoint=True
)
scanner.load_map("maps/demoMap")

# Create and initialize particle filter
RANDOM_SEED = 42
INITIAL_POSE = (0.0, 0.0, 0.0)
np.random.seed(RANDOM_SEED)
pf = ParticleFilter(num_particles=500)

pf.initialize(
    x=INITIAL_POSE[0],
    y=INITIAL_POSE[1],
    theta=INITIAL_POSE[2],
    position_noise=0.2,
    heading_noise=0.2,
    scanner=scanner,
    scale=10,
)
display = ParticleDisplay(scanner)
estimated_pose = pf.estimate()
ess = pf.effective_sample_size()

# Store the initial odometry pose
previous_odom_pose = odom.get_pose()

# Register robot callbacks
wld.add_step_callback(r.step_action)
wld.add_keyboard_callback(r.user_control)

# Particle filter update interval
scan_interval = 240
visualization_interval = 12
simulation_step = 0
recorded_odometry_deltas = []
recorded_lidar_scans = []
recorded_pf_poses = []
recorded_odometry_poses = []
recorded_ground_truth_poses = []


def get_ground_truth_pose():
    position, orientation = p.getBasePositionAndOrientation(r.agv_id)
    heading = p.getEulerFromQuaternion(orientation)[2]
    return position[0], position[1], heading


def save_recording():
    if not recorded_odometry_deltas:
        print("No localization samples recorded.")
        return

    recording_path = os.path.join(os.path.dirname(__file__), "recording.npz")
    np.savez_compressed(
        recording_path,
        odometry_delta=np.asarray(recorded_odometry_deltas),
        lidar_distances_m=np.asarray(recorded_lidar_scans),
        pf_pose=np.asarray(recorded_pf_poses),
        odometry_pose=np.asarray(recorded_odometry_poses),
        ground_truth_pose=np.asarray(recorded_ground_truth_poses),
        scan_interval=scan_interval,
        lidar_max_range_m=10.0,
        lidar_fov_deg=180.0,
        lidar_num_rays=r.num_lidar_beams,
        map_scale=10,
        num_particles=pf.num_particles,
        initial_pose=INITIAL_POSE,
        random_seed=RANDOM_SEED,
        filter_version=FILTER_VERSION,
    )
    print(f"Saved {len(recorded_odometry_deltas)} samples to {recording_path}")

# Run simulation
try:
    while True:
        wld.simStep()
        odom.update(r.agv_id)

        simulation_step += 1

        # Update particle filter once per scan interval
        if simulation_step % scan_interval == 0:

            # Get current odometry pose
            current_odom_pose = odom.get_pose()

            x, y, theta = current_odom_pose
            old_x, old_y, old_theta = previous_odom_pose

            # Calculate odometry movement
            delta_x = x - old_x
            delta_y = y - old_y

            # Wrap heading change to -pi ... pi
            delta_theta = (
                theta - old_theta + np.pi
            ) % (2 * np.pi) - np.pi

            # Predict particle movement
            pf.predict(delta_x, delta_y, delta_theta)

            # Get robot LiDAR observation
            observed_distances, observed_angles = r.ray_cast_lidar()

            # Convert metres to map-grid units
            observed_distances = observed_distances * 10

            # Update particle weights using the map
            pf.update_weights(
                scanner,
                observed_distances,
                scale=10,
                measurement_sigma=5.0
            )

            # Estimate robot pose
            estimated_x, estimated_y, estimated_theta = pf.estimate()

            # Calculate effective sample size
            ess = pf.effective_sample_size()

            # Record the sensor and filter data; ground truth is evaluation-only.
            recorded_odometry_deltas.append((delta_x, delta_y, delta_theta))
            recorded_lidar_scans.append(observed_distances / 10)
            recorded_pf_poses.append((estimated_x, estimated_y, estimated_theta))
            recorded_odometry_poses.append(current_odom_pose)
            recorded_ground_truth_poses.append(get_ground_truth_pose())

            # Print odometry, true pose and PF estimate
            print_odometry()

            print(
                f"PF: x={estimated_x:.3f}, "
                f"y={estimated_y:.3f}, "
                f"theta={np.degrees(estimated_theta):.1f}° | "
                f"ESS={ess:.1f}"
            )
            print("------------------------------")

            # Keep useful particles; only resample after substantial weight loss.
            if ess < pf.num_particles / 2:
                pf.resample()

            # Save odometry pose for next update
            previous_odom_pose = current_odom_pose

        if simulation_step % visualization_interval == 0:
            display.draw(
                pf,
                odom.get_pose(),
                estimated_pose,
                get_ground_truth_pose(),
                ess,
            )

finally:
    save_recording()
    display.close()
    try:
        wld.end()
    except Exception as e:
        print("World cleanup error:", e)