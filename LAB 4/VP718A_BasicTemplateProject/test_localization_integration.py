from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from ScannableMap import ScanableMap
from demo_odometry import DifferentialDriveOdometry
from particle_filter import FILTER_VERSION, ParticleFilter
from replay_localization import replay
from utils.demo_robot import DemoRobot


PROJECT_DIR = Path(__file__).resolve().parent
MAP_PATH = PROJECT_DIR / "maps" / "demoMap"
MAP_SCALE = 10
NUM_PARTICLES = 500
RANDOM_SEED = 42
INITIAL_POSE = (0.0, 0.0, 0.0)


def test_particle_filter_tracks_drive_and_replays_recording(tmp_path):
    map_lines = [line.strip() for line in MAP_PATH.read_text().splitlines() if line.strip()]
    p.connect(p.DIRECT)
    recorded_odometry_deltas = []
    recorded_lidar_scans = []
    recorded_pf_poses = []
    recorded_odometry_poses = []
    recorded_ground_truth_poses = []
    position_errors = []
    heading_errors = []

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        wall_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.5, 0.5, 0.5],
        )
        for row, line in enumerate(map_lines):
            for column, cell in enumerate(line):
                if cell in "12":
                    p.createMultiBody(
                        baseMass=0.5 if cell == "2" else 0,
                        baseCollisionShapeIndex=wall_shape,
                        basePosition=[column + 0.5, row + 0.5, 0.5],
                    )

        robot = DemoRobot(x=0, y=0)
        scanner = ScanableMap(
            map_dimensions=(12, 25),
            fov=180,
            max_distance=100,
            num_rays=robot.num_lidar_beams,
            scale=MAP_SCALE,
            endpoint=True,
        )
        scanner.load_map(str(MAP_PATH))
        odometry = DifferentialDriveOdometry(
            wheel_radius=0.15,
            wheel_base=0.60,
            left_wheel_joints=[1, 3],
            right_wheel_joints=[0, 2],
            use_simulator_pose=False,
        )
        odometry.update(robot.agv_id)

        np.random.seed(RANDOM_SEED)
        particle_filter = ParticleFilter(num_particles=NUM_PARTICLES)
        particle_filter.initialize(
            *INITIAL_POSE,
            scanner=scanner,
            scale=MAP_SCALE,
        )
        previous_odom_pose = odometry.get_pose()

        for key in (ord("w"), ord("a")):
            robot.user_control({key: p.KEY_IS_DOWN})
            for step in range(1, 1201):
                robot.step_action()
                p.stepSimulation()
                odometry.update(robot.agv_id)

                if step % 240 != 0:
                    continue

                current_odom_pose = odometry.get_pose()
                delta_x = current_odom_pose[0] - previous_odom_pose[0]
                delta_y = current_odom_pose[1] - previous_odom_pose[1]
                delta_theta = (
                    current_odom_pose[2] - previous_odom_pose[2] + np.pi
                ) % (2 * np.pi) - np.pi
                particle_filter.predict(delta_x, delta_y, delta_theta)

                observed_m, _ = robot.ray_cast_lidar()
                particle_filter.update_weights(
                    scanner,
                    observed_m * MAP_SCALE,
                    scale=MAP_SCALE,
                    measurement_sigma=5.0,
                )
                estimate = particle_filter.estimate()
                ess = particle_filter.effective_sample_size()
                true_position, true_orientation = p.getBasePositionAndOrientation(
                    robot.agv_id
                )
                true_heading = p.getEulerFromQuaternion(true_orientation)[2]

                recorded_odometry_deltas.append((delta_x, delta_y, delta_theta))
                recorded_lidar_scans.append(observed_m)
                recorded_pf_poses.append(estimate)
                recorded_odometry_poses.append(current_odom_pose)
                recorded_ground_truth_poses.append(
                    (true_position[0], true_position[1], true_heading)
                )
                position_errors.append(
                    np.hypot(estimate[0] - true_position[0], estimate[1] - true_position[1])
                )
                heading_errors.append(
                    abs(
                        np.arctan2(
                            np.sin(estimate[2] - true_heading),
                            np.cos(estimate[2] - true_heading),
                        )
                    )
                )

                if ess < particle_filter.num_particles / 2:
                    particle_filter.resample()
                previous_odom_pose = current_odom_pose
    finally:
        p.disconnect()

    recording_path = tmp_path / "recording.npz"
    np.savez_compressed(
        recording_path,
        odometry_delta=np.asarray(recorded_odometry_deltas),
        lidar_distances_m=np.asarray(recorded_lidar_scans),
        pf_pose=np.asarray(recorded_pf_poses),
        odometry_pose=np.asarray(recorded_odometry_poses),
        ground_truth_pose=np.asarray(recorded_ground_truth_poses),
        lidar_max_range_m=10.0,
        lidar_fov_deg=180.0,
        lidar_num_rays=21,
        map_scale=MAP_SCALE,
        num_particles=NUM_PARTICLES,
        initial_pose=INITIAL_POSE,
        random_seed=RANDOM_SEED,
        filter_version=FILTER_VERSION,
    )

    replayed_poses = replay(recording_path)

    assert len(recorded_pf_poses) == 10
    assert max(position_errors) < 0.8
    assert max(heading_errors) < np.radians(5)
    assert np.allclose(replayed_poses, recorded_pf_poses, atol=1e-12)
