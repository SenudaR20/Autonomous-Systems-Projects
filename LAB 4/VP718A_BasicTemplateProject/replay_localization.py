import argparse
from pathlib import Path

import numpy as np

from ScannableMap import ScanableMap
from particle_filter import FILTER_VERSION, ParticleFilter


PROJECT_DIR = Path(__file__).resolve().parent
MAP_PATH = PROJECT_DIR / "maps" / "demoMap"
RECORDING_PATH = PROJECT_DIR / "recording.npz"


def replay(recording_path):
    with np.load(recording_path, allow_pickle=False) as data:
        odometry_deltas = data["odometry_delta"]
        lidar_distances_m = data["lidar_distances_m"]
        recorded_pf_poses = data["pf_pose"]
        ground_truth_poses = data["ground_truth_pose"]
        map_scale = int(data["map_scale"])
        lidar_max_range_m = float(data["lidar_max_range_m"])
        lidar_fov_deg = float(data["lidar_fov_deg"])
        lidar_num_rays = int(data["lidar_num_rays"])
        num_particles = int(data["num_particles"])
        initial_pose = tuple(data["initial_pose"])
        random_seed = int(data["random_seed"])
        recorded_filter_version = (
            int(data["filter_version"]) if "filter_version" in data else 1
        )

    sample_count = len(odometry_deltas)
    if not (
        len(lidar_distances_m)
        == len(recorded_pf_poses)
        == len(ground_truth_poses)
        == sample_count
    ):
        raise ValueError("Recording arrays have different sample counts")
    if sample_count == 0:
        print("The recording contains no localization samples.")
        return

    map_lines = [line.strip() for line in MAP_PATH.read_text().splitlines() if line.strip()]
    map_dimensions = (len(map_lines), max(len(line) for line in map_lines))
    scanner = ScanableMap(
        map_dimensions=map_dimensions,
        fov=lidar_fov_deg,
        max_distance=lidar_max_range_m * map_scale,
        num_rays=lidar_num_rays,
        scale=map_scale,
        endpoint=True,
    )
    scanner.load_map(str(MAP_PATH))

    np.random.seed(random_seed)
    particle_filter = ParticleFilter(num_particles=num_particles)
    particle_filter.initialize(
        *initial_pose,
        scanner=scanner,
        scale=map_scale,
    )

    replayed_poses = []
    position_errors = []
    report_every = max(1, sample_count // 10)

    for index, (odometry_delta, observed_scan_m) in enumerate(
        zip(odometry_deltas, lidar_distances_m)
    ):
        particle_filter.predict(*odometry_delta)
        particle_filter.update_weights(
            scanner,
            observed_scan_m * map_scale,
            scale=map_scale,
            measurement_sigma=5.0,
        )
        estimate = particle_filter.estimate()
        ess = particle_filter.effective_sample_size()
        replayed_poses.append(estimate)

        truth = ground_truth_poses[index]
        position_error = float(np.hypot(estimate[0] - truth[0], estimate[1] - truth[1]))
        position_errors.append(position_error)

        if ess < particle_filter.num_particles / 2:
            particle_filter.resample()

        if index % report_every == 0 or index == sample_count - 1:
            print(
                f"Sample {index + 1}/{sample_count}: "
                f"PF=({estimate[0]:.2f}, {estimate[1]:.2f}, "
                f"{np.degrees(estimate[2]):.1f} deg), "
                f"ESS={ess:.1f}, evaluation error={position_error:.2f} m"
            )

    replayed_poses = np.asarray(replayed_poses)
    position_errors = np.asarray(position_errors)
    if recorded_filter_version == FILTER_VERSION:
        replay_difference = np.max(np.abs(replayed_poses - recorded_pf_poses))
        print(
            f"Replay check: max pose difference from recorded PF estimates "
            f"{replay_difference:.6f}"
        )
    else:
        print(
            f"Pose comparison skipped: recording used PF model "
            f"v{recorded_filter_version}, current replay uses v{FILTER_VERSION}."
        )
    print(
        f"Evaluation only: mean position error {np.mean(position_errors):.3f} m, "
        f"final position error {position_errors[-1]:.3f} m"
    )
    return replayed_poses


def main():
    parser = argparse.ArgumentParser(description="Replay the recorded Lab 4 localization run.")
    parser.add_argument(
        "recording",
        nargs="?",
        type=Path,
        default=RECORDING_PATH,
        help="path to recording.npz (default: this project directory)",
    )
    args = parser.parse_args()
    replay(args.recording)


if __name__ == "__main__":
    main()
