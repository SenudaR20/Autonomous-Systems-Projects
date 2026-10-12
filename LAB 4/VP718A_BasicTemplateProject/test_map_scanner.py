
import numpy as np
from pathlib import Path
from ScannableMap import ScanableMap

# Create a 12 x 12 map.
# Each map cell is expanded to 10 x 10 grid units.
test_map = ScanableMap(
    map_dimensions=(12, 12),
    fov=360,
    max_distance=200,
    num_rays=4,
    scale=10
)

# Add a wall to the right of the robot.
test_map.set_solid(9, 3)

# First test
robot_x = 5.5 * test_map.scale
robot_y = 3.5 * test_map.scale
robot_theta = 0

distances, angles = test_map.scan(robot_x, robot_y, robot_theta)

print("Robot pose:", robot_x, robot_y, robot_theta)
print("\nPredicted lidar scan:")

for i, (angle, distance) in enumerate(zip(angles, distances)):
    print(f"Ray {i}: angle={np.degrees(angle):.1f} degrees, distance={distance:.2f} grid units")

right_ray_index = 2
expected_right_distance = 90 - robot_x

print("\nExpected right-facing distance:", expected_right_distance)
print("Actual right-facing distance:", distances[right_ray_index])

assert abs(distances[right_ray_index] - expected_right_distance) <= 0.5

print("\nBasic map scanner test passed!")


# Second test: move robot closer to the wall
robot_x = 75.0
robot_y = 35.0
robot_theta = 0

distances, angles = test_map.scan(robot_x, robot_y, robot_theta)

right_ray_index = 2
expected_distance = 90 - robot_x

print("\nSecond test:")
print("Expected right-facing distance:", expected_distance)
print("Actual right-facing distance:", distances[right_ray_index])

assert abs(distances[right_ray_index] - expected_distance) <= 0.5

print("Second scanner test passed!")


def test_lidar_fov_endpoints_match_robot_beams():
    scanner = ScanableMap(
        map_dimensions=(12, 25),
        fov=180,
        max_distance=100,
        num_rays=21,
        scale=10,
        endpoint=True,
    )

    _, angles = scanner.scan(50, 50, 0)
    expected = np.linspace(-np.pi / 2, np.pi / 2, 21)
    assert np.allclose(angles, expected)


def test_map_edge_is_not_treated_as_a_wall():
    scanner = ScanableMap(
        map_dimensions=(2, 2),
        fov=0,
        max_distance=20,
        num_rays=1,
        scale=1,
    )

    distances, _ = scanner.scan(1.5, 0.5, 0)
    assert distances[0] == scanner.max_distance


def test_robot_lidar_matches_map_scan_in_pybullet_world():
    import pybullet as p
    import pybullet_data

    from utils.demo_robot import DemoRobot

    map_path = Path(__file__).resolve().parent / "maps" / "demoMap"
    map_lines = [line.strip() for line in map_path.read_text().splitlines() if line.strip()]
    scale = 10
    p.connect(p.DIRECT)
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
                        baseMass=0,
                        baseCollisionShapeIndex=wall_shape,
                        basePosition=[column + 0.5, row + 0.5, 0.5],
                    )

        robot = DemoRobot(x=0, y=0)
        scanner = ScanableMap(
            map_dimensions=(12, 25),
            fov=180,
            max_distance=100,
            num_rays=robot.num_lidar_beams,
            scale=scale,
            endpoint=True,
        )
        scanner.load_map(str(map_path))
        observed, _ = robot.ray_cast_lidar()
        predicted, _ = scanner.scan(0, 0, 0)

        assert np.allclose(observed, predicted / scale, atol=0.05)
    finally:
        p.disconnect()