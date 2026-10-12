import os

import numpy as np
import pybullet as p

from demo_odometry import DifferentialDriveOdometry
from utils.demo_robot import DemoRobot


def test_low_speed_straight_odometry_matches_simulator():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(os.path.join(ROOT, "models"))
    p.loadURDF("plane.urdf")

    robot = DemoRobot(x=0, y=0)
    robot.left_speed = 4.0
    robot.right_speed = 4.0

    odom = DifferentialDriveOdometry(
        wheel_radius=0.15,
        wheel_base=0.60,
        left_wheel_joints=[1, 3],
        right_wheel_joints=[0, 2],
        distance_scale=0.87,
        rotation_scale=0.0325,
    )

    for _ in range(300):
        robot.step_action()
        p.stepSimulation()
        odom.update(robot.agv_id)

    pos, orn = p.getBasePositionAndOrientation(robot.agv_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]
    x, y, theta = odom.get_pose()

    assert abs(x - pos[0]) < 0.05, (x, pos[0])
    assert abs(y - pos[1]) < 0.05, (y, pos[1])
    assert abs(np.degrees(theta) - np.degrees(true_theta)) < 5.0, (
        np.degrees(theta),
        np.degrees(true_theta),
    )

    p.disconnect()


def test_default_odometry_matches_90_degree_turn_then_straight():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(os.path.join(ROOT, "models"))
    p.loadURDF("plane.urdf")

    robot = DemoRobot(x=0, y=0)
    odom = DifferentialDriveOdometry(
        wheel_radius=0.15,
        wheel_base=0.60,
        left_wheel_joints=[1, 3],
        right_wheel_joints=[0, 2],
    )
    odom.update(robot.agv_id)

    previous_yaw = 0.0
    accumulated_yaw = 0.0
    robot.left_speed = -8.0
    robot.right_speed = 8.0

    for _ in range(12000):
        robot.step_action()
        p.stepSimulation()
        odom.update(robot.agv_id)
        _, orientation = p.getBasePositionAndOrientation(robot.agv_id)
        yaw = p.getEulerFromQuaternion(orientation)[2]
        accumulated_yaw += np.arctan2(
            np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw)
        )
        previous_yaw = yaw
        if accumulated_yaw >= np.pi / 2:
            break
    else:
        raise AssertionError("Robot did not complete the 90-degree turn")

    assert abs(np.degrees(odom.get_pose()[2] - np.pi / 2)) < 3.0
    true_start = p.getBasePositionAndOrientation(robot.agv_id)[0]
    odom_start = odom.get_pose()

    robot.left_speed = -8.0
    robot.right_speed = -8.0
    for _ in range(300):
        robot.step_action()
        p.stepSimulation()
        odom.update(robot.agv_id)

    true_end = p.getBasePositionAndOrientation(robot.agv_id)[0]
    odom_end = odom.get_pose()
    true_displacement = np.array(true_end[:2]) - np.array(true_start[:2])
    odom_displacement = np.array(odom_end[:2]) - np.array(odom_start[:2])

    assert np.linalg.norm(odom_displacement - true_displacement) < 0.05, (
        odom_displacement,
        true_displacement,
    )
    assert abs(np.degrees(odom_end[2]) - 90.0) < 3.0

    p.disconnect()


def test_default_odometry_tracks_translation_during_pivot_turn():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(os.path.join(ROOT, "models"))
    p.loadURDF("plane.urdf")

    robot = DemoRobot(x=0, y=0)
    odom = DifferentialDriveOdometry(
        wheel_radius=0.15,
        wheel_base=0.60,
        left_wheel_joints=[1, 3],
        right_wheel_joints=[0, 2],
    )
    odom.update(robot.agv_id)
    robot.left_speed = 4.0
    robot.right_speed = -4.0

    max_position_error = 0.0
    for step in range(1, 3361):
        robot.step_action()
        p.stepSimulation()
        odom.update(robot.agv_id)

        if step % 240 == 0:
            true_position = p.getBasePositionAndOrientation(robot.agv_id)[0]
            odom_x, odom_y, _ = odom.get_pose()
            position_error = np.linalg.norm(
                np.array([odom_x, odom_y]) - np.array(true_position[:2])
            )
            max_position_error = max(max_position_error, position_error)

    true_position = p.getBasePositionAndOrientation(robot.agv_id)[0]
    odom_x, odom_y, odom_theta = odom.get_pose()
    assert max_position_error < 0.45, max_position_error
    assert np.linalg.norm(
        np.array([odom_x, odom_y]) - np.array(true_position[:2])
    ) < 0.20, (odom_x, odom_y, true_position[:2])
    assert abs(np.degrees(odom_theta) + 180.0) < 15.0, np.degrees(odom_theta)

    p.disconnect()


def test_odometry_turn_sign_matches_left_right_wheel_motion():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(os.path.join(ROOT, "models"))
    p.loadURDF("plane.urdf")

    robot = DemoRobot(x=0, y=0)
    odom = DifferentialDriveOdometry(
        wheel_radius=0.15,
        wheel_base=0.60,
        left_wheel_joints=[1, 3],
        right_wheel_joints=[0, 2],
        distance_scale=0.87,
        rotation_scale=0.0325,
    )

    # Simulate a turn where the left wheels are faster than the right wheels.
    # In this robot's PyBullet convention, that produces a negative yaw change.
    for joint_id in robot.left_wheels:
        p.resetJointState(robot.agv_id, joint_id, 0.5)
    for joint_id in robot.right_wheels:
        p.resetJointState(robot.agv_id, joint_id, -0.5)

    odom.update(robot.agv_id)

    # The first update initializes the prior wheel positions; this second update is the turn delta.
    for joint_id in robot.left_wheels:
        p.resetJointState(robot.agv_id, joint_id, 1.0)
    for joint_id in robot.right_wheels:
        p.resetJointState(robot.agv_id, joint_id, -1.0)

    odom.update(robot.agv_id)
    _, _, theta = odom.get_pose()

    assert theta < 0.0, theta

    p.disconnect()


def test_manual_forward_and_left_commands_match_map_axes():
    ROOT = os.path.dirname(os.path.abspath(__file__))
    p.connect(p.DIRECT)
    try:
        p.setGravity(0, 0, -9.81)
        p.setAdditionalSearchPath(os.path.join(ROOT, "models"))
        p.loadURDF("plane.urdf")

        robot = DemoRobot(x=0, y=0)
        odom = DifferentialDriveOdometry(
            wheel_radius=0.15,
            wheel_base=0.60,
            left_wheel_joints=[1, 3],
            right_wheel_joints=[0, 2],
        )
        odom.update(robot.agv_id)

        robot.user_control({ord("w"): p.KEY_IS_DOWN})
        for _ in range(300):
            robot.step_action()
            p.stepSimulation()
            odom.update(robot.agv_id)

        position, orientation = p.getBasePositionAndOrientation(robot.agv_id)
        assert position[0] > 0.4, position
        assert odom.get_pose()[0] > 0.4, odom.get_pose()

        robot.user_control({})
        for _ in range(20):
            robot.step_action()
            p.stepSimulation()
            odom.update(robot.agv_id)
        start_yaw = p.getEulerFromQuaternion(orientation)[2]

        robot.user_control({ord("a"): p.KEY_IS_DOWN})
        for _ in range(1200):
            robot.step_action()
            p.stepSimulation()
            odom.update(robot.agv_id)

        _, final_orientation = p.getBasePositionAndOrientation(robot.agv_id)
        final_yaw = p.getEulerFromQuaternion(final_orientation)[2]
        yaw_change = np.arctan2(
            np.sin(final_yaw - start_yaw),
            np.cos(final_yaw - start_yaw),
        )
        assert yaw_change > np.radians(45), np.degrees(yaw_change)
    finally:
        p.disconnect()
