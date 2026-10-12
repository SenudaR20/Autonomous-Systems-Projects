# Autonomous Systems Projects — AMR Simulation with PyBullet

A collection of four autonomous robotics laboratory projects completed as part of the MSc Intelligent Automation programme at the University of Skövde, Sweden.

The projects progressively explore the development of an autonomous mobile robot (AMR) in a simulated environment, from basic robot control to obstacle avoidance and probabilistic localization.

## Overview

The objective of these projects was to develop and investigate core autonomous robotics capabilities using Python and PyBullet, a physics-based robotics simulation environment.

The four laboratories cover:

1. **Robot Control and Maze Environment Design** — Creating a custom maze and implementing keyboard-based control for a simulated mobile robot.
2. **Motion Planning and Motor Noise** — Executing a predefined route through a maze and investigating the effects of motor uncertainty on navigation.
3. **LiDAR-Based Obstacle Avoidance** — Simulating LiDAR sensing, detecting a moving pedestrian, and implementing stop-and-resume behaviour to avoid collisions.
4. **Robot Localization Using a Particle Filter** — Estimating the robot's position and orientation using simulated LiDAR measurements and wheel odometry.

## Technologies and Concepts

- **Programming:** Python
- **Simulation:** PyBullet
- **Robot type:** Autonomous Mobile Robot (AMR)
- **Sensing:** Simulated LiDAR and ray casting
- **Navigation:** Predefined route execution and motion control
- **Obstacle avoidance:** Distance-based obstacle detection and stop-and-resume behaviour
- **Localization:** Particle filtering and odometry
- **Core concepts:** Robotics simulation, sensor modelling, uncertainty, autonomous navigation, and state estimation

## Laboratory Projects

### Lab 1 — Robot Control and Maze Environment Design

Designed a custom maze environment using a grid-based map and configured a simulated mobile robot in PyBullet.

**Key work:**
- Created a maze using a binary grid representation.
- Configured the simulation environment and robot.
- Implemented keyboard-based remote control for robot movement.
- Investigated robot motion and simulation parameters.

**Key concepts:** PyBullet, environment modelling, robot control, differential-drive motion.

### Lab 2 — Route Execution and Motor Noise

Extended the first laboratory by programming the robot to follow a predefined route from its starting position towards the centre of the maze.

**Key work:**
- Developed a sequence of movement actions, including turning and forward motion.
- Controlled the duration of each action using simulation time steps.
- Introduced random variation into motor speed commands to simulate actuation uncertainty.
- Tested how motor noise affected the robot's ability to follow the planned route.

**Key concepts:** Motion control, route execution, wheel actuation, uncertainty, simulation-based testing.

### Lab 3 — LiDAR-Based Obstacle Avoidance

Implemented simulated LiDAR sensing and obstacle-avoidance behaviour in a maze containing a moving pedestrian.

**Key work:**
- Implemented simulated LiDAR measurements using ray casting.
- Used distance measurements to identify obstacles in the robot's path.
- Introduced a moving pedestrian that crossed and temporarily blocked the robot's route.
- Implemented stop-and-resume behaviour, allowing the robot to stop when the pedestrian obstructed its path and continue when the path became clear.
- Tested the robot's behaviour in a dynamic environment.

**Key concepts:** LiDAR simulation, ray casting, obstacle detection, collision avoidance, dynamic environments.

### Lab 4 — Robot Localization Using a Particle Filter

Investigated robot localization by estimating the robot's position and orientation within a known maze.

**Key work:**
- Used a simulated map-based scanning system to obtain wall-distance measurements.
- Incorporated wheel-rotation information through simulated odometry.
- Worked with particle-based representations of possible robot poses.
- Investigated position estimation by comparing sensor observations with the known environment.
- Evaluated localization behaviour using recorded simulation data.

**Key concepts:** Particle filters, probabilistic localization, odometry, sensor measurements, state estimation.

## Project Structure

```text
Autonomous-Systems-Projects/
│
├── README.md
├── requirements.txt
│
├── LAB1/
│   ├── main.py
│   ├── maps/
│   ├── utils/
│   └── ...
│
├── LAB2/
│   ├── main.py
│   ├── maps/
│   ├── utils/
│   └── ...
│
├── LAB3/
│   ├── main.py
│   ├── maps/
│   ├── utils/
│   └── ...
│
└── LAB4/
    ├── main.py
    ├── maps/
    ├── utils/
    └── ...
```

*Note: The structure above is illustrative. Adjust filenames and folders to match the actual project files.*

## Setup and Execution

### Prerequisites

- Python
- PyBullet
- NumPy
- Any additional dependencies required by the individual laboratories

### Installation

Clone the repository:

```bash
git clone https://github.com/SenudaR20/Autonomous-Systems-Projects.git
cd Autonomous-Systems-Projects
```

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

If a dependency file is not yet available, install the confirmed dependencies manually, for example:

```bash
pip install pybullet numpy
```

Run the relevant laboratory's entry-point script from its required working directory. The exact command depends on the structure of each laboratory.

## Learning Outcomes

Through these projects, I developed practical experience with:

- Building and configuring simulated mobile robots.
- Implementing robot motion control and route execution.
- Investigating the effect of actuation uncertainty on navigation.
- Simulating LiDAR sensing and detecting obstacles.
- Developing reactive obstacle-avoidance behaviour in a dynamic environment.
- Exploring probabilistic robot localization using particle filtering and odometry.
- Testing autonomous robot behaviour in physics-based simulation.

## Academic Context

These projects were completed as part of the MSc Intelligent Automation programme at the University of Skövde, Sweden.

The repository is intended to document my implementation, experiments, and learning in autonomous robotics and mobile robot systems.

## Notes

- This repository contains selected academic project work.
- Please refer to individual laboratory folders for implementation details and execution instructions.
- Dependencies, simulation settings, and expected behaviour may differ between laboratories.
- Academic starter code and third-party resources are subject to their respective licenses and usage permissions.
