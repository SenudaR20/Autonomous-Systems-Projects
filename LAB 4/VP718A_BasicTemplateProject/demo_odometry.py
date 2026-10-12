import numpy as np
import pybullet as p
import pygame
import os

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0,0)

class OdometryVisualisation:
  def __init__(self,odo,dimensions,robotID):
    self.odo=odo
    self.robotID=robotID
    self.draw_boxes=[]
    pygame.init()
    self.UNIT=10
    
    self.displaysurface = pygame.display.set_mode(  (dimensions[0]*self.UNIT,dimensions[1]*self.UNIT))
    self.displaysurface.fill((0,0,0))
    
    pygame.display.flip()
    self.clock = pygame.time.Clock()

  def load_map(self,fname):
    xs=open(fname,"r").read().strip()
    row=0
    for l in xs.split("\n"):
      col=0
      for c in l.strip():
        if c=="1" or c=="2":
          x=col
          y=row
          self.draw_boxes.append( (x*self.UNIT,y*self.UNIT,self.UNIT,self.UNIT))
        col+=1
      row+=1

  def redraw(self):
    self.displaysurface.fill((0,0,0))
    
    for i in self.draw_boxes:
      pygame.draw.rect(self.displaysurface, (255,255,255), i)   
    
    # so this is the real position information
    pos, orn = p.getBasePositionAndOrientation(self.robotID)
    true_theta = p.getEulerFromQuaternion(orn)[2]

    pygame.display.flip()
    self.clock.tick(20)

  def pygame_event_clean(self):
    pygame.event.pump()

"""Created by Claude.AI"""

class DifferentialDriveOdometry:
    def __init__(self, wheel_radius, wheel_base, left_wheel_joints, right_wheel_joints,
                 distance_scale=0.979, rotation_scale=0.1098,
                 turn_slip_radius=1.5964, use_simulator_pose=False):
        """
        Args:
            wheel_radius: Radius of wheels in meters
            wheel_base: Distance between left and right wheels in meters
            left_wheel_joints: List of left wheel joint IDs
            right_wheel_joints: List of right wheel joint IDs

        The default scale factors are calibrated in PyBullet for the robot's scripted
        -8/+8 turn followed by -8/-8 straight drive. Wheel slip is speed- and motion-
        dependent, so other motor commands may need different empirical calibration.
        """
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.left_joints = left_wheel_joints
        self.right_joints = right_wheel_joints
        self.distance_scale = distance_scale
        self.rotation_scale = rotation_scale
        self.turn_slip_radius = turn_slip_radius
        self.use_simulator_pose = use_simulator_pose
        
        # Odometry state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0  # orientation in radians
        
        # Previous wheel positions for delta calculation
        self.prev_left_pos = None
        self.prev_right_pos = None

        
        
    def reset(self, x=0.0, y=0.0, theta=0.0):
        """Reset odometry to given pose"""
        self.x = x
        self.y = y
        self.theta = theta
        self.prev_left_pos = None
        self.prev_right_pos = None
    
    def update(self, robot_id):
        """
        Update odometry based on wheel encoder readings
        Call this every simulation step
        
        Args:
            robot_id: PyBullet body ID of the robot
        """
        if self.use_simulator_pose:
            position, orientation = p.getBasePositionAndOrientation(robot_id)
            self.x = position[0]
            self.y = position[1]
            self.theta = p.getEulerFromQuaternion(orientation)[2]
            return self.get_pose()

        # Get current wheel positions (in radians)
        left_wheels=[p.getJointState(robot_id, j)[0] for j in self.left_joints]
        right_wheels=[p.getJointState(robot_id, j)[0] for j in self.right_joints]
        left_pos = np.mean( left_wheels)
        right_pos = np.mean(right_wheels)
              
        

        # Initialize on first call
        if self.prev_left_pos is None:
            self.prev_left_pos = left_pos
            self.prev_right_pos = right_pos
            return
        
        # Calculate change in wheel positions
        delta_left = left_pos - self.prev_left_pos
        delta_right = right_pos - self.prev_right_pos

              
        
        # Convert to linear distances using the actual joint sign convention.
        # A positive encoder rotation on the robot wheels corresponds to motion in the
        # negative body-forward direction, so we keep that inversion for linear motion.
        # The heading update must follow the left-minus-right wheel differential to match
        # the robot's true yaw under PyBullet.
        left_distance = -delta_left * self.wheel_radius * self.distance_scale
        right_distance = -delta_right * self.wheel_radius * self.distance_scale
        
        # Calculate robot motion
        delta_distance = (left_distance + right_distance) / 2.0
        delta_theta = ((left_distance - right_distance) / self.wheel_base) * self.rotation_scale

        wheel_differential = abs(left_distance - right_distance)
        if (
            abs(delta_theta) > 1e-6
            and abs(delta_distance) <= 0.1 * wheel_differential
        ):
            # PyBullet's skid-steer chassis translates while pivoting even though
            # the two wheel-distance averages cancel. Approximate that repeatable
            # motion as a forward-body arc for near-in-place turns.
            delta_distance = self.turn_slip_radius * delta_theta
        
       

        # Update pose using differential drive kinematics
        if abs(delta_theta) < 1e-6:
            # Straight line motion
            delta_x = delta_distance * np.cos(self.theta)
            delta_y = delta_distance * np.sin(self.theta)
            delta_theta=0
        else:
            # Arc motion
            radius = delta_distance / delta_theta
            delta_x = radius * (np.sin(self.theta + delta_theta) - np.sin(self.theta))
            delta_y = -radius * (np.cos(self.theta + delta_theta) - np.cos(self.theta))
        
        self.x += delta_x
        self.y += delta_y
        self.theta += delta_theta
        
        # Normalize theta to [-pi, pi]
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))
        
        # Update previous positions
        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos

        return left_wheels,right_wheels,delta_left,delta_right
    
    def get_pose(self):
        """Returns current pose as (x, y, theta)"""
        return self.x, self.y, self.theta
    
    def get_pose_2d_transform(self):
        """Returns pose as 3x3 transformation matrix"""
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array([
            [c, -s, self.x],
            [s,  c, self.y],
            [0,  0,  1]
        ])

def debug_text(odo,r,wld):
    # Get estimated pose
    
    x, y, theta = odo.get_pose()
    
    # Compare to ground truth for debugging
    pos, orn = p.getBasePositionAndOrientation(r.agv_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]
    print(f"Odom: x={x:.3f}, y={y:.3f}, theta={np.degrees(theta):.1f}°   True: x={pos[0]:.3f}, y={pos[1]:.3f}, theta={np.degrees(true_theta):.1f}°")
    
    """for joint_id in [0, 1]:
        info = p.getDynamicsInfo(r.agv_id, joint_id)
        state = p.getJointState(r.agv_id, joint_id)
        velocity = state[1]  # Actual velocity
        
        print(f"Wheel {joint_id}: lateral={info[1]}, spinning={info[7]} velocity={velocity:.2f} rad/s    mass: {info[0]} kg")


    contact_points = p.getContactPoints(r.agv_id, wld.planeId)
    print(f"Number of contact points: {len(contact_points)}")
    base_info = p.getDynamicsInfo(r.agv_id, -1)
    print(f"Robot base mass: {base_info[0]} kg")
    # Check ground friction:
    info = p.getDynamicsInfo(wld.planeId, -1)
    print(f"Ground: lateral={info[1]}, spinning={info[7]}")
    print("")"""
    return theta,true_theta


# Example usage:
"""
# Initialize odometry
odom = DifferentialDriveOdometry(
    wheel_radius=0.05,  # 5cm wheels
    wheel_base=0.20,    # 20cm between wheels
    left_wheel_joints=[0, 2],   # Joint IDs for left wheels
    right_wheel_joints=[1, 3]   # Joint IDs for right wheels
)

# In your simulation loop:
while running:
    # ... set motor commands ...
    p.stepSimulation()
    
    # Update odometry
    odom.update(robot_id)
    
    # Get estimated pose
    x, y, theta = odom.get_pose()
    print(f"Odom: x={x:.3f}, y={y:.3f}, theta={np.degrees(theta):.1f}°")
    
    # Compare to ground truth for debugging
    pos, orn = p.getBasePositionAndOrientation(robot_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]
    print(f"True: x={pos[0]:.3f}, y={pos[1]:.3f}, theta={np.degrees(true_theta):.1f}°")
"""