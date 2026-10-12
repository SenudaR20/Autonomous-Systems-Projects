import pybullet as p 
import random as r
import pybullet_data 
import time 
import numpy as np
 
class DemoRobot: 
    def __init__(self, x=0, y=0): 
        body_length = 1.2 
        body_width = 0.5 
        body_height = 0.3 
        wheel_radius = 0.15 
        wheel_width = 0.1 
 
        chassis_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[body_length/2, body_width/2, body_height/2]) 
        chassis_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[body_length/2, body_width/2, body_height/2], 
                                        rgbaColor=[0.2, 0.4, 0.8, 1.0]) 
 
        wheel_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=wheel_radius, height=wheel_width) 
        wheel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=wheel_radius, length=wheel_width, 
                                         rgbaColor=[0.3, 0.3, 0.3, 1.0]) 
 
        wheel_positions = [ 
            [body_length/3, body_width/2 + wheel_width/2, -wheel_width/2],    # 0: Front right 
            [body_length/3, -body_width/2 - wheel_width/2, -wheel_width/2],   # 1: Front left 
            [-body_length/3, body_width/2 + wheel_width/2, -wheel_width/2],   # 2: Rear right 
            [-body_length/3, -body_width/2 - wheel_width/2, -wheel_width/2],  # 3: Rear left 
        ] 
 
        link_masses = [2.0] * 4 
        link_collision_shapes = [wheel_collision] * 4 
        link_visual_shapes = [wheel_visual] * 4 
        link_positions = wheel_positions 
        link_orientations = [p.getQuaternionFromEuler([1.57, 0, 0])] * 4 
        link_inertial_frame_positions = [[0, 0, 0]] * 4 
        link_inertial_frame_orientations = [[0, 0, 0, 1]] * 4 
        link_parent_indices = [0] * 4 
        link_joint_types = [p.JOINT_REVOLUTE] * 4 
        link_joint_axis = [[0, 0, 1]] * 4 
 
        self.agv_id = p.createMultiBody( 
            baseMass=50, 
            baseCollisionShapeIndex=chassis_collision, 
            baseVisualShapeIndex=chassis_visual, 
            basePosition=[x, y, wheel_radius + body_height/2], 
            linkMasses=link_masses, 
            linkCollisionShapeIndices=link_collision_shapes, 
            linkVisualShapeIndices=link_visual_shapes, 
            linkPositions=link_positions, 
            linkOrientations=link_orientations, 
            linkInertialFramePositions=link_inertial_frame_positions, 
            linkInertialFrameOrientations=link_inertial_frame_orientations, 
            linkParentIndices=link_parent_indices, 
            linkJointTypes=link_joint_types, 
            linkJointAxis=link_joint_axis 
        ) 
 
        self.num_joints = p.getNumJoints(self.agv_id) 
        self.wheel_joints = list(range(self.num_joints)) 
 
        # Right-side wheels vs left-side wheels, based on wheel_positions above 
        self.right_wheels = [0, 2]   # front right, rear right 
        self.left_wheels = [1, 3]    # front left, rear left 
 
        for joint_id in self.wheel_joints: 
            p.setJointMotorControl2( 
                bodyUniqueId=self.agv_id, 
                jointIndex=joint_id, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=0, 
                force=0 
            ) 
            p.changeDynamics(self.agv_id, joint_id, lateralFriction=1.0, rollingFriction=0.01) 
 
        self.left_speed = 0 
        self.right_speed = 0 
        self.step = 0 
 
        # Tunable control parameters 
        self.max_speed = 8.0      # rad/s at the wheel for forward/back 
        self.turn_speed = 4.0    # rad/s differential added for turning 
        self.wheel_force = 50.0   # max motor torque 

        # LiDAR parameters
        self.num_lidar_beams = 21
        self.camera_fov_min = -np.pi / 2
        self.camera_fov_max = np.pi / 2
        self.lidar_z_offset = body_height / 2 + 0.01
        self.lidar_stop_target = None
        self.lidar_stop_distance = 1.5

    def set_lidar_stop_target(self,body_id,stop_distance=1.5):
        self.lidar_stop_target = body_id
        self.lidar_stop_distance = stop_distance

    def ray_cast_lidar(self,return_hit_ids=False):

      max_range=10
      # we need the real position, since the sensor will emulate where it really is

      robot_pos,robot_orn=p.getBasePositionAndOrientation(self.agv_id)  
      # Convert orientation to angles
      true_theta = p.getEulerFromQuaternion(robot_orn)[2]
      angles = np.linspace(true_theta+self.camera_fov_min, true_theta+self.camera_fov_max, self.num_lidar_beams)
      
      ray_from = []
      ray_to = []
    
      for angle in angles:
        # Ray direction in robot frame
        direction = np.array([np.cos(angle), np.sin(angle), 0])
        
        ray_from.append([
            robot_pos[0],
            robot_pos[1],
            robot_pos[2] + self.lidar_z_offset,
        ])
        ray_to.append([
            robot_pos[0] + direction[0] * max_range,
            robot_pos[1] + direction[1] * max_range,
            robot_pos[2] + self.lidar_z_offset,
        ])
    
      # Batch ray test (much faster than individual rays)
      results = p.rayTestBatch(ray_from, ray_to)
    
      distances = []
      #hit_positions = []
    
      for i, result in enumerate(results):
        hit_object_id = result[0]
        hit_fraction = result[2]  # Fraction of ray length where hit occurred
        #hit_position = result[3]
        
        if hit_object_id >= 0:  # Hit something
            distance = hit_fraction * max_range
        else:  # No hit
            distance = max_range
        
        distances.append(distance)
        #hit_positions.append(hit_position)
    
      if return_hit_ids:
          return np.array(distances), angles, np.array([result[0] for result in results])
      return np.array(distances), angles
    
 
    def user_control(self, keys): 
        """Reads WASD from a pybullet.getKeyboardEvents() dict and updates 
        self.left_speed / self.right_speed accordingly.""" 
 
        forward = 0 
        turn = 0 
 
        if ord('w') in keys and (keys[ord('w')] & p.KEY_IS_DOWN): 
            forward += 1 
        if ord('s') in keys and (keys[ord('s')] & p.KEY_IS_DOWN): 
            forward -= 1 
        if ord('a') in keys and (keys[ord('a')] & p.KEY_IS_DOWN): 
            turn += 1 
        if ord('d') in keys and (keys[ord('d')] & p.KEY_IS_DOWN): 
            turn -= 1 
 
        self.left_speed = -self.max_speed * forward - self.turn_speed * turn
        self.right_speed = -self.max_speed * forward + self.turn_speed * turn
 
    def fix_actions(self): 
        """Follow the route from the start to the centre chamber."""

        if self.step < 720:
        # Turn left
            self.left_speed = -8.0
            self.right_speed = 8.0

        elif self.step < 1050:
            # Go straight
            self.left_speed = -8.0
            self.right_speed = -8.0

        elif self.step < 1680:
            # Turn right
            self.left_speed = 8.0
            self.right_speed = -8.0

        elif self.step < 5550:
            # Go straight across maze
            self.left_speed = -8.0
            self.right_speed = -8.0

        elif self.step < 6380:
            # Turn left
            self.left_speed = -8.0
            self.right_speed = 8.0

        elif self.step < 6500:
            # Go straight down
            self.left_speed = -8.0
            self.right_speed = -8.0

        elif self.step < 6550:
            # stop
            self.left_speed = 0
            self.right_speed = 0

        elif self.step < 7490:
            # Turn left
            self.left_speed = -8.0
            self.right_speed = 8.0

        elif self.step < 8800:
            # Go straight into centre
            self.left_speed = -8.0
            self.right_speed = -8.0

        else:
            # Stop
            self.left_speed = 0
            self.right_speed = 0
 
 
    def step_action(self): 
        """Applies self.left_speed / self.right_speed to the wheel motors. 
        Call this once per simulation step, after user_control().""" 

        p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME,0) # disables wireframe
        # self.fix_actions() 

        """ lIDAR stop detection """
        # distances, angles, hit_object_ids = self.ray_cast_lidar(return_hit_ids=True)
        # person_detected = any(
        #     hit_id == self.lidar_stop_target and distance <= self.lidar_stop_distance
        #     for hit_id, distance in zip(hit_object_ids, distances)
        # )
        # if person_detected:
        #     self.left_speed = 0
        #     self.right_speed = 0

        # if self.step % 100 == 0:
        #     print("LiDAR:", np.round(distances, 2))

        """ Motor control with optional noise """
        # noisy_left_speed = r.uniform(0.98, 1.02) * self.left_speed
        # noisy_right_speed = r.uniform(0.98, 1.02) * self.right_speed
        
        # if self.step % 100 == 0:
        #     print(
        #         "step =", self.step,
        #         "left =", self.left_speed,
        #         "right =", self.right_speed,
        #         #  "noisy left =", noisy_left_speed,
        #         #  "noisy right =", noisy_right_speed,
        #     )



        
        for j in self.right_wheels: 
            p.setJointMotorControl2( 
                bodyUniqueId=self.agv_id, 
                jointIndex=j, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=self.right_speed, 
                # targetVelocity=noisy_right_speed,
                force=self.wheel_force 
            ) 
        for j in self.left_wheels: 
            p.setJointMotorControl2( 
                bodyUniqueId=self.agv_id, 
                jointIndex=j, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=self.left_speed, 
                # targetVelocity=noisy_left_speed,
                force=self.wheel_force 
            ) 
         
        # if not person_detected:
        #     self.step += 1
 
     
 
 
 
 
 
# if __name__ == "__main__": 
#     p.connect(p.GUI) 
#     p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
#     p.setGravity(0, 0, -9.81) 
#     p.loadURDF("plane.urdf") 
 
#     robot = DemoRobot(0, 0) 
 
#     while True: 
#         robot.step_action() 
 
#         p.stepSimulation() 
#         time.sleep(1.0 / 240.0) 

