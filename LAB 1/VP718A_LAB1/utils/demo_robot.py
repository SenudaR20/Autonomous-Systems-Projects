import pybullet as p 
import random as r
import pybullet_data 
import time 
 
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
        self.turn_speed = 4.0     # rad/s differential added for turning 
        self.wheel_force = 50.0   # max motor torque 
 
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
 
        self.left_speed = self.max_speed * forward + self.turn_speed * turn 
        self.right_speed = self.max_speed * forward - self.turn_speed * turn 
 

 
 
    def step_action(self): 
        """Applies self.left_speed / self.right_speed to the wheel motors. 
        Call this once per simulation step, after user_control().""" 
 
        for j in self.right_wheels: 
            p.setJointMotorControl2( 
                bodyUniqueId=self.agv_id, 
                jointIndex=j, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=self.right_speed, 
                force=self.wheel_force 
            ) 
        for j in self.left_wheels: 
            p.setJointMotorControl2( 
                bodyUniqueId=self.agv_id, 
                jointIndex=j, 
                controlMode=p.VELOCITY_CONTROL, 
                targetVelocity=self.left_speed, 
                force=self.wheel_force 
            ) 
         
        self.step += 1 
 
     
 
 
 
 
 
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

