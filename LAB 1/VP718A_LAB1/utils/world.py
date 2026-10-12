import pybullet as p
import pybullet_data
import time
import math

import os
os.environ['B3_NO_PROFILE'] = '1'
os.environ['BT_DISABLE_PROFILE'] = '1'
os.environ['BT_NO_PROFILE'] = '1'
os.environ['BULLET_NO_PROFILE'] = '1'
os.environ['B3_NO_PROFILE'] = '1'

def get_forward_vector(yaw,pitch):
    yaw_rad = math.radians(yaw+90)
    pitch_rad = math.radians(pitch)
    return  [
            math.cos(pitch_rad) * math.cos(yaw_rad),
            math.cos(pitch_rad) * math.sin(yaw_rad), 
            math.sin(pitch_rad)
        ]
        
def get_right_vector(yaw,pitch):
    """Get right direction from current camera"""
    yaw_rad = math.radians(yaw)
    return [
            math.cos(yaw_rad),
            math.sin(yaw_rad),
            0
        ]
        

class BaseWorld:
    def __init__(self):
        print("PyBullet Version ",p.getAPIVersion())
        # Connect to the physics server
        self.client = p.connect(p.GUI,options="--disable_timer --disable_file_caching")  # Use p.DIRECT for headless mode
        
        # p.setPhysicsEngineParameter(enableInternalProfiling=0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Set gravity
        p.setGravity(0, 0, -9.81)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)  # Enable GUI panels
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)  # Enable keyboard shortcuts

        # Load a plane (ground)
        self.planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(self.planeId , -1, lateralFriction=0.5)
        
        
        self.move_speed=0.02

        self.cuboids=[]

        self.step_callbacks=[]

        self.additional_key_callbacks=[]

        p.resetDebugVisualizerCamera(
            cameraDistance=3,
            cameraYaw=0,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 8]
        )

    def add_keyboard_callback(self,x):
        self.additional_key_callbacks.append(x)

    def add_step_callback(self,x):
        self.step_callbacks.append(x)

    def add_cuboid(self,dims,position,color,mass=0):
        """Expects 3 tuples, 
                (length, width, height)
                (x,y,z)
                (r,g,b)  # 0,1 range, not 0-255"""
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=dims, rgbaColor=color + [1.0])
            
        wall_id = p.createMultiBody(
                baseMass=mass,  # Static
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=position
            )
        if mass==0:
            self.cuboids.append(wall_id)
        return wall_id

    def map_from_file(self,fname,unitwidth=1.0):
        xs=open(fname,"r").read().strip()
        row=0
        for l in xs.split("\n"):
            col=0
            for c in l.strip():
                y=row*unitwidth+unitwidth/2
                x=col*unitwidth+unitwidth/2
                if c=="1":
                    self.add_cuboid((unitwidth/2,unitwidth/2,unitwidth/2),(x,y,unitwidth/2),[0.7,0.7,0.7])
                if c=="2":
                    self.add_cuboid((unitwidth/2,unitwidth/2,unitwidth/2),(x,y,unitwidth/2),[0.7,0.7,0.7],mass=0.5)
                col+=1
            row+=1

    def texture_walls(self):
        texture_id = p.loadTexture("textures/silly.png")
        for cube_id in self.cuboids:
            # Apply the texture
            p.changeVisualShape(cube_id, -1, textureUniqueId=texture_id)

    def camera_movement(self):
        keys = p.getKeyboardEvents()
        cam_info = p.getDebugVisualizerCamera()
        yaw = cam_info[8]      # Horizontal rotation (degrees)
        pitch = cam_info[9]    # Vertical rotation (degrees)
        
        cameraPosition = list(cam_info[11])   # get current target, since we expect to move it
        
        moved=False

        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
            forward=get_forward_vector(yaw,pitch)
            cameraPosition[0] += forward[0] * self.move_speed
            cameraPosition[1] += forward[1] * self.move_speed
            cameraPosition[2] += forward[2] * self.move_speed
            moved = True
            
        # DOWN ARROW - Move backward (same as S)
        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
            forward=get_forward_vector(yaw,pitch)
            cameraPosition[0] -= forward[0] * self.move_speed
            cameraPosition[1] -= forward[1] * self.move_speed
            cameraPosition[2] -= forward[2] * self.move_speed
            moved = True
            
        # LEFT ARROW - Strafe left (same as A)
        if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
            right=get_right_vector(yaw,pitch)
            cameraPosition[0] -= right[0] * self.move_speed
            cameraPosition[1] -= right[1] * self.move_speed
            moved = True
            
        # RIGHT ARROW - Strafe right (same as D)
        if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
            right=get_right_vector(yaw,pitch)
            cameraPosition[0] += right[0] * self.move_speed
            cameraPosition[1] += right[1] * self.move_speed
            moved = True

        if moved:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.0,
                cameraYaw=yaw,
                cameraPitch=pitch, 
                cameraTargetPosition=cameraPosition
            )
        
        for i in self.additional_key_callbacks:
            i(keys)

    def simStep(self):
        for i in self.step_callbacks:
            i()
        self.camera_movement()
        p.stepSimulation()
                
        time.sleep(1./240.)  # 240 Hz

    def end(self):
        self.client.disconnect()
        p.disconnect()
