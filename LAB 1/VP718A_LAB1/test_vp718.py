import utils.world as W
import utils.demo_robot as R
import pybullet as p

# create the world object, which will set up most of pybullet
wld=W.BaseWorld()
# load a map
wld.map_from_file("maps/demoMap")
# you do not have to do this in this way, but I created a method to add textures to the walls
# be warned it interacts with the way I did the map and the storage of the map, so it only
# impacts walls of type 1
wld.texture_walls() 
# create a robot, positioned at 0,0   
r=R.DemoRobot(x=0,y=0)     # 9,9

def camera_over_rover():
    pos, _ = p.getBasePositionAndOrientation(r.agv_id)

    p.resetDebugVisualizerCamera(
        cameraDistance=10,
        cameraYaw=0,
        cameraPitch=-89.9,
        cameraTargetPosition=pos,
        physicsClientId=wld.client
    )

# my world object has call backs for the step function and the keyboard
# I add only the robots step function here, but you could have 2 robots,
# or other special objects
wld.add_step_callback(r.step_action)
wld.add_keyboard_callback(r.user_control)
wld.add_step_callback(camera_over_rover)

camera_over_rover()

# Run simulation
while True:
    wld.simStep()

# Disconnect
wld.end()