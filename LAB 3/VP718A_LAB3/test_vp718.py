import utils.world as W
import utils.demo_robot as R
import utils.demo_person as P

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

# my world object has call backs for the step function and the keyboard
# I add only the robots step function here, but you could have 2 robots,
# or other special objects
wld.add_step_callback(r.step_action)
# wld.add_keyboard_callback(r.user_control)

# create a person, psoition at 4,3, z=0.81
person=P.DemoPerson(x=4, y=3, z=0.81)
person.set_y_sweep(3, 5)
person.ignore_collisions_with(wld.planeId)
person.add_collidable(r.agv_id)
r.set_lidar_stop_target(person.objectId)


def myReaction():
    print("ROBOT HIT")

person.set_collision_callback(myReaction)

wld.add_step_callback(person.step_action)

# Run simulation
while True:
    wld.simStep()

# Disconnect
wld.end()