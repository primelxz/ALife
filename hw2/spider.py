# from dm_control import mujoco
# from dm_control import viewer

# # Load the model from the XML file
# model = mujoco.Physics.from_xml_path('C:/Users/prime/study/Winter_2024/CS396/hws/setup/test.xml')

# # Instantiate a viewer for the simulation
# viewer.launch(model)

# from dm_control import mujoco
# from dm_control import viewer
# from dm_control.suite import base
#
# # Define an environment loader function
# def environment_loader():
#     xml_path = 'C:/Users/prime/study/Winter_2024/CS496/hws/hw1/hw1.xml'
#     physics = mujoco.Physics.from_xml_path(xml_path)
#     task = base.Task()  # Define a task (if necessary, you can create a custom one)
#     environment = base.Environment(physics, task, time_limit=None)
#     return environment
#
# # Pass the environment loader to the viewer
# viewer.launch(environment_loader=environment_loader)

import dm_control.mujoco
import mujoco.viewer
import numpy as np
import time
import random
import xml.etree.ElementTree as ET

tree = ET.parse('spider.xml')
root = tree.getroot()

actuators = root.find('.//actuator')
if actuators is None:
    actuators = ET.SubElement(root, 'actuator')

random_addition = random.randint(0, 2)
for i in range(1, 9):
    leg_name = f'leg{i}'
    leg = root.find(f'.//body[@name="{leg_name}"]')
    if leg is not None:

        for j in range(random_addition):
            segment_name = f'{leg_name}_segment{j}'
            joint_name = f'j_{segment_name}'

            sub_leg = ET.SubElement(leg, 'body', name=segment_name, pos=f'0 0 {1.5 + j * 0.2}', euler='0 0 0')
            ET.SubElement(sub_leg, 'joint', name=joint_name, type='hinge', axis='0 1 0', pos='0 0 -.25', range='-10 10')
            ET.SubElement(sub_leg, 'geom', type='box', size='.05 .05 .2', rgba='0 .9 0 1', mass='1')

            ET.SubElement(actuators, 'motor', joint=joint_name, name=f'm_{segment_name}', gear='10', ctrllimited='true', ctrlrange='-10 10')

tree.write('spider_copy.xml')

# Load the model and create a data instance
model = dm_control.mujoco.MjModel.from_xml_path("spider_copy.xml")
data = dm_control.mujoco.MjData(model)

# Control parameters
angle_range = 70  # Degrees
flip_frequency = 1  # Flips per second
total_steps = 2000  # Total number of steps for the simulation
timestep = 0.01  # Time step for simulation

# Convert angle to radians
angle_rad = np.deg2rad(angle_range)

# Open the viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 180
    viewer.cam.elevation = -20
    viewer.cam.distance = 3.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.75]

    # Simulation loop
    for step in range(total_steps):
        # Calculate the desired angle for this timestep
        angle = angle_rad * np.sin(2 * np.pi * flip_frequency * step * timestep)

        # Apply the angle to each joint
        for i in range(8):  # Assuming 8 legs
            if i % 2 == 0:
                data.ctrl[i] = angle * -20
                for j in range(random_addition):
                    data.ctrl[i * random_addition + 8 + j] = data.ctrl[i] / (j + 1)
            else:
                data.ctrl[i] = angle * 20
                for j in range(random_addition):
                    data.ctrl[i * random_addition + 8 + j] = data.ctrl[i] / (j + 1)

        # Step the simulation
        dm_control.mujoco.mj_step(model, data)

        # Sync the viewer with the current state of the simulation
        viewer.sync()

        # Control loop rate (optional, for smoother visualization)
        time.sleep(timestep)
