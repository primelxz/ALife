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

# Load the model and create a data instance
model = dm_control.mujoco.MjModel.from_xml_path("spider.xml")
data = dm_control.mujoco.MjData(model)

# Control parameters
angle_range = 70  # Degrees
flip_frequency = 2  # Flips per second
total_steps = 1000  # Total number of steps for the simulation
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
                data.ctrl[i] = angle * -10
            else:
                data.ctrl[i] = angle * 10

        # Step the simulation
        dm_control.mujoco.mj_step(model, data)

        # Sync the viewer with the current state of the simulation
        viewer.sync()

        # Control loop rate (optional, for smoother visualization)
        time.sleep(timestep)
