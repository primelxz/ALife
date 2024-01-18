# from dm_control import mujoco
# from dm_control import viewer

# # Load the model from the XML file
# model = mujoco.Physics.from_xml_path('C:/Users/prime/study/Winter_2024/CS396/hws/setup/test.xml')

# # Instantiate a viewer for the simulation
# viewer.launch(model)

from dm_control import mujoco
from dm_control import viewer
from dm_control.suite import base

# Define an environment loader function
def environment_loader():
    xml_path = 'C:/Users/prime/study/Winter_2024/CS396/hws/setup/test.xml'
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = base.Task()  # Define a task (if necessary, you can create a custom one)
    environment = base.Environment(physics, task, time_limit=None)
    return environment

# Pass the environment loader to the viewer
viewer.launch(environment_loader=environment_loader)
