import dm_control.mujoco
import mujoco.viewer
import numpy as np
import time
import random
import xml.etree.ElementTree as ET
import csv


def indent(elem, level=0):
    """Function to add whitespace to the tree, making it readable."""
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for e in elem:
            indent(e, level+1)
        if not e.tail or not e.tail.strip():
            e.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def random_generation(random_addition, identity_idx, body_mass, leg_length, leg_mass):
    tree = ET.parse('spider.xml')
    root = tree.getroot()
    unit = root.find('.//worldbody')

    actuators = root.find('.//actuator')
    if actuators is None:
        actuators = ET.SubElement(root, 'actuator')

    for i in range(0, random_addition):
        unit_name = f"unit{i}"
        body_joint_name = f"body_joint{i}"
        body_name = f"body{i}"

        unit_body = ET.SubElement(unit, 'body', name=unit_name, pos=f"0 0 0", euler="0 0 0")
        if i == 0:
            ET.SubElement(unit_body, 'joint', name=body_joint_name, type='free', axis='0 0 0', pos=f'0 0 0',
                          range='-20 20')
        else:
            ET.SubElement(unit_body, 'joint', name=body_joint_name, type='hinge', axis='0 0 1', pos=f'0 .35 0',
                          range='-20 20')

        body = ET.SubElement(unit_body, 'body', name=body_name, pos=f'0 {.6 * i} 0', euler='0 0 0')
        ET.SubElement(body, 'geom', type='box', size='.2 .3 .1', rgba='0 .9 0 1', mass=str(body_mass))

        for j in range(1, 3):  # Assuming each unit has 2 legs
            leg_name = f"{body_name}leg{j}"
            joint_name = f"{body_name}j{j}"
            leg_pos = f".4 {.6 * i} 0" if j == 1 else f"-.4 {.6 * i} 0"
            euler = "0 -90 90" if j == 1 else "0 90 -90"
            leg_body = ET.SubElement(unit_body, 'body', name=leg_name, pos=leg_pos, euler=euler)
            ET.SubElement(leg_body, 'joint', name=joint_name, type='hinge', axis='0 1 0', pos='0 0 .25', range='-30 30')
            ET.SubElement(leg_body, 'geom', type='box', size=f'.05 .05 {leg_length}', rgba='0 .9 0 1', mass=str(leg_mass))

            # Add actuators for each leg
            ET.SubElement(actuators, 'motor', joint=joint_name, name=f"m_{joint_name}", gear='5', ctrllimited='true',
                          ctrlrange='-10 10')

        if i != 0:
            ET.SubElement(actuators, 'motor', joint=body_joint_name, name=f'm_{body_joint_name}', gear='3',
                          ctrllimited='true', ctrlrange='-5 5')

        unit = unit.find(f'.//body[@name="{unit_name}"]')

    indent(root)
    tree.write(f'spider_{identity_idx}.xml')


def evaluate_fitness(initial_position, data, total_steps):
    distance_traveled = abs(data.qpos[0] - initial_position[0])
    stability_measure = np.sum(data.qvel**2)
    penalty_coef = 0.01
    fitness = distance_traveled - (penalty_coef * np.sqrt(stability_measure)) / (100 * total_steps)
    return fitness


def simulation(random_addition, filename):
    # Load the model and create a data instance
    model = dm_control.mujoco.MjModel.from_xml_path(filename)
    data = dm_control.mujoco.MjData(model)

    # Control parameters
    angle_range = 70  # Degrees
    flip_frequency = 1  # Flips per second
    total_steps = 500  # Total number of steps for the simulation
    timestep = 0.01  # Time step for simulation

    # Convert angle to radians
    angle_rad = np.deg2rad(angle_range)

    initial_position = np.copy(data.qpos)

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

            data.ctrl[0] = angle * -20
            data.ctrl[1] = angle * 20
            # Apply the angle to each joint
            for i in range(2, random_addition * 3 - 1):
                if i % 3 == 2:
                    data.ctrl[i] = angle * -20
                elif i % 3 == 0:
                    data.ctrl[i] = angle * 20
                else:
                    data.ctrl[i] = angle * random.randint(-10, 10)

            # Step the simulation
            dm_control.mujoco.mj_step(model, data)

            # Sync the viewer with the current state of the simulation
            viewer.sync()

            # Control loop rate (optional, for smoother visualization)
            time.sleep(timestep)

        fitness = evaluate_fitness(initial_position, data, total_steps)
        return fitness


num_individuals = 10
fitness_list = []
for i in range(num_individuals):
    random_num = random.randint(1, 5)
    body_mass = random.uniform(0.5, 2.0)
    leg_length = random.uniform(0.1, 0.25)
    leg_mass = random.uniform(0.5, 1)
    xml_file = f'spider_{i}.xml'

    random_generation(random_num, i, body_mass, leg_length, leg_mass)
    fitness = simulation(random_num, xml_file)
    identity_features = {'name': xml_file[:-4],
                         'body_length': random_num,
                         'body_mass': body_mass,
                         'leg_length': leg_length,
                         'leg_mass': leg_mass,
                         'fitness': fitness}
    fitness_list.append(identity_features)

csv_file_name = 'spider_fitness.csv'
fieldnames = ['name', 'body_length', 'body_mass', 'leg_length', 'leg_mass', 'fitness']

with open(csv_file_name, mode='w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    for fitness_data in fitness_list:
        writer.writerow(fitness_data)

print(f"CSV file '{csv_file_name}' has been created with fitness data.")

