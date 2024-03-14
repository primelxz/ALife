# Spider Simulation
## Overview

Briefly describe the purpose and functionality of your project. Mention that this project simulates the movement and evolution of a robotic spider in a virtual environment, focusing on genetic algorithms, robotic control, and physical simulation.
Final Work Location

All final work for this project, including code, simulation results, and documentation, is located in the hw4 directory.
Requirements

    MuJoCo
    dm_control
    Python 3.x
    NumPy
    Random
    Time

## Usage
Install dm_control and numpy using pip
```
pip install dm_control numpy
```
Navigate to the hw4 directory
```
cd hw4
```
Run the simulation script
```
python spider.py
```
## Implementation Details

- **XML Configuration**: Automatic generation of MuJoCo XML configurations to define the spider's physical structure.
- **Simulation**: Utilization of the MuJoCo engine for simulating the spider's movement.
- **Genetic Algorithm**: Implementation of genetic algorithms for simulating evolution through mutation and fitness evaluation.
- **Generational Evolution**: Simulation of multiple generations, with selection, mutation, and evaluation of fitness.
