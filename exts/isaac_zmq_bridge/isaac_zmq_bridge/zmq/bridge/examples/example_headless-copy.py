# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# to run this file:
# ISAACSIM_PYTHON exts/isaac_zmq_bridge/isaacsim/zmq/bridge/examples/example_headless.py --ext-folder ./exts

import isaacsim
from isaacsim.simulation_app import SimulationApp
from omni.isaac.core.utils.extensions import enable_extension

# Set headless mode to True for GUI enabled.
simulation_app = SimulationApp({"headless": True})

import carb
import omni.kit.app

# Get the extension manager and enable our extension
# manager = omni.kit.app.get_app().get_extension_manager()
# manager.set_extension_enabled_immediate("isaac_zmq_bridge", True)
enable_extension("isaac_zmq_bridge")
simulation_app.update()


from isaac_zmq_bridge import EXT_NAME
from isaac_zmq_bridge.example_missions import (
    FrankaMultiVisionMission,
    FrankaVisionMission,
)

# select an example mission here
mission = FrankaVisionMission()
# mission = FrankaMultiVisionMission()  # Uncomment to use this mission instead

# Load the mission USD file and set up the scene
mission.load_mission()
mission.reset_world()

# Warm up the simulation with a few steps
# This helps ensure caputured images wont have artifacts
for i in range(100):
    simulation_app.update()
    print(f"[{EXT_NAME}] Warm up step {i+1}/20")

# Start the mission
# This will start the world simulation.
mission.start_mission()
print(f"[{EXT_NAME}] Streaming data...")


while True:
    simulation_app.update()  # run forever

# This line is never reached,
# Left to demonstrate how to gracefully close the app
simulation_app.close()
