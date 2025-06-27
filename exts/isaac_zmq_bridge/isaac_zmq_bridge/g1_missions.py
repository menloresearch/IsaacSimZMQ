# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import asyncio
from collections import deque
from datetime import timedelta, datetime
from heapq import heappush, heappop
import time

import carb
import numpy as np
import omni.usd

# from omni.isaac.core.utils.extensions import enable_extension
# enable_extension("omni.isaac.robot_assembler")
from isaacsim.robot_setup.assembler import RobotAssembler
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.storage.native import get_assets_root_path
from isaacsim.util.debug_draw import _debug_draw
from pxr import Gf, Sdf, Tf, Usd, UsdGeom, UsdPhysics, UsdShade

from . import EXT_NAME, G1Annotator, G1StateConvert
from .mission import Mission

# The omni.__proto__ namespace is created by this extention
# read more at coreproto_util.py
from omni.__proto__ import server_control_message_pb2
from omni.isaac.core.articulations import ArticulationView


class ActionQueue:

    def __init__(self, limit=128):
        self._que = []
        self.limit = limit

    def push(self, time, item):
        if len(self._que) < self.limit:
            heappush(self._que, (time, item))

    @property
    def empty(self):
        return len(self._que) == 0

    def pop_top(self,):
        return heappop(self._que)


class G1StackBlockMission(Mission):
    """Mission that demonstrates a Franka robot with vision capabilities.

    This mission sets up a Franka robot with a camera and enables control via ZMQ.
    It streams camera data and allows external control of the robot's end effector.
    """

    name = "G1StackBlockMission"
    world_usd_path = "g1_world.usda"

    def __init__(self, server_ip: str = "localhost"):
        Mission.__init__(self, server_ip=server_ip)
        self.server_ip = server_ip

        # Scene setup parameters
        self.scene_root = "/World"
        self._camera_path = None
        self._camera_prim = None

        self.draw = _debug_draw.acquire_debug_draw_interface()

        self.cur_focal_length = 10

        # Simulation parameters
        self.physics_dt = 60.0  # Rate of the physics simulation
        self.camera_hz = 60.0  # Do not go above physics_dt!
        self.dimension = (480, 640)

        # Camera setup
        self.camera_annotator = None
        self.camera_annotators = []

        self.use_ogn_nodes = False  # True > use OGN C++ node, False > use Python

        # Target position randomization
        self.last_trigger_time = 0

        self.last_action_time = datetime.now()
        self.action_que = ActionQueue(limit=128)
        _seed = 1234
        self.rng = np.random.default_rng(_seed)

    def start_mission(self) -> None:
        """Start the mission by setting up ZMQ communication and camera streaming.

        This method initializes ZMQ sockets, sets up the camera annotator, and starts
        the command reception loops for various control channels.
        """
        # Define communication ports for different data streams
        self.ports = {
            "camera_annotator": 5561,
            "camera_control_command": 5557,
            "settings": 5559,
            "franka": 5560,
        }

        # Set up ZMQ sockets for receiving commands
        self.camera_control_socket = self.zmq_client.get_pull_socket(self.ports["camera_control_command"])
        self.settings_socket = self.zmq_client.get_pull_socket(self.ports["settings"])
        self.franka_socket = self.zmq_client.get_pull_socket(self.ports["franka"])

        # Set up camera for streaming
        self._camera_path = "/World/base_link/y_link/Camera"
        stage = omni.usd.get_context().get_stage()
        self._camera_prim = stage.GetPrimAtPath(self._camera_path)

        # Enable command reception
        self.receive_commands = True

        # Create camera annotator for streaming camera data
        self.camera_annotator = G1Annotator(
            self._camera_path,
            (self.dimension[1], self.dimension[0],),
            self.franka,
            self.action_que,
            use_ogn_nodes=self.use_ogn_nodes,
            server_ip=self.server_ip,
            port=self.ports["camera_annotator"],
            bbox=False,
        )
        self.camera_annotators.append(self.camera_annotator)

        # If not using OGN nodes, set up Python-based streaming
        if not self.use_ogn_nodes:
            print(f"[{EXT_NAME}] Using Python-based streaming")
            self.camera_annot_sock_pub = self.zmq_client.get_push_socket(self.ports["camera_annotator"])
            self.camera_annotator.sock = self.camera_annot_sock_pub
            self.zmq_client.add_physx_step_callback(
                "camera_annotator", 1 / self.camera_hz, self.camera_annotator.stream
            )

        # Set up async receive loops for all command channels
        self.subscribe_to_protobuf_in_loop(
            self.settings_socket,
            server_control_message_pb2.ServerControlMessage,
            self.settings_sub_loop,
        )
        self.subscribe_to_protobuf_in_loop(
            self.franka_socket,
            server_control_message_pb2.ServerControlMessage,
            self.franka_sub_loop
        )

    def update_annotator(self):
        for anno in self.camera_annotators:
            if isinstance(anno, G1Annotator):
                anno.g1 = self.franka

    def stop_mission(self) -> None:
        """Stop the mission and clean up resources.

        This method stops the simulation, disconnects ZMQ sockets, and destroys annotators.
        """

        async def _stop() -> None:
            await self.world.stop_async()
            self.receive_commands = False
            self.zmq_client.remove_physx_callbacks()
            # must wait for all callbacks to finish before disconnecting from the server
            await asyncio.sleep(0.5)
            await self.zmq_client.disconnect_all()
            # must wait for all client to disconnect before destroying the annotators
            await asyncio.sleep(0.5)
            if self.world.is_stopped():
                for annotator in self.camera_annotators:
                    annotator.destroy()
            else:
                carb.log_warn(f"[{EXT_NAME}] Cant destory annotators while simulation is running!")

        asyncio.ensure_future(_stop())

    def settings_sub_loop(self, proto_msg: server_control_message_pb2.ServerControlMessage) -> None:
        """General purpose control loop to tweak parameters of the simulator

        Args:
            proto_msg: ServerControlMessage containing a ControlCommand
        """
        if proto_msg.HasField("settings_command"):
            self.zmq_client.adaptive_rate = proto_msg.settings_command.adaptive_rate

    def franka_sub_loop(self, proto_msg: server_control_message_pb2.ServerControlMessage) -> None:
        """Handle Franka robot commands received via ZMQ.

        Controls the Franka robot's end effector position using RMPFlow and
        randomizes the target position every 5 seconds.

        joint_names:  [
            'left_hip_pitch_joint', 'right_hip_pitch_joint',             0~1
            'waist_yaw_joint',                                           2
            'left_hip_roll_joint', 'right_hip_roll_joint',               3~4
            'waist_roll_joint',                                          5
            'left_hip_yaw_joint', 'right_hip_yaw_joint',                 6~7
            'waist_pitch_joint',                                         8
            'left_knee_joint', 'right_knee_joint',                       9~10
            'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint',   11~12
            'left_ankle_pitch_joint', 'right_ankle_pitch_joint',         13~14
            'left_shoulder_roll_joint', 'right_shoulder_roll_joint',     15~16
            'left_ankle_roll_joint', 'right_ankle_roll_joint',           17~18
            'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',       19~20
            'left_elbow_joint', 'right_elbow_joint',                     21~22
            'left_wrist_roll_joint', 'right_wrist_roll_joint',           23~24
            'left_wrist_pitch_joint', 'right_wrist_pitch_joint',         25~26
            'left_wrist_yaw_joint', 'right_wrist_yaw_joint',             27~28
            'left_hand_index_0_joint', 'left_hand_middle_0_joint', 'left_hand_thumb_0_joint',    29~31
            'right_hand_index_0_joint', 'right_hand_middle_0_joint', 'right_hand_thumb_0_joint', 32~34
            'left_hand_index_1_joint', 'left_hand_middle_1_joint', 'left_hand_thumb_1_joint',    35~37
            'right_hand_index_1_joint', 'right_hand_middle_1_joint', 'right_hand_thumb_1_joint', 38~40
            'left_hand_thumb_2_joint', 'right_hand_thumb_2_joint'        41~42
        ]

        Args:
            proto_msg: ServerControlMessage containing a FrankaCommand
        """
        # print("joint_names: ", Articulation(prim_paths_expr='/World/G1').joint_names)
        self.draw.clear_points()

        if self.world.is_playing():

            if proto_msg.HasField("g1_command"):
                # effector_pos = proto_msg.franka_command.effector_pos
                # new_effector_pos = [effector_pos.x, effector_pos.y, effector_pos.z]
                cmd = proto_msg.g1_command
                print("cmd.left_shoulder_angle.y: ", cmd.left_shoulder_angle.y)
                print("cmd.left_shoulder_angle.z: ", cmd.left_shoulder_angle.z)
                joint_pos = G1StateConvert.cmd_to_isaac(cmd)
                self.action_que.push(proto_msg.sys_time, joint_pos)
            else:
                joint_pos = np.array([0] * 33 + [1.0] * 10)

            interval = datetime.now() - self.last_action_time
            if not self.action_que.empty:
                enque_t, target_pos = self.action_que.pop_top()
                try:
                    action = ArticulationAction(
                        joint_positions=target_pos,
                        joint_velocities=None,
                        joint_efforts=None
                    )
                    self.franka.apply_action(action)
                    self.last_action_time = datetime.now()
                except Exception as e:
                    self.action_que.push(enque_t, target_pos)
                    carb.log_warn(f"[{EXT_NAME}] Error applying action: {e}")

            self.check_and_repostion_blocks()

    def check_and_repostion_blocks(self):
        # randomize the target position every 8 seconds :)
        # current_time = time.time()
        # if current_time - self.last_trigger_time > 8:
        #     lower_bounds = np.array([0.2, -0.2, 0.1])
        #     upper_bounds = np.array([0.6, 0.2, 0.5])
        #     random_array = self.rng.uniform(lower_bounds, upper_bounds)
        #     self.target.set_world_poses(positions=np.array([random_array]))
        #     self.last_trigger_time = current_time
        pass

    def reset_franka_mission(self) -> None:
        """Reset the Franka robot and its controller."""
        self.franka = Robot(prim_path="/World/G1")
        self.franka.initialize()
        self.update_annotator()
        self.franka_articulation_controller = self.franka.get_articulation_controller()
        self.target = XFormPrim(prim_paths_expr="/World/Target_R")

    def before_reset_world(self) -> None:
        """Prepare the world for reset.

        This method is called before resetting the world to set up the camera robot.
        """
        self.draw.clear_points()
        self.camera_robot = Robot(prim_path=f"/World/base_link", name="robot")
        self.world.scene.add(self.camera_robot)

    def after_reset_world(self) -> None:
        """Execute operations after the world has been reset.

        This method is called after resetting the world to set up controllers and the Franka robot.
        """
        self.zmq_client.simulation_start_timecode = time.time()
        self.camera_controller = self.camera_robot.get_articulation_controller()
        self.meters_per_unit = self.world.scene.stage.GetMetadata(UsdGeom.Tokens.metersPerUnit)
        self.reset_franka_mission()

    @classmethod
    def add_robot(cls) -> None:
        """Add a Franka robot to the scene."""
        translate = [0.0, 0.0, 0.79]
        g1_gsd = str(cls.asset_dir_path() / "g1_29dof_with_hand_rev_1_0.usd")
        omni.kit.commands.execute(
            "CreateReferenceCommand",
            usd_context=omni.usd.get_context(),
            path_to="/World/G1",
            asset_path=g1_gsd,
            instanceable=False,
        )
        omni.kit.commands.execute(
            'ChangeProperty',
            prop_path=Sdf.Path("/World/G1" + '.xformOp:translate'),
            value=Gf.Vec3d(*translate),
            prev=Gf.Vec3d(0.0, 0.0, 0.0),  # Previous value (optional)
            # target_layer=omni.usd.get_context().get_stage(),
            usd_context_name=omni.usd.get_context()
        )
        omni.kit.selection.SelectNoneCommand().do()

        # NOTE: fix pelvis joint position, so G1 won't sway around during operation.
        robot_assembler = RobotAssembler()
        robot_assembler.create_fixed_joint(
            prim_path="/World/G1/pelvis",
            target1 ="/World/G1/pelvis",
            fixed_joint_offset=np.asarray(translate),
        )

    @classmethod
    async def _async_load(cls) -> None:
        """Load the mission asynchronously."""
        await Mission._async_load(cls.mission_usd_path())
        cls.add_robot()
        await asyncio.sleep(0.5)
        omni.kit.selection.SelectNoneCommand().do()

    @classmethod
    def load_mission(cls) -> None:
        """Load the mission synchronously."""
        Mission.load_mission(cls.mission_usd_path())
        cls.add_robot()
        omni.kit.selection.SelectNoneCommand().do()
