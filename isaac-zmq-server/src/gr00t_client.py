
import math
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Any, Callable, Dict, List

import torch
import zmq
import numpy as np
import client_stream_message_pb2
import server_control_message_pb2


MsgList = List["client_stream_message_pb2.G1JoinState"]

class TorchSerializer:
    @staticmethod
    def to_bytes(data: dict) -> bytes:
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        buffer = BytesIO(data)
        obj = torch.load(buffer, weights_only=False)
        return obj


class BaseInferenceClient:
    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        """Initialize or reinitialize the socket with current settings"""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """
        Kill the server.
        """
        self.call_endpoint("kill", requires_input=False)

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        """
        Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b"ERROR":
            raise RuntimeError("Server error")
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction"""
        self.socket.close()
        self.context.term()


class G1StateConvert:
    """
    Isaac Sim Join State <--> G1JoinState <--> GR00T-N1 in/output state tensor
    """

    @staticmethod
    def cmd_to_isaac(src: "client_stream_message_pb2.G1JoinState") -> np.ndarray:
        """
        fullbody [43,] join states tensor -> uppbody [23,] tensor for GR00T-N1
        """
        dst = np.zeros([43], dtype=np.float32)

        dst[11] = src.left_shoulder_angle.y # pitch
        dst[15] = src.left_shoulder_angle.x # roll
        dst[19] = src.left_shoulder_angle.z # yaw

        dst[12] = src.right_shoulder_angle.y
        dst[16] = src.right_shoulder_angle.x
        dst[20] = src.right_shoulder_angle.z

        dst[21] = src.left_elbow
        dst[22] = src.right_elbow

        dst[23] = src.left_wrist_angle.x # roll
        dst[25] = src.left_wrist_angle.y # pitch
        dst[27] = src.left_wrist_angle.z # yaw

        dst[24] = src.right_wrist_angle.x
        dst[26] = src.right_wrist_angle.y
        dst[28] = src.right_wrist_angle.z

        dst[31] = src.left_hand.thumb_0
        dst[37] = src.left_hand.thumb_1
        dst[41] = src.left_hand.thumb_2
        dst[29] = src.left_hand.index_0
        dst[35] = src.left_hand.index_1
        dst[30] = src.left_hand.middle_0
        dst[36] = src.left_hand.middle_1

        dst[34] = src.right_hand.thumb_0
        dst[40] = src.right_hand.thumb_1
        dst[42] = src.right_hand.thumb_2
        dst[32] = src.right_hand.index_0
        dst[38] = src.right_hand.index_1
        dst[33] = src.right_hand.middle_0
        dst[39] = src.right_hand.middle_1

        return dst

    @staticmethod
    def cmd_to_gr00t(src: "client_stream_message_pb2.G1JoinState") -> Dict[str, np.ndarray]:
        state = {
            "state.left_shoulder": [
                src.left_shoulder_angle.y,  # NOTE: [roll, pitch, yaw] -> [pitch, roll, yaw]
                src.left_shoulder_angle.x,
                src.left_shoulder_angle.z,
            ],
            "state.right_shoulder": [
                src.right_shoulder_angle.y,
                src.right_shoulder_angle.x,
                src.right_shoulder_angle.z,
            ],
            "state.left_elbow": [src.left_elbow],
            "state.right_elbow": [src.right_elbow],
            "state.left_wrist": [
                src.left_wrist_angle.x,
                src.left_wrist_angle.y,
                src.left_wrist_angle.z,
            ],
            "state.right_wrist": [
                src.right_wrist_angle.x,
                src.right_wrist_angle.y,
                src.right_wrist_angle.z,
            ],
            "state.left_hand": [
                src.left_hand.thumb_0,
                src.left_hand.thumb_1,
                src.left_hand.thumb_2,
                src.left_hand.index_0,
                src.left_hand.index_1,
                src.left_hand.middle_0,
                src.left_hand.middle_1,
            ],
            "state.right_hand": [
                src.right_hand.thumb_0,
                src.right_hand.thumb_1,
                src.right_hand.thumb_2,
                src.right_hand.index_0,
                src.right_hand.index_1,
                src.right_hand.middle_0,
                src.right_hand.middle_1,
            ]
        }
        state = {
            k: np.asarray(v)[np.newaxis, ...]
            for k, v in state.items()
        }
        return state

    @staticmethod
    def isaac_to_cmd(src: np.ndarray):
        """
        fullbody [43,] join states tensor -> G1ActionCommand protobuf message
        """
        dst = server_control_message_pb2.G1ActionCommand()

        dst.left_shoulder_angle.y = src[11]
        dst.left_shoulder_angle.x = src[15]
        dst.left_shoulder_angle.z = src[19]

        dst.right_shoulder_angle.y = src[12]
        dst.right_shoulder_angle.x = src[16]
        dst.right_shoulder_angle.z = src[20]

        dst.left_elbow = src[21]
        dst.right_elbow = src[22]

        dst.left_wrist_angle.x = src[23]
        dst.left_wrist_angle.y = src[25]
        dst.left_wrist_angle.z = src[27]

        dst.right_wrist_angle.x = src[24]
        dst.right_wrist_angle.y = src[26]
        dst.right_wrist_angle.z = src[28]

        dst.left_hand.thumb_0 = src[31]
        dst.left_hand.thumb_1 = src[37]
        dst.left_hand.thumb_2 = src[41]
        dst.left_hand.index_0 = src[29]
        dst.left_hand.index_1 = src[35]
        dst.left_hand.middle_0 = src[30]
        dst.left_hand.middle_1 = src[36]

        dst.right_hand.thumb_0 = src[34]
        dst.right_hand.thumb_1 = src[40]
        dst.right_hand.thumb_2 = src[42]
        dst.right_hand.index_0 = src[32]
        dst.right_hand.index_1 = src[38]
        dst.right_hand.middle_0 = src[33]
        dst.right_hand.middle_1 = src[39]

        return dst

    @staticmethod
    def gr00t_to_cmd(src: Dict[str, np.ndarray]):
        action_keys = list(src.keys())
        action_horizon = src[action_keys[0]].shape[0]

        predicts = []
        for i in range(action_horizon):
            state = server_control_message_pb2.G1ActionCommand()

            state.left_shoulder_angle.x = src['action.left_shoulder'][i][1] # roll
            state.left_shoulder_angle.y = src['action.left_shoulder'][i][0] # picth
            state.left_shoulder_angle.z = src['action.left_shoulder'][i][2] # yaw

            state.right_shoulder_angle.x = src['action.right_shoulder'][i][1]
            state.right_shoulder_angle.y = src['action.right_shoulder'][i][0]
            state.right_shoulder_angle.z = src['action.right_shoulder'][i][2]

            state.left_elbow = src['action.left_elbow'][i]
            state.right_elbow = src['action.right_elbow'][i]

            state.left_wrist_angle.x = src['action.left_wrist'][i][0]
            state.left_wrist_angle.y = src['action.left_wrist'][i][1]
            state.left_wrist_angle.z = src['action.left_wrist'][i][2]

            state.right_wrist_angle.x = src['action.right_wrist'][i][0]
            state.right_wrist_angle.y = src['action.right_wrist'][i][1]
            state.right_wrist_angle.z = src['action.right_wrist'][i][2]

            state.left_hand.thumb_0 = src['action.left_hand'][i][0]
            state.left_hand.thumb_1 = src['action.left_hand'][i][1]
            state.left_hand.thumb_2 = src['action.left_hand'][i][2]
            state.left_hand.index_0 = src['action.left_hand'][i][3]
            state.left_hand.index_1 = src['action.left_hand'][i][4]
            state.left_hand.middle_0 = src['action.left_hand'][i][5]
            state.left_hand.middle_1 = src['action.left_hand'][i][6]

            state.right_hand.thumb_0 = src['action.right_hand'][i][0]
            state.right_hand.thumb_1 = src['action.right_hand'][i][1]
            state.right_hand.thumb_2 = src['action.right_hand'][i][2]
            state.right_hand.index_0 = src['action.right_hand'][i][3]
            state.right_hand.index_1 = src['action.right_hand'][i][4]
            state.right_hand.middle_0 = src['action.right_hand'][i][5]
            state.right_hand.middle_1 = src['action.right_hand'][i][6]

            predicts.append(state)

        return predicts


class G1_Pose(Enum):
    LEFT_HAND_FINGER_BEND_INWARD = 1
    LEFT_HAND_FINGER_BEND_OUTWARD = 3
    LEFT_HAND_FINGER_GRIP = 2

    RIGHT_HAND_FINGER_BEND_INWARD = 4
    RIGHT_HAND_FINGER_BEND_OUTWARD = 6
    RIGHT_HAND_FINGER_GRIP = 5

    LEFT_WRIST_BEND=7
    RIGHT_WRIST_BEND=8

    LEFT_ELBOW_BEND=9
    RIGHT_ELBOW_BEND=10

    LEFT_SHOULDER_RAISE=11
    RIGHT_SHOULDER_RAISE=12

    ARM_STRAIGHT = 13


def build_join_pos(pose: G1_Pose):
    # left hand join move inwoard.
    jp = np.array([0] * 43, dtype=np.float32)
    rad_90 = 1.57079633

    if pose == G1_Pose.LEFT_HAND_FINGER_BEND_INWARD:
        jp[41] = -1
        jp[35:38] = -1
        jp[29:32] = -1
        jp[31] = 0 # first joins of the thumb only rotate entire finger. dont move it here.
    elif pose == G1_Pose.LEFT_HAND_FINGER_BEND_OUTWARD:
        jp[41] = 1
        jp[35:38] = 1 # NOTE: index, middle finger cant bend backward, so this wont do anything
        jp[29:32] = 1
        jp[31] = 0 # first joins of the thumb only rotate entire finger. dont move it here.
    elif pose == G1_Pose.LEFT_HAND_FINGER_GRIP:
        jp[35:38] = -1
        jp[29:32] = -1
        # move thumb in different direction
        jp[31] = 0
        jp[37] = 1
        jp[41] = 1
    elif pose == G1_Pose.RIGHT_HAND_FINGER_BEND_INWARD:
        jp[42] = 1
        jp[38:41] = 1
        jp[32:35] = 1
        jp[34] = 0 # first joins of the thumb only rotate entire finger. dont move it here.
    elif pose == G1_Pose.RIGHT_HAND_FINGER_BEND_OUTWARD:
        jp[42] = -1
        jp[38:41] = -1
        jp[32:35] = -1
        jp[34] = 0 # first joins of the thumb only rotate entire finger. dont move it here.
    elif pose == G1_Pose.RIGHT_HAND_FINGER_GRIP:
        jp[38:41] = 1
        jp[32:35] = 1
        # move thumb in different direction
        jp[34] = 0
        jp[40] = -1
        jp[42] = -1

    elif pose == G1_Pose.LEFT_WRIST_BEND:
        # [roll, pitch, yaw]
        jp[23] = 0  # valid range around -1.97 ~ 1.97
        jp[25] = 0
        jp[27] = rad_90 * 3
    elif pose == G1_Pose.RIGHT_WRIST_BEND:
        jp[24] = 1
        jp[26] = 1
        jp[28] = 1

    elif pose == G1_Pose.LEFT_ELBOW_BEND:
        jp[21] = -2  # -1.04 ~ 1.98
    elif pose == G1_Pose.RIGHT_ELBOW_BEND:
        jp[22] = 1


    elif pose == G1_Pose.LEFT_SHOULDER_RAISE:
        jp[11] = 0  # -3.08 ~ 2.6697 / forward rotate ~ backward
        jp[15] = 0  # 0 - ~ 2.25 / ~ raise sideway
        jp[19] = rad_90 * 4  # -2.6 ~ 2.616
    elif pose == G1_Pose.RIGHT_SHOULDER_RAISE:
        jp[12] = -rad_90 * 2  # -1.04 ~ 1.98
        jp[16] = 0  # -1.04 ~ 1.98
        jp[20] = 0  # -1.04 ~ 1.98
    elif pose == G1_Pose.ARM_STRAIGHT:
        jp[11 + 1] = -0.64  # -1.04 ~ 1.98
        jp[15 + 1] = -0.15  # -1.04 ~ 1.98
        jp[19 + 1] = -0.23  # -1.04 ~ 1.98
        jp[22] = 0.4

    return jp


def cycle_single_pose(step: int, pose: G1_Pose, loop_step=90):
    cur = (step % loop_step) / loop_step
    if cur < 0.5:
        return build_join_pos(pose) * cur * 2
    else:
        return build_join_pos(pose) * (1 - (cur - 0.5) * 2)


def cycle_pose_list(step: int, pose_list: List[G1_Pose], loop_step=90):
    n = len(pose_list)
    total_step = loop_step * n
    action_step = step % total_step
    sub_step = step % loop_step

    pose = pose_list[action_step // loop_step]
    return cycle_single_pose(sub_step, pose, loop_step=loop_step)


class GR00T_N1_Client(BaseInferenceClient):

    def __init__(self, contorl_hz: int, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)
        self.action_horizons = 16
        self.dof = 43
        self.action_consumed = 0
        self.contorl_hz = contorl_hz
        self.vla_interval = (16 / 30) # how long each VLA's prediction cover(in seconds)
        # self.action_cache = np.zeros([self.action_horizons, self.dof])
        self.action_cache: MsgList = []
        self.action_cache_np: Dict[str, np.ndarray] | None = None

    def smooth_action(self, pred: Dict[str, np.ndarray]):

        def exponential_moving_average(data, alpha=0.2):
            smoothed = np.zeros_like(data)
            smoothed[0, ...] = data[0, ...]
            for t in range(1, data.shape[0]):
                smoothed[t, ...] = alpha * data[t, ...] + (1 - alpha) * smoothed[t-1, ...]
            return smoothed

        output = {}
        for k, v in pred.items():
            actions = v
            if self.action_cache_np:
                # if actions.ndim == 1:
                #     actions = np.stack([self.action_cache_np[k], actions], axis=0)
                # else:
                actions = np.concatenate([self.action_cache_np[k], actions], axis=0)
            smt_actions = exponential_moving_average(actions)
            output[k] = smt_actions[-v.shape[0]:]

        self.action_cache_np = pred
        # self.action_cache_np = output
        return output

    def linear_interpolation(self, act_cmds: MsgList):
        src_pts = len(act_cmds)
        tar_pts = math.ceil(self.vla_interval * self.contorl_hz)
        # print("src_pts", src_pts, len(act_cmds), type(act_cmds))
        if tar_pts <= src_pts:
            return act_cmds

        act_tensor = [G1StateConvert.cmd_to_isaac(cmd) for cmd in act_cmds]
        inter_t = [
            (i / tar_pts) * (src_pts - 1)
            for i in range(tar_pts)
        ][::-1]
        # print("inter_t", len(inter_t), inter_t)

        new_act_ten = []
        for a, b in zip(range(src_pts), range(1, src_pts)):
            # print(a, b)
            ten_a = act_tensor[a]
            ten_b = act_tensor[b]

            while inter_t and inter_t[-1] <= b:
                pt = inter_t.pop()
                ratio = (pt - a) / (b - a)
                new_act_ten.append(ten_a + (ten_b - ten_a) * ratio)

        assert len(new_act_ten) == tar_pts, f"{len(new_act_ten)} != {tar_pts}"
        return [G1StateConvert.isaac_to_cmd(ten) for ten in new_act_ten]

    def update_state(self, img: np.ndarray, state: "client_stream_message_pb2.G1JoinState"):
        # call inference server
        img_int = (img * 255).astype(np.uint8)[np.newaxis, ...]
        img_int = img_int[..., :3]
        observations = {
            "video.ego_view": img_int,
        }
        observations.update(G1StateConvert.cmd_to_gr00t(state))

        predict = self.call_endpoint("get_action", observations)
        predict = self.smooth_action(predict)

        self.action_consumed = 0
        self.action_cache = G1StateConvert.gr00t_to_cmd(predict)
        assert len(self.action_cache) == self.action_horizons
        self.action_cache = self.action_cache
        self.action_cache = self.linear_interpolation(self.action_cache)
        print(f"[{datetime.now().isoformat()}] Inference!")

    def action_avaible(self) -> bool:
        return self.action_consumed < len(self.action_cache)

    def get_action(self) -> "client_stream_message_pb2.G1JoinState":
        if self.action_consumed >= len(self.action_cache):
            raise RuntimeError()
        action = self.action_cache[self.action_consumed]
        self.action_consumed += 1
        return action


def test_single_inference_step():
    state = client_stream_message_pb2.G1JoinState()
    state.left_hand.index_1 = 1.0
    state.right_shoulder_angle.x = 1.0
    print(state.right_shoulder_angle.x)

    img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    # state.to_dict()

    client = GR00T_N1_Client(60)
    client.update_state(img, state)
    for i in range(len(client.action_cache)):
        out = client.get_action()
        print(i, '-' * 100)
        print(out)


def test_cycle_consist():
    isaac_joints = np.arange(43) + 1
    # isaac_joints = np.random.uniform(1, 10, [43])
    cmd = G1StateConvert.isaac_to_cmd(isaac_joints)
    recover = G1StateConvert.cmd_to_isaac(cmd)
    mask = recover > 0 # NOTE: only using upper body join
    assert np.isclose(isaac_joints[mask], recover[mask]).all(), (isaac_joints - recover)

    obs = {
        # "video.ego_view": np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8),
        "action.left_shoulder": np.random.rand(16, 3),
        "action.right_shoulder": np.random.rand(16, 3),
        "action.left_elbow": np.random.rand(16,),
        "action.right_elbow": np.random.rand(16,),
        "action.left_wrist": np.random.rand(16, 3),
        "action.right_wrist": np.random.rand(16, 3),
        "action.left_hand": np.random.rand(16, 7),
        "action.right_hand": np.random.rand(16, 7),
    }
    cmd_list = G1StateConvert.gr00t_to_cmd(obs)
    for i, cmd in enumerate(cmd_list):
        recover = G1StateConvert.cmd_to_gr00t(cmd)
        for k, v in recover.items():
            og = obs[k.replace('state.', 'action.')][i]
            assert np.isclose(og, v).all()


if __name__ == '__main__':
    test_cycle_consist()
    test_single_inference_step()