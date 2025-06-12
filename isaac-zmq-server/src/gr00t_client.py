
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable, Dict

import torch
import zmq
import numpy as np
import client_stream_message_pb2
import server_control_message_pb2


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
    def cmd_to_isaac(src):
        """
        fullbody [43,] join states tensor -> uppbody [23,] tensor for GR00T-N1
        """
        dst = np.zeros([43], dtype=np.float32)

    @staticmethod
    def cmd_to_gr00t(src: "client_stream_message_pb2.G1JoinState") -> Dict[str, np.ndarray]:
        state = {
            "state.left_shoulder": [
                src.left_shoulder_angle.y,  # NOTE: [pitch, roll, yaw] -> [roll, pitch, yaw]
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
                src.left_wrist_angle.y,
                src.left_wrist_angle.x,
                src.left_wrist_angle.z,
            ],
            "state.right_wrist": [
                src.right_wrist_angle.y,
                src.right_wrist_angle.x,
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
    def isaac_to_cmd(src):
        """
        fullbody [43,] join states tensor -> G1ActionCommand protobuf message
        """
        dst = server_control_message_pb2.G1ActionCommand()

    @staticmethod
    def gr00t_to_cmd(src: Dict[str, np.ndarray]):
        action_keys = list(src.keys())
        action_horizon = src[action_keys[0]].shape[0]

        predicts = []
        for i in range(action_horizon):
            state = client_stream_message_pb2.G1JoinState()

            state.left_shoulder_angle.y = src['action.left_shoulder'][i][0]
            state.left_shoulder_angle.x = src['action.left_shoulder'][i][1]
            state.left_shoulder_angle.z = src['action.left_shoulder'][i][2]

            state.right_shoulder_angle.y = src['action.right_shoulder'][i][0]
            state.right_shoulder_angle.x = src['action.right_shoulder'][i][1]
            state.right_shoulder_angle.z = src['action.right_shoulder'][i][2]

            state.left_elbow = src['action.left_elbow'][i]
            state.right_elbow = src['action.right_elbow'][i]

            state.left_wrist_angle.y = src['action.left_wrist'][i][0]
            state.left_wrist_angle.x = src['action.left_wrist'][i][1]
            state.left_wrist_angle.z = src['action.left_wrist'][i][2]

            state.right_wrist_angle.y = src['action.right_wrist'][i][0]
            state.right_wrist_angle.x = src['action.right_wrist'][i][1]
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


class GR00T_N1_Client(BaseInferenceClient):

    def __init__(self, host: str = "localhost", port: int = 5555, timeout_ms: int = 15000):
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)
        self.action_horizons = 16
        self.dof = 43
        self.action_consumed = 0
        # self.action_cache = np.zeros([self.action_horizons, self.dof])
        self.action_cache = []

    def get_action(self, img: np.ndarray, state: "client_stream_message_pb2.G1JoinState") -> "client_stream_message_pb2.G1JoinState":
        if self.action_consumed >= len(self.action_cache):
            # call inference server
            observations = {
                "video.ego_view": (img * 255).astype(np.uint8),
            }
            observations.update(G1StateConvert.cmd_to_gr00t(state))

            predict = self.call_endpoint("get_action", observations)
            # for k, v in predict.items():
            #     print('predict', k, v.shape)

            self.action_consumed = 0
            self.action_cache = G1StateConvert.gr00t_to_cmd(predict)
            assert len(self.action_cache) == self.action_horizons

        action = self.action_cache[self.action_consumed]
        self.action_consumed += 1
        return action


def test_single_inference_step():
    state = client_stream_message_pb2.G1JoinState()
    state.left_hand.index_1 = 1.0
    state.right_shoulder_angle.x = 1.0
    print(state.right_shoulder_angle.x)

    img = np.random.randint(0, 256, (1, 480, 640, 3), dtype=np.uint8)
    # state.to_dict()

    client = GR00T_N1_Client()
    client.get_action(img, state)


if __name__ == '__main__':
    test_single_inference_step()