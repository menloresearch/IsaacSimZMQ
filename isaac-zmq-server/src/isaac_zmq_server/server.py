# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import threading
import time
import traceback

import zmq


class ZMQServer:
    """
    Server for handling ZMQ communication.

    This class implements a singleton pattern and provides methods for creating
    ZMQ sockets, sending and receiving data in separate threads, and cleaning up
    resources when they are no longer needed.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for ZMQServer."""
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """Initialize the ZMQServer with empty collections for sockets and threads."""
        # Skip initialization if already initialized (singleton pattern)
        if hasattr(self, "push_sockets"):
            return

        self.push_sockets = {}
        self.pull_sockets = {}
        self.reciveing_threads = {}
        self.sending_threads = {}

        # ZMQ context
        self._context = None

    def context(self) -> zmq.Context:
        """
        Returns the ZMQ context instance.
        If the context has not been initialized, it creates a new ZMQ context and assigns it to the `_context` attribute.

        Returns:
            zmq.Context: The ZMQ context instance.
        """
        if not self._context:
            self._context = zmq.Context()
        return self._context

    def get_pull_socket(self, port: int) -> zmq.Socket:
        """
        Creates and returns a new pull socket that is bound to the specified port.

        Args:
            port (int): The port number to bind the socket to.

        Returns:
            zmq.Socket: The newly created pull socket.
        """
        addr = f"tcp://*:{port}"
        sock = self.context().socket(zmq.PULL)
        sock.set_hwm(1)  # High water mark: only buffer 1 message
        sock.bind(addr)
        sock.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for receiving
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        self.pull_sockets[port] = sock
        return sock

    def get_push_socket(self, port: int) -> zmq.Socket:
        """
        Creates and returns a ZeroMQ PUSH socket bound to the specified port.

        Args:
            port (int): The port number to bind the socket to.

        Returns:
            zmq.Socket: The created PUSH socket.
        """
        addr = f"tcp://*:{port}"
        sock = self.context().socket(zmq.PUSH)
        sock.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second timeout for sending
        sock.bind(addr)
        self.push_sockets[port] = sock
        return sock

    def subscribe_to_socket_in_loop(self, name: str, port: int, fn: callable) -> None:
        """
        Receives messages from a socket in a loop and calls a given function for each message.

        This method creates a new thread that continuously receives messages from the specified
        port and passes them to the provided callback function.

        Args:
            name (str): The name of the receiving thread.
            port (int): The port number to receive messages from.
            fn (callable): A callable function that takes a message as input.
        """
        # Create socket for receiving
        sock = self.get_pull_socket(port)
        stop_event = threading.Event()

        def loop():
            """Thread function that continuously receives messages."""
            while not stop_event.is_set():
                try:
                    msg = sock.recv()
                    fn(msg)
                except zmq.Again:
                    continue
                except:
                    print("[isaac-zmq-server] Unable to unpack from socket...")
                    print(traceback.format_exc())
                    continue

            # Clean up when thread is finsihed
            sock.close()
            del self.pull_sockets[port]

        # Start the thread
        worker = threading.Thread(target=loop)
        self.reciveing_threads[name] = (worker, stop_event)
        worker.start()

    def publish_protobuf_in_loop(self, name: str, port: int, rate_hz: float, fn: callable) -> None:
        """
        Sends protobuf messages from a socket in a loop at a specified rate.

        This method creates a new thread that continuously sends protobuf messages at the specified
        rate to the specified port. The protobuf message to send is obtained by calling the provided
        callback function.

        Args:
            name (str): The name of the sending thread.
            port (int): The port number to send data to.
            rate_hz (float): The rate at which data is sent in Hz.
            fn (callable): A callable function that returns a protobuf message.
        """
        # Create socket for sending
        sock = self.get_push_socket(port)
        stop_event = threading.Event()

        def loop():
            """Thread function that continuously sends protobuf messages at the specified rate."""
            while not stop_event.is_set():
                try:
                    # Get the protobuf message from the callback function
                    proto_msg = fn()
                    # Serialize the protobuf message and send it
                    sock.send(proto_msg.SerializeToString())
                except zmq.Again:
                    continue
                except Exception as e:
                    print(f"[isaac-zmq-server] Unable to send protobuf to socket: {e}")
                    continue

                # Sleep to maintain the desired rate
                time.sleep(1 / rate_hz)

            # Clean up when thread is finished
            sock.close()
            del self.push_sockets[port]

        # Start the sending thread
        worker = threading.Thread(target=loop)
        self.sending_threads[name] = (worker, stop_event)
        worker.start()

    def cleanup(self) -> None:
        """
        Stops and joins all receiving and sending threads.

        This function is used to clean up the threads when they are no longer needed.
        It sets the stop event for each thread and then joins them to ensure they have finished.
        """
        # Stop and join all receiving threads
        for name, (worker, stop_event) in self.reciveing_threads.items():
            stop_event.set()
            worker.join()

        # Stop and join all sending threads
        for name, (worker, stop_event) in self.sending_threads.items():
            stop_event.set()
            worker.join()
