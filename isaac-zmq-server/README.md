# Example container to mock as Server for Isaac Sim ZMQ Bridge


This example container provides a starting point for building your own server to communicate with Isaac Sim using ZMQ and Protobuf.
You can use it to run and test your CV models, or any other task that will form a closed loop with Isaac Sim.

The server also provides a GUI to visualize the data sensor messages being recived, using the [DearPyGui](https://github.com/hoffstadt/DearPyGui) library, which is a simple and easy to use and extend.

---

## Instructions

#### Server (Python inside a contatiner)

1. Build the docker image and run it
```bash
cd isaac-zmq-server
./build_server.sh
./run_server.sh
```
2. Inside the container, run the server
```bash
python example.py
```
3. Optional - For the Franka RMPFlow (Multi Camera), start two servers


```bash
# Inside the container
python example.py # server 1 for main camera
# in a second container
python example.py --subscribe_only 1 --port 5591 # server 2 for gripper camera
```
