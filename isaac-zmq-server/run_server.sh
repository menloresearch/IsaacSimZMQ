xhost +local:appuser
docker run --gpus all --network host \
       -e DISPLAY=$DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -e XAUTHORITY=$XAUTHORITY \
       -v $XAUTHORITY:$XAUTHORITY \
       -v ./src:/isaac-zmq-server/src \
       --device /dev/input \
       --device /dev/input/event21 \
       --device /dev/input/event22 \
       --device /dev/input/event23 \
       --privileged \
       -it --rm \
        isaac-zmq-server bash
