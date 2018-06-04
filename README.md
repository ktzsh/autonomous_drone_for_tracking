# Autonomous Drone for Object Tracking
The task is to create a self-driving UAV capable of keeping a target object under some constrained motion in center of its view thus effectively tracking it.

Object Tracking is done by using simulator to get the realtime location of the object being tracked, Car incase of AirSim. This scenario might not be practical since getting accurate location of tracked object is not possible in most scenarios, however it is suffecient to validate the hypothesis of autonomous tracking in constrained motion. However using techniques of computer vision and basic geometry it is possible to estimate location of the object from the captured frame given the altitude, elevation and focal length are known.

## Dependencies
1. Tensorflow
2. OpenCV (optional)
3. AirSim (custom fork)

## Instllation
1. Run `git clone --recursive https://github.com/kshitiz38/autonomous_drone_for_tracking.git`
    - NOTE: If you didn't clone with the --recursive flag run manually the following code
        `git submodule update --init --recursive`

2. AirSim
    - Follow instructions at https://github.com/Microsoft/AirSim
    - Tested on Linux(Ubuntu 16.04/18.04) UE build 4.18

## Usage
1. Update the config file in config.json
    - Choose Training Parameters
    - Choose Car Driving Algorithim or drive manually
2. Run `python DQNAgentSim.py`

## NOTE
The code for using the object detection network is available under `experiments`
