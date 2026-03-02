# Teleoperation control of humanoid robots using XR devices
This repository contains the implementation of a high-performance, real-time robot teleoperation system. By integrating Augmented Reality (AR) with advanced physics engines and kinematics libraries, this project aims to bridge the gap between human intuition and robotic precision.

## Prerequisites
- Python 3.10+
- Ubuntu 22.04 (64-bit)
- ROS2 Humble (Ubuntu 22.04)
- ([Pinocchio](https://github.com/stack-of-tasks/pinocchio))
- ([Gazebo](https://gazebosim.org/docs/all/getstarted/))
- ([dex-retargeting](https://github.com/dexsuite/dex-retargeting))

## Installation
Clone this repository to your workspace.
```bash
git clone https://github.com/Mia-estudiante/teleop_project.git
```

## Simulation Teleoperation Example
```bash
ros2 launch teleop_manager teleop.avp.h12.launch
```

## TODO
- [x] Tutorial for initialization
- [x] Tutorial for arm control
- [ ] Tutorial for hand retargeting
- [ ] Real control code of XArm7 and Ability Hand
