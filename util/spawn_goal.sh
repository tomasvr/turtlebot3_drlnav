#!/bin/bash
# Spawn a goal at a specific location: ./spawngoal 1 2
ros2 topic pub --once /goal_pose geometry_msgs/msg/Pose "{position: {x: $1, y: $2, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}}"
