#
#!/usr/bin/env python3
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Ryan Shim, Gilbert, Tomas

import math
import numpy
import sys
import copy

from geometry_msgs.msg import Pose, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from turtlebot3_msgs.srv import DrlStep, Goal

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from ..common import utilities as util
from ..common.settings import ENABLE_BACKWARD, UNKNOWN, SUCCESS, COLLISION_WALL, REAL_ARENA_LENGTH, REAL_ARENA_WIDTH, \
                                REAL_SPEED_LINEAR_MAX, REAL_SPEED_ANGULAR_MAX, REAL_N_SCAN_SAMPLES, REAL_LIDAR_DISTANCE_CAP, REAL_LIDAR_CORRECTION, \
                                    REAL_THRESHOLD_COLLISION, REAL_THRESHOLD_GOAL, REAL_TOPIC_SCAN, REAL_TOPIC_VELO, REAL_TOPIC_ODOM

LINEAR = 0
ANGULAR = 1

MAX_GOAL_DISTANCE = math.sqrt(REAL_ARENA_LENGTH**2 + REAL_ARENA_WIDTH**2)

class DRLEnvironment(Node):
    def __init__(self):
        super().__init__('drl_environment')
        print(f"running on real stage")

        self.scan_topic = REAL_TOPIC_SCAN
        self.velo_topic = REAL_TOPIC_VELO
        self.odom_topic = REAL_TOPIC_ODOM
        self.goal_topic = 'goal_pose'

        self.goal_x, self.goal_y = 0.0, 0.0
        self.robot_x, self.robot_y = 0.0, 0.0
        self.robot_x_prev, self.robot_y_prev = 0.0, 0.0
        self.robot_heading = 0.0
        self.total_distance = 0.0
        self.robot_tilt = 0.0

        self.done = False
        self.succeed = UNKNOWN


        self.new_goal = False
        self.goal_angle = 0.0
        self.goal_distance = MAX_GOAL_DISTANCE
        self.initial_distance_to_goal = MAX_GOAL_DISTANCE

        self.scan_ranges = [REAL_LIDAR_DISTANCE_CAP] * REAL_N_SCAN_SAMPLES
        self.obstacle_distance = REAL_LIDAR_DISTANCE_CAP

        self.difficulty_radius = 1
        self.local_step = 0

        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)
        # publishers
        self.cmd_vel_pub = self.create_publisher(Twist, self.velo_topic, qos)
        # subscribers
        self.goal_pose_sub = self.create_subscription(Pose, self.goal_topic, self.goal_pose_callback, qos)
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos)
        self.scan_sub = self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, qos_profile=qos_profile_sensor_data)
        # servers
        self.step_comm_server = self.create_service(DrlStep, 'step_comm', self.step_comm_callback)
        self.goal_comm_server = self.create_service(Goal, 'goal_comm', self.goal_comm_callback)

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""

    def goal_pose_callback(self, msg):
        self.goal_x = msg.position.x
        self.goal_y = msg.position.y
        self.new_goal = True
        print(f"new goal! x: {self.goal_x} y: {self.goal_y}")

    def goal_comm_callback(self, request, response):
        response.new_goal = self.new_goal
        return response

    def odom_callback(self, msg):
        # self.robot_x = msg.pose.pose.position.x * -1
        # self.robot_y = msg.pose.pose.position.y * -1
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        _, _, self.robot_heading = util.euler_from_quaternion(msg.pose.pose.orientation)
        self.robot_tilt = msg.pose.pose.orientation.y

        # calculate traveled distance for logging
        if self.local_step % 32 == 0:
            self.total_distance += math.sqrt(
                (self.robot_x_prev - self.robot_x)**2 +
                (self.robot_y_prev - self.robot_y)**2)
            self.robot_x_prev = self.robot_x
            self.robot_y_prev = self.robot_y

        diff_y = self.goal_y - self.robot_y
        diff_x = self.goal_x - self.robot_x
        distance_to_goal = math.sqrt(diff_x**2 + diff_y**2)
        heading_to_goal = math.atan2(diff_y, diff_x)
        goal_angle = heading_to_goal - self.robot_heading

        while goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        while goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = distance_to_goal
        self.goal_angle = goal_angle

    def scan_callback(self, msg):
        if len(msg.ranges) != REAL_N_SCAN_SAMPLES:
            print(f"more or less scans than expected! check model.sdf, got: {len(msg.ranges)}, expected: {REAL_N_SCAN_SAMPLES}")
        # normalize laser values
        self.obstacle_distance = 1
        for i in range(REAL_N_SCAN_SAMPLES):
                self.scan_ranges[i] = numpy.clip(float(msg.ranges[i] - REAL_LIDAR_CORRECTION) / REAL_LIDAR_DISTANCE_CAP, 0, 1)
                if self.scan_ranges[i] < self.obstacle_distance:
                    self.obstacle_distance = self.scan_ranges[i]
        self.obstacle_distance *= REAL_LIDAR_DISTANCE_CAP

    def stop_reset_robot(self, success):
        self.cmd_vel_pub.publish(Twist()) # stop robot
        self.done = True


    def get_state(self, action_linear_previous, action_angular_previous):
        state = copy.deepcopy(self.scan_ranges)                                             # range: [ 0, 1]
        state.append(float(numpy.clip((self.goal_distance / MAX_GOAL_DISTANCE), 0, 1)))     # range: [ 0, 1]
        state.append(float(self.goal_angle) / math.pi)                                      # range: [-1, 1]
        state.append(float(action_linear_previous))                                         # range: [-1, 1]
        state.append(float(action_angular_previous))                                        # range: [-1, 1]
        self.local_step += 1

        if self.local_step <= 15: # Grace period to wait for fresh sensor input
            return state
        # Success
        if self.goal_distance < REAL_THRESHOLD_GOAL:
            print("Outcome: Goal reached! :)")
            self.succeed = SUCCESS
        # Collision
        elif self.obstacle_distance < REAL_THRESHOLD_COLLISION:
            print("Collision! (wall) :(")
            self.succeed = COLLISION_WALL
        # Timeout
        # elif self.time_sec >= self.episode_deadline:
        #     print("Outcome: Time out! :(")
        #     self.succeed = TIMEOUT
        if self.succeed is not UNKNOWN:
            self.stop_reset_robot(self.succeed == SUCCESS)
        return state

    def initalize_episode(self, response):
        self.initial_distance_to_goal = self.goal_distance
        response.state = self.get_state(0, 0)
        response.reward = 0.0
        response.done = False
        response.distance_traveled = 0.0
        return response

    def step_comm_callback(self, request, response):
        if len(request.action) == 0:
            return self.initalize_episode(response)

        # Unnormalize actions
        if ENABLE_BACKWARD:
            action_linear = request.action[LINEAR] * REAL_SPEED_LINEAR_MAX
        else:
            action_linear = (request.action[LINEAR] + 1) / 2 * REAL_SPEED_LINEAR_MAX
        action_angular = request.action[ANGULAR] * REAL_SPEED_ANGULAR_MAX

        # Publish action cmd
        twist = Twist()
        twist.linear.x = action_linear
        twist.angular.z = action_angular
        self.cmd_vel_pub.publish(twist)

        # Prepare repsonse
        response.state = self.get_state(request.previous_action[LINEAR], request.previous_action[ANGULAR])
        response.reward = 0.0
        response.done = self.done
        response.success = self.succeed
        response.distance_traveled = 0.0
        if self.done:
            self.new_goal = False
            response.distance_traveled = self.total_distance
            # Reset variables
            self.succeed = UNKNOWN
            self.total_distance = 0.0
            self.local_step = 0
            self.done = False
        if self.local_step % 10 == 0:
            print(f"Rtot: {response.reward:<8.2f}GD: {self.goal_distance:<8.2f}GA: {math.degrees(self.goal_angle):.1f}°\t", end='')
            print(f"MinD: {self.obstacle_distance:<8.2f}Alin: {request.action[LINEAR]:<7.1f}Aturn: {request.action[ANGULAR]:<7.1f}")
        return response

def main(args=sys.argv[1:]):
    rclpy.init(args=args)
    if len(args) == 0:
        drl_environment = DRLEnvironment()
    else:
        rclpy.shutdown()
        quit("ERROR: wrong number of arguments!")
    rclpy.spin(drl_environment)
    drl_environment.destroy()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
