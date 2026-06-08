#!/usr/bin/env python3
# coding=utf-8
"""
virtual_mover.py
----------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

Simulates the robot physically moving along the shortest path
produced by lidar_slam_planner.py.

How it works:
  1. lidar_slam_planner.py runs normally and publishes /planned_path
  2. This node subscribes to /planned_path
  3. It walks the robot along each waypoint virtually:
       - publishes fake odometry on /odom_raw  (feeds back into planner)
       - broadcasts map -> base_footprint TF   (RViz shows robot moving)
       - publishes /virtual_cmd_vel            (shows what velocity would be sent)
  4. Prints a live progress log in the terminal

Run order:
  Terminal 1:  python3 lidar_slam_planner.py
  Terminal 2:  python3 virtual_mover.py
  Terminal 3:  ros2 topic pub /goal geometry_msgs/Point "{x: 2.0, y: 1.0, z: 0.0}" --once

RViz:
  Fixed Frame : map
  Add topics  : /map, /planned_path, /slam_pose
"""

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import (
    PoseStamped, Twist, TransformStamped,
    Quaternion, Point
)
from tf2_ros import TransformBroadcaster
import numpy as np


# ─────────────────────── CONFIG ───────────────────────────────
ROBOT_SPEED_MPS   = 0.3    # simulated forward speed  (m/s)
WAYPOINT_TOLERANCE = 0.08  # distance to consider waypoint reached (m)
PUBLISH_HZ        = 20.0   # odometry / TF publish rate
GOAL_TOLERANCE    = 0.30   # distance to consider goal reached (m)
# ──────────────────────────────────────────────────────────────


def yaw_from_quaternion(q):
    return math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )


def quaternion_from_yaw(yaw):
    return Quaternion(
        x=0.0, y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


class VirtualMoverNode(Node):

    def __init__(self):
        super().__init__("virtual_mover")

        # Robot virtual state
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # Path following state
        self.waypoints     = []   # list of (wx, wy) tuples
        self.wp_index      = 0    # current waypoint index
        self.is_moving     = False
        self.goal_reached  = False

        self.tf_broadcaster = TransformBroadcaster(self)

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.create_subscription(Path, "/planned_path", self._path_cb, 10)

        # Publishers
        self.odom_pub    = self.create_publisher(Odometry, "/odom_raw",          10)
        self.cmdvel_pub  = self.create_publisher(Twist,    "/virtual_cmd_vel",   10)
        self.pose_pub    = self.create_publisher(PoseStamped, "/virtual_pose",   10)

        # Main control loop
        dt = 1.0 / PUBLISH_HZ
        self.create_timer(dt, self._control_loop)

        self.get_logger().info("VirtualMoverNode ready.")
        self.get_logger().info("Waiting for /planned_path from lidar_slam_planner...")
        self.get_logger().info(
            'Send a goal: ros2 topic pub /goal geometry_msgs/Point '
            '"{x: 2.0, y: 1.0, z: 0.0}" --once'
        )

    # ── callbacks ──────────────────────────────────────────────

    def _path_cb(self, msg: Path):
        """Called whenever lidar_slam_planner publishes a new path."""
        if not msg.poses:
            return

        new_waypoints = [(p.pose.position.x, p.pose.position.y)
                         for p in msg.poses]

        # Only restart if path changed significantly
        if self._path_changed(new_waypoints):
            self.waypoints    = new_waypoints
            self.wp_index     = 1   # skip waypoint 0 (robot's current position)
            self.is_moving    = True
            self.goal_reached = False
            self.get_logger().info(
                f"New path received: {len(self.waypoints)} waypoints, "
                f"total ≈ {self._path_length():.2f} m"
            )

    # ── main loop ──────────────────────────────────────────────

    def _control_loop(self):
        """Runs at PUBLISH_HZ. Moves robot toward next waypoint."""
        now = self.get_clock().now().to_msg()

        if self.is_moving and self.waypoints and self.wp_index < len(self.waypoints):
            tx, ty = self.waypoints[self.wp_index]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            if dist < WAYPOINT_TOLERANCE:
                # Reached this waypoint — advance
                self.wp_index += 1
                if self.wp_index >= len(self.waypoints):
                    self._on_goal_reached()
                    return
                self.get_logger().info(
                    f"  Waypoint {self.wp_index}/{len(self.waypoints)} reached  "
                    f"→ next: {self.waypoints[self.wp_index]}"
                )
            else:
                # Move toward waypoint
                target_yaw = math.atan2(dy, dx)
                step = ROBOT_SPEED_MPS / PUBLISH_HZ
                step = min(step, dist)   # don't overshoot

                self.x   += step * math.cos(target_yaw)
                self.y   += step * math.sin(target_yaw)
                self.yaw  = target_yaw

                # Publish virtual cmd_vel
                twist = Twist()
                twist.linear.x  = ROBOT_SPEED_MPS
                twist.angular.z = 0.0
                self.cmdvel_pub.publish(twist)

        # Always broadcast TF and odometry
        self._broadcast_tf(now)
        self._publish_odom(now)
        self._publish_pose(now)

    # ── publishers ─────────────────────────────────────────────

    def _broadcast_tf(self, stamp):
        t = TransformStamped()
        t.header.stamp    = stamp
        t.header.frame_id = "map"
        t.child_frame_id  = "base_footprint"
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = quaternion_from_yaw(self.yaw)
        self.tf_broadcaster.sendTransform(t)

    def _publish_odom(self, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id  = "base_footprint"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation = quaternion_from_yaw(self.yaw)
        msg.twist.twist.linear.x  = ROBOT_SPEED_MPS if self.is_moving else 0.0
        msg.twist.twist.angular.z = 0.0
        self.odom_pub.publish(msg)

    def _publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.position.z = 0.0
        msg.pose.orientation = quaternion_from_yaw(self.yaw)
        self.pose_pub.publish(msg)

    # ── helpers ────────────────────────────────────────────────

    def _on_goal_reached(self):
        self.is_moving = True
        self.goal_reached = True
        self.is_moving = False

        # Stop robot
        twist = Twist()
        self.cmdvel_pub.publish(twist)

        goal = self.waypoints[-1]
        self.get_logger().info(
            f"\n{'='*45}\n"
            f"  GOAL REACHED: ({goal[0]:.2f}, {goal[1]:.2f})\n"
            f"  Total waypoints followed: {len(self.waypoints)}\n"
            f"  Final position : ({self.x:.3f}, {self.y:.3f})\n"
            f"  Path length    : {self._path_length():.2f} m\n"
            f"{'='*45}"
        )

    def _path_changed(self, new_wps) -> bool:
        """Returns True if new path is meaningfully different from current."""
        if not self.waypoints:
            return True
        if len(new_wps) != len(self.waypoints):
            return True
        # Check if goal changed
        if (abs(new_wps[-1][0] - self.waypoints[-1][0]) > 0.1 or
                abs(new_wps[-1][1] - self.waypoints[-1][1]) > 0.1):
            return True
        return False

    def _path_length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array(self.waypoints)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = VirtualMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()