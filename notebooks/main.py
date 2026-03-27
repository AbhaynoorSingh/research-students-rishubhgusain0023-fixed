#!/usr/bin/env python3
# coding=utf-8
"""Unified ROS2 node for SLAM + RRT* navigation."""
import math
import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import (
    PoseStamped, Point, Quaternion,
    TransformStamped, Twist
)
from tf2_ros import TransformBroadcaster

# ── Module imports ────────────────────────────────────────────
from slam_module import SLAMModule, SLAMResult
from rrt_virtual_mover import RRTPlanner
from lidar_slam_planner import GridPathPlanner, NavigationMetrics


# ──────────────────────────── CONFIG ──────────────────────────
ROBOT_SPEED_MPS  = 0.3    # virtual movement speed (m/s)
WAYPOINT_TOL     = 0.10   # metres — waypoint reached threshold
GOAL_TOL         = 0.30   # metres — goal reached threshold
PUBLISH_HZ       = 20.0   # control loop rate
REPLAN_EVERY_N   = 20     # replan every N scans
REPLAN_COOLDOWN  = 2.0    # seconds between replans
ARM_SCAN_ON_GOAL = False  # disabled until camera is connected
# ──────────────────────────────────────────────────────────────


def quat_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(
        x=0.0, y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


class UnifiedMainNode(Node):
    """ROS2 node that integrates SLAM, RRT* path planning, and virtual movement control."""

    def __init__(self):
        super().__init__("unified_main")

        # ── SLAM ─────────────────────────────────────────────
        self.slam = SLAMModule(
            width_m    = 20.0,
            height_m   = 20.0,
            resolution = 0.05,
        )
        self.slam.reset()

        # ── Planners ──────────────────────────────────────────
        self.rrt_planner   = RRTPlanner(self.slam._map)          # primary
        self.astar_planner = GridPathPlanner(                    # fallback
            self.slam._map, inflation_radius_m=0.15)

        # ── Path / navigation state ───────────────────────────
        self.waypoints      = []
        self.wp_index       = 0
        self.is_moving      = False
        self.goal           = None
        self.metrics        = None
        self.active_planner = "none"

        # ── Virtual robot pose ────────────────────────────────
        self.vx   = 0.0
        self.vy   = 0.0
        self.vyaw = 0.0

        # ── Sensor cache ──────────────────────────────────────
        self._latest_scan   = None
        self._latest_odom   = None
        self._latest_action = "idle"
        self._scan_count    = 0
        self._last_replan   = 0.0

        self.tf_broadcaster = TransformBroadcaster(self)

        # ── QoS ───────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 10,
            durability  = QoSDurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan',     self._scan_cb,  sensor_qos)
        self.create_subscription(
            Odometry,  '/odom_raw', self._odom_cb,  sensor_qos)
        self.create_subscription(
            Odometry,  '/odom',     self._odom_cb,  sensor_qos)
        self.create_subscription(
            Point,     '/goal',     self._goal_cb,  10)

        # ── Publishers ────────────────────────────────────────
        self.map_pub    = self.create_publisher(
            OccupancyGrid, '/map',             10)
        self.path_pub   = self.create_publisher(
            Path,          '/planned_path',    10)
        self.pose_pub   = self.create_publisher(
            PoseStamped,   '/slam_pose',       10)
        self.vpose_pub  = self.create_publisher(
            PoseStamped,   '/virtual_pose',    10)
        self.cmdvel_pub = self.create_publisher(
            Twist,         '/virtual_cmd_vel', 10)
        self.odom_pub   = self.create_publisher(
            Odometry,      '/odom_raw',        10)

        # ── Timers ────────────────────────────────────────────
        self.create_timer(1.0 / PUBLISH_HZ, self._control_loop)
        self.create_timer(5.0,              self._publish_map)
        self.create_timer(1.0,              self._publish_path)
        self.create_timer(0.1,              self._broadcast_tf)

        self.get_logger().info("=" * 50)
        self.get_logger().info("  UnifiedMainNode ready")
        self.get_logger().info("  Primary  : RRT*")
        self.get_logger().info("  Fallback : A*")
        self.get_logger().info("  SLAM     : slam_module.py")
        self.get_logger().info("=" * 50)
        self.get_logger().info(
            'Send goal: ros2 topic pub /goal geometry_msgs/Point '
            '"{x: 2.0, y: 1.5, z: 0.0}" --once'
        )

    # ══════════════════════════════════════════════════════════
    #  Sensor callbacks
    # ══════════════════════════════════════════════════════════

    def _scan_cb(self, msg: LaserScan):
        self._latest_scan = msg
        self._slam_step()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg

    def _goal_cb(self, msg: Point):
        self.goal = (msg.x, msg.y)
        self.get_logger().info(
            f"New goal: ({msg.x:.2f}, {msg.y:.2f})")
        x, y, _ = self.slam.pose
        self.metrics = NavigationMetrics(
            msg.x, msg.y, tolerance_m=GOAL_TOL)
        self.metrics.update(x, y)
        self._replan()

    # ══════════════════════════════════════════════════════════
    #  SLAM step — called every scan
    # ══════════════════════════════════════════════════════════

    def _slam_step(self):
        """
        Core loop — slam.update(observation) every scan.
        Satisfies VLN-15 requirement.
        """
        if self._latest_scan is None or self._latest_odom is None:
            return

        observation = {
            "depth":  self._latest_scan,
            "odom":   self._latest_odom,
            "action": self._latest_action,
        }

        # ── slam.update(observation) ──────────────────────────
        result: SLAMResult = self.slam.update(observation)

        x, y, yaw = result.pose_est
        explored  = result.explored_ratio
        self._scan_count += 1

        # Update navigation metrics
        if self.metrics:
            self.metrics.update(x, y)
            if self.metrics.reached:
                self._on_goal_reached()
                return

        # Periodic replan
        if self.goal and (self._scan_count % REPLAN_EVERY_N == 0):
            now = time.time()
            if now - self._last_replan > REPLAN_COOLDOWN:
                self._replan()
                self._last_replan = now

        # Log every 100 steps
        if self._scan_count % 100 == 0:
            self.get_logger().info(
                f"[SLAM] Step {result.step} | "
                f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}°) | "
                f"Explored: {explored*100:.1f}%"
            )

        self._publish_slam_pose(x, y, yaw)

    # ══════════════════════════════════════════════════════════
    #  Path planning — RRT* primary, A* fallback
    # ══════════════════════════════════════════════════════════

    def _replan(self):
        if self.goal is None:
            return

        x, y, _ = self.slam.pose

        # ── Try RRT* ─────────────────────────────────────────
        self.get_logger().info(
            f"[RRT*] Planning to ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        t0   = time.time()
        path = self.rrt_planner.plan((x, y), self.goal)
        dt   = time.time() - t0

        if path:
            self.waypoints      = self.rrt_planner.smooth_path(path)
            self.wp_index       = 1
            self.is_moving      = True
            self.active_planner = "rrt"
            length = self._path_length()
            self.get_logger().info(
                f"[RRT*] Found in {dt:.3f}s | "
                f"{len(self.waypoints)} waypoints | {length:.2f}m"
            )

        else:
            # ── Fallback to A* ───────────────────────────────
            self.get_logger().warn(
                f"[RRT*] Failed ({dt:.3f}s) — falling back to A*")
            rx, ry   = self.slam._map.world_to_cell(x, y)
            raw_path = self.astar_planner.plan(
                (x, y), self.goal, rx, ry)

            if raw_path:
                self.waypoints      = self.astar_planner.smooth_path(raw_path)
                self.wp_index       = 1
                self.is_moving      = True
                self.active_planner = "astar"
                length = self._path_length()
                self.get_logger().info(
                    f"[A*] Fallback path | "
                    f"{len(self.waypoints)} waypoints | {length:.2f}m"
                )
            else:
                self.get_logger().error(
                    "Both RRT* and A* failed to find a path.")
                self.is_moving = False
                return

        if self.metrics:
            self.metrics.set_planned_path(self.waypoints)

    # ══════════════════════════════════════════════════════════
    #  Virtual movement control loop — 20Hz
    # ══════════════════════════════════════════════════════════

    def _control_loop(self):
        now = self.get_clock().now().to_msg()

        if (self.is_moving and self.waypoints
                and self.wp_index < len(self.waypoints)):

            tx, ty = self.waypoints[self.wp_index]
            dx     = tx - self.vx
            dy     = ty - self.vy
            dist   = math.hypot(dx, dy)

            if dist < WAYPOINT_TOL:
                self.wp_index += 1
                if self.wp_index >= len(self.waypoints):
                    self._on_goal_reached()
                    return
                pct = 100.0 * self.wp_index / len(self.waypoints)
                nxt = self.waypoints[self.wp_index]
                self.get_logger().info(
                    f"  [{self.active_planner.upper()}] "
                    f"WP {self.wp_index}/{len(self.waypoints)} "
                    f"({pct:.0f}%) → ({nxt[0]:.2f}, {nxt[1]:.2f})"
                )
            else:
                target_yaw = math.atan2(dy, dx)
                step = min(ROBOT_SPEED_MPS / PUBLISH_HZ, dist)
                self.vx   += step * math.cos(target_yaw)
                self.vy   += step * math.sin(target_yaw)
                self.vyaw  = target_yaw
                self._latest_action = "forward"

                twist = Twist()
                twist.linear.x  = ROBOT_SPEED_MPS
                twist.angular.z = 0.0
                self.cmdvel_pub.publish(twist)

        self._publish_virtual_pose(now)
        self._publish_virtual_odom(now)

    # ══════════════════════════════════════════════════════════
    #  Goal reached
    # ══════════════════════════════════════════════════════════

    def _on_goal_reached(self):
        self.is_moving      = False
        self._latest_action = "idle"
        self.cmdvel_pub.publish(Twist())

        goal   = self.waypoints[-1] if self.waypoints else self.goal
        length = self._path_length()

        self.get_logger().info(
            f"\n{'='*50}\n"
            f"  GOAL REACHED via {self.active_planner.upper()}\n"
            f"  Goal        : ({goal[0]:.2f}, {goal[1]:.2f})\n"
            f"  Virtual pos : ({self.vx:.3f}, {self.vy:.3f})\n"
            f"  Path length : {length:.2f} m\n"
            f"  Waypoints   : {len(self.waypoints)}\n"
            f"  Explored    : {self.slam.explored_ratio*100:.1f}%\n"
            f"  SLAM steps  : {self.slam.step}\n"
            f"{'='*50}"
        )

        if self.metrics:
            self.metrics.report()

        self.metrics        = None
        self.goal           = None
        self.waypoints      = []
        self.active_planner = "none"

    # ══════════════════════════════════════════════════════════
    #  Publishers
    # ══════════════════════════════════════════════════════════

    def _publish_map(self):
        msg = self.slam.get_map_msg(
            stamp=self.get_clock().now().to_msg(), frame_id="map")
        self.map_pub.publish(msg)

    def _publish_path(self):
        if not self.waypoints:
            return
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for wx, wy in self.waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x  = wx
            ps.pose.position.y  = wy
            ps.pose.orientation = quat_from_yaw(0.0)
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def _publish_slam_pose(self, x, y, yaw):
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.orientation = quat_from_yaw(yaw)
        self.pose_pub.publish(msg)

    def _publish_virtual_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = self.vx
        msg.pose.position.y = self.vy
        msg.pose.orientation = quat_from_yaw(self.vyaw)
        self.vpose_pub.publish(msg)

    def _publish_virtual_odom(self, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id  = "base_footprint"
        msg.pose.pose.position.x  = self.vx
        msg.pose.pose.position.y  = self.vy
        msg.pose.pose.orientation = quat_from_yaw(self.vyaw)
        msg.twist.twist.linear.x  = ROBOT_SPEED_MPS if self.is_moving else 0.0
        self.odom_pub.publish(msg)

    def _broadcast_tf(self):
        now = self.get_clock().now().to_msg()
        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = "map"
        t.child_frame_id  = "base_footprint"
        t.transform.translation.x = self.vx
        t.transform.translation.y = self.vy
        t.transform.translation.z = 0.0
        t.transform.rotation = quat_from_yaw(self.vyaw)
        self.tf_broadcaster.sendTransform(t)

    def _path_length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array(self.waypoints)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = UnifiedMainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.metrics:
            node.metrics.report()
        node.get_logger().info(
            f"Shutdown | Steps: {node.slam.step} | "
            f"Explored: {node.slam.explored_ratio*100:.1f}%"
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()