#!/usr/bin/env python3
# coding=utf-8
"""
main.py
-------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

Unified node: SLAM + RRT* navigation + YOLOv8 visual task execution.

Task pipeline (triggered via /task topic):
  1. ARM SCAN   — sweep arm camera at 45/90/135 deg, run YOLO on each frame
  2. LOCALISE   — SLAM reports current robot position
  3. ESTIMATE   — compute object world position from detection angle + LiDAR
  4. PLAN       — RRT* plans shortest path to object (A* fallback)
  5. NAVIGATE   — virtually move robot along path
  6. INTERACT   — arm moves to reach/grip object on arrival

Send a task:
  ros2 topic pub /task std_msgs/String "{data: 'cup of water'}" --once

Send a raw XY goal (bypasses vision):
  ros2 topic pub /goal geometry_msgs/Point "{x: 2.0, y: 1.5, z: 0.0}" --once

YOLO label mapping — extend as needed:
  "cup of water" -> ["cup", "bottle"]
  "chair"        -> ["chair"]
  "person"       -> ["person"]
"""

import math
import os
import sys
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
from std_msgs.msg import String
from tf2_ros import TransformBroadcaster

# ── Project modules ───────────────────────────────────────────
from slam_module import SLAMModule, SLAMResult
from rrt_virtual_mover import QuadRRTPlanner
from lidar_slam_planner import GridPathPlanner, NavigationMetrics

# ── Rosmaster_Camera (direct OpenCV) ─────────────────────────
try:
    sys.path.insert(0, os.path.expanduser("~/Rosmaster/rosmaster"))
    from camera_rosmaster import Rosmaster_Camera
    ROSMASTER_CAM_AVAILABLE = True
except ImportError:
    ROSMASTER_CAM_AVAILABLE = False

# ── Arm driver ────────────────────────────────────────────────
try:
    from Rosmaster_Lib import Rosmaster
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False

# ── YOLOv8 ───────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# ──────────────────────────── CONFIG ──────────────────────────
ROBOT_SPEED_MPS   = 0.3     # virtual movement speed (m/s)
WAYPOINT_TOL      = 0.10    # metres — waypoint reached threshold
GOAL_TOL          = 0.30    # metres — goal reached threshold
PUBLISH_HZ        = 20.0    # control loop rate
REPLAN_EVERY_N    = 100      # replan every N scans
REPLAN_COOLDOWN   = 2.0     # seconds between replans

# Camera / arm scan
SCAN_POSITIONS    = [45, 90, 135]   # servo 1 angles for visual scan
MOVE_DELAY        = 1.5             # seconds to wait after servo move
CAM_INDEX         = 1               # OpenCV device index
CAM_HFOV_DEG      = 60.0           # horizontal FOV of camera (degrees)
SAVE_DIR          = os.path.expanduser("~/scan_images")

# Arm positions
ARM_HOME          = [90, 115, 25, 45, 90, 90]   # all servos home
ARM_REACH         = [90, 45, 90, 90, 90, 60]   # reach forward
ARM_GRIP_CLOSE    = 30   # servo 6 angle to close grip
ARM_GRIP_OPEN     = 90   # servo 6 angle to open grip

# YOLO
YOLO_MODEL        = "yolov8n.pt"    # nano = fastest on Jetson
YOLO_CONF         = 0.40            # minimum confidence threshold

# Task to YOLO label mapping — add more tasks here as needed
TASK_LABEL_MAP = {
    "cup of water": ["cup", "bottle"],
    "water":        ["bottle", "cup"],
    "cup":          ["cup"],
    "bottle":       ["bottle"],
    "chair":        ["chair"],
    "person":       ["person"],
    "phone":        ["cell phone"],
    "book":         ["book"],
    "laptop":       ["laptop"],
}
# ──────────────────────────────────────────────────────────────


def quat_from_yaw(yaw: float) -> Quaternion:
    return Quaternion(
        x=0.0, y=0.0,
        z=math.sin(yaw / 2.0),
        w=math.cos(yaw / 2.0)
    )


# ══════════════════════════════════════════════════════════════
#  Task state machine
# ══════════════════════════════════════════════════════════════
class TaskState:
    IDLE        = "idle"
    SCANNING    = "scanning"
    NAVIGATING  = "navigating"
    INTERACTING = "interacting"
    DONE        = "done"
    FAILED      = "failed"


# ══════════════════════════════════════════════════════════════
#  Main unified node
# ══════════════════════════════════════════════════════════════
class UnifiedMainNode(Node):
    """
    ROS2 node integrating:
      - SLAM (slam_module.py)
      - RRT* path planning (rrt_virtual_mover.py) + A* fallback
      - YOLOv8 object detection via arm-mounted camera
      - Arm servo control for reach + grip
      - Task execution via /task topic
    """

    def __init__(self):
        super().__init__("unified_main")

        os.makedirs(SAVE_DIR, exist_ok=True)

        # ── SLAM ─────────────────────────────────────────────
        self.slam = SLAMModule(
            width_m    = 20.0,
            height_m   = 20.0,
            resolution = 0.05,
        )
        self.slam.reset()

        # ── Planners ─────────────────────────────────────────
        self.rrt_planner = QuadRRTPlanner(self.slam._map)
        self.astar_planner = GridPathPlanner(
            self.slam._map, inflation_radius_m=0.15)

        # ── Navigation state ─────────────────────────────────
        self.waypoints      = []
        self.wp_index       = 0
        self.is_moving      = False
        self.goal           = None
        self.metrics        = None
        self.active_planner = "none"

        # ── Virtual robot pose ────────────────────────────────
        self.vx   = self.slam.pose[0]
        self.vy   = self.slam.pose[1]
        self.vyaw = self.slam.pose[2]

        # ── Sensor cache ─────────────────────────────────────
        self._latest_scan   = None
        self._latest_odom   = None
        self._latest_action = "idle"
        self._scan_count    = 0
        self._last_replan   = 0.0

        # ── Task state ────────────────────────────────────────
        self.task_state          = TaskState.IDLE
        self.current_task        = None
        self.target_labels       = []
        self.detected_object_pos = None   # (world_x, world_y)

        # ── Camera ───────────────────────────────────────────
        self.camera = None
        if ROSMASTER_CAM_AVAILABLE:
            try:
                self.camera = Rosmaster_Camera(
                    video_id=CAM_INDEX, debug=False)
                if not self.camera.isOpened():
                    self.get_logger().warn(
                        "Rosmaster_Camera failed to open.")
                    self.camera = None
                else:
                    self.get_logger().info(
                        f"Camera opened on device {CAM_INDEX}.")
            except Exception as e:
                self.get_logger().warn(f"Camera init error: {e}")
                self.camera = None
        else:
            self.get_logger().warn(
                "camera_rosmaster.py not found — visual tasks disabled.")

        # ── YOLO ─────────────────────────────────────────────
        self.yolo = None
        if YOLO_AVAILABLE:
            try:
                self.yolo = YOLO(YOLO_MODEL)
                self.get_logger().info(f"YOLOv8 loaded: {YOLO_MODEL}")
            except Exception as e:
                self.get_logger().warn(f"YOLO load error: {e}")
        else:
            self.get_logger().warn(
                "ultralytics not found — object detection disabled.")

        # ── Arm driver ───────────────────────────────────────
        self.bot = None
        if ARM_AVAILABLE:
            try:
                self.bot = Rosmaster()
                self.bot.create_receive_threading()
                time.sleep(0.5)
                self.bot.set_uart_servo_torque(True)
                self._arm_home()
                self.get_logger().info("Arm driver initialised.")
            except Exception as e:
                self.get_logger().error(f"Arm init failed: {e}")
                self.bot = None
        else:
            self.get_logger().warn(
                "Rosmaster_Lib not found — arm will be SIMULATED.")

        # ── TF ───────────────────────────────────────────────
        self.tf_broadcaster = TransformBroadcaster(self)

        # ── QoS ──────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability = QoSReliabilityPolicy.BEST_EFFORT,
            history     = QoSHistoryPolicy.KEEP_LAST,
            depth       = 10,
            durability  = QoSDurabilityPolicy.VOLATILE,
        )

        # ── Subscribers ──────────────────────────────────────
        self.create_subscription(
            LaserScan, '/scan',     self._scan_cb,  sensor_qos)
        self.create_subscription(
            Odometry,  '/odom_raw', self._odom_cb,  sensor_qos)
        self.create_subscription(
            Odometry,  '/odom',     self._odom_cb,  sensor_qos)
        self.create_subscription(
            Point,     '/goal',     self._goal_cb,  10)
        self.create_subscription(
            String,    '/task',     self._task_cb,  10)

        # ── Publishers ───────────────────────────────────────
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
        self.status_pub = self.create_publisher(
            String,        '/task_status',     10)

        # ── Timers ───────────────────────────────────────────
        self.create_timer(1.0 / PUBLISH_HZ, self._control_loop)
        self.create_timer(5.0,              self._publish_map)
        self.create_timer(1.0,              self._publish_path)
        self.create_timer(0.1,              self._broadcast_tf)

        self.get_logger().info("=" * 55)
        self.get_logger().info("  UnifiedMainNode ready")
        self.get_logger().info("  SLAM     : slam_module.py")
        self.get_logger().info("  Primary  : Quad RRT*")
        self.get_logger().info("  Fallback : A*")
        self.get_logger().info(
            f"  Camera   : {'YES' if self.camera else 'NO'}")
        self.get_logger().info(
            f"  YOLO     : {'YES' if self.yolo else 'NO'}")
        self.get_logger().info(
            f"  Arm      : {'YES' if self.bot else 'SIMULATED'}")
        self.get_logger().info("=" * 55)
        self.get_logger().info(
            "Send task : ros2 topic pub /task std_msgs/String "
            "'{data: \"cup of water\"}' --once")
        self.get_logger().info(
            "Send goal : ros2 topic pub /goal geometry_msgs/Point "
            '"{x: 2.0, y: 1.5, z: 0.0}" --once')

    # ══════════════════════════════════════════════════════════
    #  Sensor callbacks
    # ══════════════════════════════════════════════════════════

    def _scan_cb(self, msg: LaserScan):
        self._latest_scan = msg
        self._slam_step()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg

    def _goal_cb(self, msg: Point):
        """Raw XY goal — bypasses vision pipeline."""
        self.goal = (msg.x, msg.y)
        self.get_logger().info(
            f"[GOAL] Raw goal: ({msg.x:.2f}, {msg.y:.2f})")
        x = self.vx
        y = self.vy
        self.metrics = NavigationMetrics(
            msg.x, msg.y, tolerance_m=GOAL_TOL)
        self.metrics.update(x, y)
        self.task_state = TaskState.NAVIGATING
        self._replan()

    def _task_cb(self, msg: String):
        """
        Natural language task via /task topic.
        Triggers: visual scan -> locate object -> navigate -> interact.
        """
        task = msg.data.strip().lower()
        self.get_logger().info(f"[TASK] Received: '{task}'")

        labels = None
        for key, val in TASK_LABEL_MAP.items():
            if key in task:
                labels = val
                break

        if labels is None:
            self.get_logger().warn(
                f"[TASK] Unknown task: '{task}'. "
                f"Known tasks: {list(TASK_LABEL_MAP.keys())}")
            self._publish_status(f"Unknown task: {task}")
            return

        self.current_task        = task
        self.target_labels       = labels
        self.task_state          = TaskState.SCANNING
        self.detected_object_pos = None

        self.get_logger().info(f"[TASK] Searching for: {labels}")
        self._publish_status(f"Scanning for: {labels}")
        self._run_visual_scan()

    # ══════════════════════════════════════════════════════════
    #  SLAM step — every scan callback
    # ══════════════════════════════════════════════════════════

    def _slam_step(self):
        if self._latest_scan is None or self._latest_odom is None:
            return

        observation = {
            "depth":  self._latest_scan,
            "odom":   self._latest_odom,
            "action": self._latest_action,
        }

        result: SLAMResult = self.slam.update(observation)
        x, y, yaw = result.pose_est
        self._scan_count += 1

        

        if self.goal and (self._scan_count % REPLAN_EVERY_N == 0):
            now = time.time()
            if now - self._last_replan > REPLAN_COOLDOWN:
                self._replan()
                self._last_replan = now

        if self._scan_count % 100 == 0:
            self.get_logger().info(
                f"[SLAM] Step {result.step} | "
                f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f}deg) | "
                f"Explored: {result.explored_ratio*100:.1f}%"
            )

        self._publish_slam_pose(x, y, yaw)

    # ══════════════════════════════════════════════════════════
    #  Visual scan — arm sweep + YOLO detection
    # ══════════════════════════════════════════════════════════

    def _run_visual_scan(self):
        """
        Sweep arm across SCAN_POSITIONS.
        Run YOLO on each frame.
        Estimate world position of best detection.
        Trigger RRT* navigation toward it.
        """
        if self.camera is None or self.yolo is None:
            self.get_logger().warn(
                "[SCAN] Camera or YOLO unavailable.")
            self._publish_status("Scan failed: no camera/YOLO")
            self.task_state = TaskState.FAILED
            return

        import cv2

        self.get_logger().info(
            f"[SCAN] Sweeping arm: {SCAN_POSITIONS} deg")
        best_detection = None   # (servo_angle, cx_norm, conf, label)

        for angle in SCAN_POSITIONS:
            self._servo_move(1, angle)

            ret, frame = self.camera.get_frame()
            if not ret or frame is None or isinstance(frame, bytes):
                self.get_logger().warn(
                    f"[SCAN] No frame at {angle} deg")
                continue

            # Save frame for review
            fname = os.path.join(SAVE_DIR, f"scan_{angle:03d}deg.jpg")
            cv2.imwrite(fname, frame)

            # Run YOLO inference
            results    = self.yolo(frame, conf=YOLO_CONF, verbose=False)
            detections = results[0].boxes

            if detections is None or len(detections) == 0:
                self.get_logger().info(
                    f"[SCAN] {angle} deg — nothing detected")
                continue

            for box in detections:
                label_idx = int(box.cls[0])
                label     = self.yolo.names[label_idx]
                conf      = float(box.conf[0])

                if label in self.target_labels:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    img_w   = frame.shape[1]
                    cx_norm = ((x1 + x2) / 2.0) / img_w

                    self.get_logger().info(
                        f"[SCAN] FOUND '{label}' at {angle} deg "
                        f"(conf={conf:.2f}, cx={cx_norm:.2f})")

                    if best_detection is None or conf > best_detection[2]:
                        best_detection = (angle, cx_norm, conf, label)

        # Return arm home regardless of result
        self._arm_home()

        if best_detection is None:
            self.get_logger().warn(
                f"[SCAN] {self.target_labels} not found in any frame.")
            self._publish_status(f"Object not found: {self.target_labels}")
            self.task_state = TaskState.FAILED
            return

        servo_angle, cx_norm, conf, label = best_detection

        # ── Estimate object world position ────────────────────
        # servo_angle is relative to robot forward (90 deg = straight ahead)
        # cx_norm shifts within the camera horizontal FOV
        robot_x = self.vx
        robot_y = self.vy
        robot_yaw = self.vyaw

        servo_offset_deg  = servo_angle - 90.0
        cam_offset_deg    = (cx_norm - 0.5) * CAM_HFOV_DEG
        total_bearing_rad = (
            math.radians(servo_offset_deg + cam_offset_deg) + robot_yaw)

        # Get LiDAR range at that bearing; fall back to 1.5 m
        est_dist = self._lidar_range_at_bearing(total_bearing_rad)
        if est_dist is None:
            est_dist = 1.5

        obj_x = robot_x + est_dist * math.cos(total_bearing_rad)
        obj_y = robot_y + est_dist * math.sin(total_bearing_rad)

        self.detected_object_pos = (obj_x, obj_y)
        self.get_logger().info(
            f"[SCAN] '{label}' estimated at world "
            f"({obj_x:.2f}, {obj_y:.2f}), dist={est_dist:.2f}m")
        self._publish_status(
            f"Found '{label}' at ({obj_x:.2f}, {obj_y:.2f})")

        # ── Navigate to object ────────────────────────────────
        cx,cy = self.slam._map.world_to_cell(
                obj_x,
                obj_y
            )

        obs = self.slam._map.inflated_mask()

        if obs[cx,cy]:
            self.get_logger().warn(
                "Detected object inside obstacle region"
            )
        self.goal       = (obj_x, obj_y)
        self.metrics    = NavigationMetrics(
            obj_x, obj_y, tolerance_m=GOAL_TOL)
        self.metrics.update(robot_x, robot_y)
        self.task_state = TaskState.NAVIGATING
        self._replan()

    def _lidar_range_at_bearing(self, bearing_rad):
        """
        Return LiDAR range closest to the given world bearing,
        or None if no valid scan available.
        """
        if self._latest_scan is None:
            return None

        scan = self._latest_scan
        _, _, ryaw = self.slam.pose

        rel = bearing_rad - ryaw
        while rel >  math.pi: rel -= 2 * math.pi
        while rel < -math.pi: rel += 2 * math.pi

        idx = int((rel - scan.angle_min) / scan.angle_increment)
        idx = max(0, min(idx, len(scan.ranges) - 1))

        window = 5
        lo = max(0, idx - window)
        hi = min(len(scan.ranges), idx + window + 1)
        valid = [
            r for r in scan.ranges[lo:hi]
            if scan.range_min < r < scan.range_max
            and not math.isnan(r) and not math.isinf(r)
        ]
        return float(np.mean(valid)) if valid else None

    # ══════════════════════════════════════════════════════════
    #  Path planning — RRT* primary, A* fallback
    # ══════════════════════════════════════════════════════════

    def _replan(self):
        if self.goal is None:
            return

        x = self.vx
        y = self.vy

        self.get_logger().info(
            f"[RRT*] Planning to ({self.goal[0]:.2f}, {self.goal[1]:.2f})")
        t0   = time.time()
        path = self.rrt_planner.plan((x, y), self.goal)
        dt   = time.time() - t0

        if path:
            self.waypoints      = path
            self.wp_index       = 1
            self.is_moving      = True
            self.active_planner = "Quad-rrt"
            self.get_logger().info(
                f"[RRT*] Found in {dt:.3f}s | "
                f"{len(self.waypoints)} waypoints | "
                f"{self._path_length():.2f}m")

            # ── Bug 7 fix: log sampling stats right after planning ──
            self.get_logger().info(
                f"[HMA] Samples → "
                f"Goal: {self.rrt_planner.goal_samples}  "
                f"Corridor: {self.rrt_planner.corridor_samples}  "
                f"Global: {self.rrt_planner.global_samples}  "
                f"| Rewires: {self.rrt_planner.rewire_count}  "
                f"| Raw: {self.rrt_planner.raw_path_length:.2f}m → "
                f"Smooth: {self.rrt_planner.smoothed_path_length:.2f}m"
            )

        else:
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
                self.get_logger().info(
                    f"[A*] Fallback | "
                    f"{len(self.waypoints)} waypoints | "
                    f"{self._path_length():.2f}m")
            else:
                self.get_logger().error(
                    "Both RRT* and A* failed. Cannot navigate.")
                self.is_moving  = False
                self.task_state = TaskState.FAILED
                self._publish_status("Navigation failed: no path found")
                return

        if self.metrics:
            self.metrics.set_planned_path(self.waypoints)

    # ══════════════════════════════════════════════════════════
    #  Virtual movement — 20 Hz
    # ══════════════════════════════════════════════════════════

    def _control_loop(self):
        now = self.get_clock().now().to_msg()
        
       
        if self.metrics and self.metrics.reached:
                    self._on_goal_reached()
                    return

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
                    f"({pct:.0f}%) -> ({nxt[0]:.2f}, {nxt[1]:.2f})")
            else:
                target_yaw = math.atan2(dy, dx)
                step = min(ROBOT_SPEED_MPS / PUBLISH_HZ, dist)
                self.vx   += step * math.cos(target_yaw)
                self.vy   += step * math.sin(target_yaw)
                self.vyaw  = target_yaw
                if self.metrics:
                    self.metrics.update(self.vx, self.vy)

                # keep SLAM pose synchronized
                # keep SLAM pose synchronized
                self.slam._pose.x   = self.vx
                self.slam._pose.y   = self.vy
                self.slam._pose.yaw = self.vyaw

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

        # ── Bug 7 fix: read HMA metrics from planner attributes ──
        m = {
            "time":             self.rrt_planner.plan_time,
            "nodes":            self.rrt_planner.total_nodes,
            "rewires":          self.rrt_planner.rewire_count,
            "goal_samples":     self.rrt_planner.goal_samples,
            "corridor_samples": self.rrt_planner.corridor_samples,
            "global_samples":   self.rrt_planner.global_samples,
            "raw_length":       self.rrt_planner.raw_path_length,
            "smooth_length":    self.rrt_planner.smoothed_path_length,
        }

        # ── Bug 4 fix: use self.vx/self.vy, not self.x ──
        self.get_logger().info(
            f"\n{'='*70}\n"
            f"           HMA-RRT* NAVIGATION REPORT\n"
            f"{'='*70}\n"

            f"Goal Position       : ({goal[0]:.2f}, {goal[1]:.2f})\n"
            f"Final Position      : ({self.vx:.3f}, {self.vy:.3f})\n"

            f"\n----- Planning Metrics -----\n"
            f"Planning Time       : {m['time']:.3f} s\n"
            f"Tree Nodes          : {m['nodes']}\n"
            f"Rewire Operations   : {m['rewires']}\n"

            f"\n----- Sampling Statistics -----\n"
            f"Goal Samples        : {m['goal_samples']}\n"
            f"Corridor Samples    : {m['corridor_samples']}\n"
            f"Global Samples      : {m['global_samples']}\n"

            f"\n----- Path Optimization -----\n"
            f"Raw Path Length     : {m['raw_length']:.2f} m\n"
            f"Smoothed Length     : {m['smooth_length']:.2f} m\n"

            f"\n----- Execution -----\n"
            f"Final Path Length   : {self._path_length():.2f} m\n"
            f"Waypoints           : {len(self.waypoints)}\n"

            f"{'='*70}"
        )

        # ── Bug 4 fix: set reached=True BEFORE calling report() ──
        if self.metrics:
            self.metrics.reached = True
            self.metrics.report()

        # Task-driven → run arm interaction
        if self.task_state == TaskState.NAVIGATING and self.current_task:
            self.task_state = TaskState.INTERACTING
            self._publish_status("Arrived — interacting with object...")
            self._run_arm_interaction()
        else:
            self.task_state = TaskState.DONE
            self._publish_status("Goal reached.")

        empty_path = Path()
        empty_path.header.frame_id = "map"
        self.path_pub.publish(empty_path)
        self.metrics        = None
        self.goal           = None
        self.waypoints      = []
        self.active_planner = "none"
        self.current_task   = None

    # ══════════════════════════════════════════════════════════
    #  Arm interaction — reach, grip, lift, home
    # ══════════════════════════════════════════════════════════

    def _run_arm_interaction(self):
        self.get_logger().info("[ARM] Starting interaction sequence")

        # 1. Reach toward object
        self.get_logger().info("[ARM] Reaching toward object...")
        self._arm_set_pose(ARM_REACH)
        time.sleep(1.5)

        # 2. Close grip
        self.get_logger().info("[ARM] Closing grip...")
        self._servo_move(6, ARM_GRIP_CLOSE)
        time.sleep(1.0)

        # 3. Lift slightly
        self.get_logger().info("[ARM] Lifting...")
        if self.bot:
            self.bot.set_uart_servo_angle(2, 70)
        else:
            self.get_logger().info("[SIM] Servo 2 -> 70 deg")
        time.sleep(1.0)

        # 4. Return home with object
        self.get_logger().info("[ARM] Returning home...")
        self._arm_home()

        self.get_logger().info(
            f"[ARM] Task complete: '{self.current_task}'")
        self._publish_status(f"Task complete: {self.current_task}")
        self.task_state = TaskState.DONE

    # ══════════════════════════════════════════════════════════
    #  Arm helpers
    # ══════════════════════════════════════════════════════════

    def _servo_move(self, servo_id: int, angle: int):
        if self.bot:
            self.bot.set_uart_servo_angle(servo_id, angle)
        else:
            self.get_logger().info(
                f"[SIM] Servo {servo_id} -> {angle} deg")
        time.sleep(MOVE_DELAY)

    def _arm_set_pose(self, angles: list):
        if self.bot:
            self.bot.set_uart_servo_angle_array(angles)
        else:
            self.get_logger().info(f"[SIM] Arm pose -> {angles}")
        time.sleep(MOVE_DELAY)

    def _arm_home(self):
        self._arm_set_pose(ARM_HOME)

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
        msg.twist.twist.linear.x  = (
            ROBOT_SPEED_MPS if self.is_moving else 0.0)
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

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(f"[STATUS] {text}")

    def _path_length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array(self.waypoints)
        return float(np.sum(
            np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    # ══════════════════════════════════════════════════════════
    #  Cleanup
    # ══════════════════════════════════════════════════════════

    def destroy_node(self):
        if self.camera:
            self.camera.clear()
        if self.bot:
            self._arm_home()
            self.bot.set_uart_servo_torque(False)
            del self.bot
        super().destroy_node()


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    if not ROSMASTER_CAM_AVAILABLE:
        print("[WARN] camera_rosmaster.py not found at "
              "~/Rosmaster/rosmaster/")
    if not YOLO_AVAILABLE:
        print("[WARN] ultralytics not installed.")
    if not ARM_AVAILABLE:
        print("[WARN] Rosmaster_Lib not found — arm simulated.")

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