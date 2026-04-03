#!/usr/bin/env python3
# coding=utf-8
"""
arm_camera_scan.py
------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble
Sweeps the arm base servo (ID 1) across a configurable arc and
captures an image at each position.

Camera source priority:
  1. Rosmaster_Camera (direct OpenCV, works without Astra connected)
  2. /color/image_raw ROS2 topic (used automatically when Astra is connected)

Servo ranges (from Yahboom sample):
  Servo 1 (base)  : 0  – 180 deg
  Servo 2         : 0  – 180 deg
  Servo 3         : 0  – 180 deg
  Servo 4         : 0  – 180 deg
  Servo 5         : 0  – 270 deg
  Servo 6 (grip)  : 30 – 180 deg

Usage:
  python3 arm_camera_scan.py

  Or as a ROS2 node:
  ros2 run <your_pkg> arm_camera_scan

Published topics:
  /scan_image   (sensor_msgs/Image)  — latest captured frame

Saved images:
  ~/scan_images/scan_<angle>deg.jpg
"""

import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

# OpenCV + cv_bridge
try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# Yahboom arm driver
try:
    from Rosmaster_Lib import Rosmaster
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False

# Rosmaster_Camera (direct OpenCV capture — works without Astra)
try:
    import sys
    sys.path.insert(0, os.path.expanduser("~/Rosmaster/rosmaster"))
    from camera_rosmaster import Rosmaster_Camera
    ROSMASTER_CAM_AVAILABLE = True
except ImportError:
    ROSMASTER_CAM_AVAILABLE = False


# ─────────────────────────── CONFIG ───────────────────────────
SCAN_POSITIONS        = [45, 90, 135]   # degrees for servo 1 (base)
MOVE_DELAY            = 1.5             # seconds to wait after each move
CAPTURE_PER_POS       = 1               # images to capture per position
SAVE_DIR              = os.path.expanduser("~/scan_images")
CAMERA_TOPIC          = "/color/image_raw"
ROSMASTER_CAM_INDEX   = 1               # OpenCV device index for Rosmaster_Camera
# ──────────────────────────────────────────────────────────────


class ArmCameraScanNode(Node):

    def __init__(self):
        super().__init__("arm_camera_scan")

        os.makedirs(SAVE_DIR, exist_ok=True)

        self.bridge       = CvBridge() if CV_AVAILABLE else None
        self.latest_img   = None        # holds latest ROS2 Image msg (Astra path)
        self.ros_cam_live = False       # becomes True when /color/image_raw receives a frame
        self.scan_pub     = self.create_publisher(Image,  "/scan_image",  10)
        self.status_pub   = self.create_publisher(String, "/scan_status", 10)

        # ── Camera source selection ──────────────────────────────
        # Try Rosmaster_Camera first (direct OpenCV, no Astra needed)
        self.rm_camera = None
        if ROSMASTER_CAM_AVAILABLE:
            try:
                self.rm_camera = Rosmaster_Camera(video_id=ROSMASTER_CAM_INDEX, debug=True)
                if self.rm_camera.isOpened():
                    self.get_logger().info(
                        f"Rosmaster_Camera opened on device index {ROSMASTER_CAM_INDEX} — "
                        "will use direct capture."
                    )
                else:
                    self.get_logger().warn("Rosmaster_Camera failed to open — falling back to ROS2 topic.")
                    self.rm_camera = None
            except Exception as e:
                self.get_logger().warn(f"Rosmaster_Camera init error: {e} — falling back to ROS2 topic.")
                self.rm_camera = None
        else:
            self.get_logger().warn(
                "camera_rosmaster.py not found at ~/Rosmaster/rosmaster — "
                "falling back to /color/image_raw topic."
            )

        # Always subscribe to the Astra topic as a fallback / future upgrade
        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, cam_qos)

        # ── Arm init ─────────────────────────────────────────────
        self.bot = None
        if ARM_AVAILABLE:
            try:
                self.bot = Rosmaster()
                self.bot.create_receive_threading()
                time.sleep(0.5)
                self.bot.set_uart_servo_torque(True)
                self.get_logger().info("Arm driver initialised on /dev/ttyUSB0")
            except Exception as e:
                self.get_logger().error(f"Arm init failed: {e}")
                self.bot = None
        else:
            self.get_logger().warn("Rosmaster_Lib not found — arm moves will be SIMULATED")

        self.get_logger().info("Waiting 2 s for camera to warm up...")
        self.scan_timer = self.create_timer(2.0, self._start_scan)

    # ── ROS2 camera callback (Astra path) ─────────────────────

    def _image_cb(self, msg: Image):
        self.latest_img   = msg
        self.ros_cam_live = True

    def _start_scan(self):
        self.scan_timer.cancel()

        # Report which camera source is active
        if self.rm_camera and self.rm_camera.isOpened():
            self.get_logger().info("Camera source: Rosmaster_Camera (direct OpenCV)")
        elif self.ros_cam_live:
            self.get_logger().info("Camera source: /color/image_raw (Astra ROS2 topic)")
        else:
            self.get_logger().warn(
                "No camera source available. Images will not be saved. "
                "Check that camera_rosmaster.py is at ~/Rosmaster/rosmaster/ "
                "or that the Astra camera is connected."
            )

        self.get_logger().info("Starting arm scan...")
        self._run_scan()

    # ── Core scan logic ────────────────────────────────────────

    def _move_servo(self, servo_id: int, angle: int):
        if self.bot:
            self.bot.set_uart_servo_angle(servo_id, angle)
        else:
            self.get_logger().info(f"[SIM] Servo {servo_id} → {angle}°")
        time.sleep(MOVE_DELAY)

    def _capture_image(self, angle: int, shot: int) -> bool:
        """
        Capture an image from whichever source is available.
        Priority: Rosmaster_Camera > /color/image_raw ROS2 topic.
        Returns True on success.
        """
        filename = os.path.join(SAVE_DIR, f"scan_{angle:03d}deg_{shot}.jpg")

        # ── Path 1: Rosmaster_Camera (direct OpenCV) ──────────
        if self.rm_camera and self.rm_camera.isOpened():
            ret, frame = self.rm_camera.get_frame()
            if not ret or frame is None or isinstance(frame, bytes):
                self.get_logger().warn(f"Rosmaster_Camera: no frame at {angle}°")
                # fall through to ROS2 topic path below
            else:
                if CV_AVAILABLE:
                    cv2.imwrite(filename, frame)
                    self.get_logger().info(f"Saved (Rosmaster_Camera): {filename}")
                    # Optionally publish to /scan_image so RViz still works
                    if self.bridge:
                        try:
                            ros_img = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                            self.scan_pub.publish(ros_img)
                        except Exception:
                            pass
                    return True
                else:
                    self.get_logger().warn("cv2 not available — cannot save frame.")
                    return False

        # ── Path 2: /color/image_raw ROS2 topic (Astra) ───────
        if self.latest_img is None:
            self.get_logger().warn(f"No image available at {angle}° (topic not received)")
            return False

        if CV_AVAILABLE and self.bridge:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
                cv2.imwrite(filename, cv_img)
                self.get_logger().info(f"Saved (Astra topic): {filename}")
            except Exception as e:
                self.get_logger().error(f"cv_bridge error: {e}")
                return False
        else:
            raw_file = filename.replace(".jpg", ".raw")
            with open(raw_file, "wb") as f:
                f.write(bytes(self.latest_img.data))
            self.get_logger().info(f"Saved raw: {raw_file}")

        self.scan_pub.publish(self.latest_img)
        return True

    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)

    def _run_scan(self):
        self._publish_status("Scan started")
        results = {}

        for angle in SCAN_POSITIONS:
            self._publish_status(f"Moving to {angle}°")
            self._move_servo(1, angle)

            if self.bot:
                actual = self.bot.get_uart_servo_angle(1)
                self.get_logger().info(f"Readback servo 1: {actual}°")

            for shot in range(CAPTURE_PER_POS):
                rclpy.spin_once(self, timeout_sec=0.3)
                ok = self._capture_image(angle, shot)
                results[angle] = "OK" if ok else "FAILED"

        self._publish_status("Returning arm to home (90°)")
        self._move_servo(1, 90)

        self._publish_status("Scan complete!")
        for ang, status in results.items():
            self.get_logger().info(f"  {ang:3d}° → {status}")
        self.get_logger().info(f"Images saved to: {SAVE_DIR}")

        rclpy.shutdown()

    def destroy_node(self):
        if self.rm_camera:
            self.rm_camera.clear()
        if self.bot:
            self.bot.set_uart_servo_torque(False)
            del self.bot
        super().destroy_node()


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    if not CV_AVAILABLE:
        print("[WARN] cv_bridge / opencv not found. Images saved as raw bytes.")
    if not ARM_AVAILABLE:
        print("[WARN] Rosmaster_Lib not found. Running in simulation mode.")
    if not ROSMASTER_CAM_AVAILABLE:
        print("[WARN] camera_rosmaster.py not found. Will attempt /color/image_raw topic.")

    rclpy.init(args=args)
    node = ArmCameraScanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()