#!/usr/bin/env python3
# coding=utf-8
"""
arm_camera_scan.py
------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble
Sweeps the arm base servo (ID 1) across a configurable arc and
captures an image from /color/image_raw at each position.
 
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
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
 
# OpenCV bridge
try:
    from cv_bridge import CvBridge
    import cv2
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False
 
# Yahboom driver
try:
    from Rosmaster_Lib import Rosmaster
    ARM_AVAILABLE = True
except ImportError:
    ARM_AVAILABLE = False
 
 
# ─────────────────────────── CONFIG ───────────────────────────
SCAN_POSITIONS   = [45, 90, 135]   # degrees for servo 1 (base)
MOVE_DELAY       = 1.5             # seconds to wait after each move
CAPTURE_PER_POS  = 1               # images to capture per position
SAVE_DIR         = os.path.expanduser("~/scan_images")
CAMERA_TOPIC     = "/color/image_raw"
# ──────────────────────────────────────────────────────────────
 
 
class ArmCameraScanNode(Node):
 
    def __init__(self):
        super().__init__("arm_camera_scan")
 
        os.makedirs(SAVE_DIR, exist_ok=True)
 
        self.bridge      = CvBridge() if CV_AVAILABLE else None
        self.latest_img  = None
        self.scan_pub    = self.create_publisher(Image,  "/scan_image",  10)
        self.status_pub  = self.create_publisher(String, "/scan_status", 10)
 
        # Camera QoS — match Astra camera driver (BEST_EFFORT)
        cam_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, CAMERA_TOPIC, self._image_cb, cam_qos)
 
        # Init arm
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
 
        # Let camera warm up, then start scan
        self.get_logger().info("Waiting 2 s for camera to warm up...")
        self.scan_timer = self.create_timer(2.0, self._start_scan)
 
    # ── callbacks ──────────────────────────────────────────────
 
    def _image_cb(self, msg: Image):
        self.latest_img = msg
 
    def _start_scan(self):
        """Called once after warm-up delay."""
        self.scan_timer.cancel()
        self.get_logger().info("Starting arm scan...")
        self._run_scan()
 
    # ── core scan logic ────────────────────────────────────────
 
    def _move_servo(self, servo_id: int, angle: int):
        """Move a single servo and wait for it to settle."""
        if self.bot:
            self.bot.set_uart_servo_angle(servo_id, angle)
        else:
            self.get_logger().info(f"[SIM] Servo {servo_id} → {angle}°")
        time.sleep(MOVE_DELAY)
 
    def _capture_image(self, angle: int, shot: int) -> bool:
        """
        Grab the latest image from the camera topic and save it.
        Returns True on success.
        """
        if self.latest_img is None:
            self.get_logger().warn(f"No image available at {angle}°")
            return False
 
        filename = os.path.join(SAVE_DIR, f"scan_{angle:03d}deg_{shot}.jpg")
 
        if CV_AVAILABLE and self.bridge:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(self.latest_img, "bgr8")
                cv2.imwrite(filename, cv_img)
                self.get_logger().info(f"Saved: {filename}")
            except Exception as e:
                self.get_logger().error(f"cv_bridge error: {e}")
                return False
        else:
            # Fallback: save raw bytes (useful for debugging)
            with open(filename.replace(".jpg", ".raw"), "wb") as f:
                f.write(bytes(self.latest_img.data))
            self.get_logger().info(f"Saved raw: {filename.replace('.jpg','.raw')}")
 
        # Re-publish for RViz / other nodes
        self.scan_pub.publish(self.latest_img)
        return True
 
    def _publish_status(self, text: str):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)
        self.get_logger().info(text)
 
    def _run_scan(self):
        """Full sweep: move arm to each position, capture image."""
        self._publish_status("Scan started")
 
        results = {}
 
        for angle in SCAN_POSITIONS:
            self._publish_status(f"Moving to {angle}°")
            self._move_servo(1, angle)   # servo 1 = base (pan)
 
            # Read back actual angle for logging
            if self.bot:
                actual = self.bot.get_uart_servo_angle(1)
                self.get_logger().info(f"Readback servo 1: {actual}°")
 
            # Capture
            for shot in range(CAPTURE_PER_POS):
                # Spin briefly so ROS2 callbacks can deliver the latest frame
                rclpy.spin_once(self, timeout_sec=0.3)
                ok = self._capture_image(angle, shot)
                results[angle] = "OK" if ok else "FAILED"
 
        # Return arm to home position
        self._publish_status("Returning arm to home (90°)")
        self._move_servo(1, 90)
 
        # Summary
        self._publish_status("Scan complete!")
        for ang, status in results.items():
            self.get_logger().info(f"  {ang:3d}° → {status}")
        self.get_logger().info(f"Images saved to: {SAVE_DIR}")
 
        # Shutdown node cleanly
        rclpy.shutdown()
 
    def destroy_node(self):
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
 