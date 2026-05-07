#!/usr/bin/env python3
# coding=utf-8"""

import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan


# ──────────────────────────── CONFIG ──────────────────────────
MAP_WIDTH_M      = 20.0
MAP_HEIGHT_M     = 20.0
MAP_RESOLUTION   = 0.05    # metres per cell
EXPLORED_THRESH  = 0.20    # log-odds threshold to count cell as "seen"
# ──────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  Output dataclass
# ══════════════════════════════════════════════════════════════
@dataclass
class SLAMResult:
    """
    Output of slam.update(observation).
    All fields are populated every step.
    """
    pose_est:       Tuple[float, float, float]  = (0.0, 0.0, 0.0)  # (x, y, yaw)
    occupancy_grid: np.ndarray                  = field(default_factory=lambda: np.array([]))
    explored_ratio: float                       = 0.0
    map_msg:        Optional[OccupancyGrid]     = None
    step:           int                         = 0
    elapsed_s:      float                       = 0.0


# ══════════════════════════════════════════════════════════════
#  Internal occupancy map  (log-odds Bayesian update)
# ══════════════════════════════════════════════════════════════
class _OccupancyMap:
    L_OCC = 0.85
    L_FREE = -0.40
    L_MAX = 3.5
    L_MIN = -3.5

    def __init__(self, width_m, height_m, resolution):
        self.res      = resolution
        self.w        = int(width_m  / resolution)
        self.h        = int(height_m / resolution)
        self.origin_x = -width_m  / 2.0
        self.origin_y = -height_m / 2.0
        self._reset_grid()
        self.add_test_obstacles()

    def _reset_grid(self):
        # -1.0 = completely unknown
        self.log_odds = np.full((self.w, self.h), -1.0, dtype=np.float32)

    def add_test_obstacles(self):
        """Add artificial obstacles for RRT testing."""

        # Vertical wall
        for x in range(180, 190):
            for y in range(120, 280):
                self.log_odds[x, y] = self.L_MAX

        # Small block obstacle
        for x in range(250, 280):
            for y in range(200, 230):
                self.log_odds[x, y] = self.L_MAX

    def reset(self):
        self._reset_grid()

    # ── coordinate transforms ──────────────────────────────────

    def world_to_cell(self, wx, wy):
        cx = int((wx - self.origin_x) / self.res)
        cy = int((wy - self.origin_y) / self.res)
        return cx, cy

    def cell_to_world(self, cx, cy):
        wx = cx * self.res + self.origin_x + self.res / 2.0
        wy = cy * self.res + self.origin_y + self.res / 2.0
        return wx, wy

    def in_bounds(self, cx, cy):
        return 0 <= cx < self.w and 0 <= cy < self.h

    # ── Bresenham ray cast ─────────────────────────────────────

    def _bresenham(self, x0, y0, x1, y1):
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0  += sx
            if e2 < dx:
                err += dx
                y0  += sy

    # ── map update from a single scan ─────────────────────────

    def update(self, scan: LaserScan, robot_x, robot_y, robot_yaw):
        rx, ry = self.world_to_cell(robot_x, robot_y)
        if not self.in_bounds(rx, ry):
            return

        angle = scan.angle_min
        for r in scan.ranges:
            angle += scan.angle_increment
            if (r < scan.range_min or r > scan.range_max
                    or math.isnan(r) or math.isinf(r)):
                continue

            hit_angle = robot_yaw + angle
            hx = robot_x + r * math.cos(hit_angle)
            hy = robot_y + r * math.sin(hit_angle)
            hcx, hcy = self.world_to_cell(hx, hy)

            # Mark free cells along ray
            for cx, cy in self._bresenham(rx, ry, hcx, hcy):
                if not self.in_bounds(cx, cy):
                    break
                self.log_odds[cx, cy] = max(
                    self.L_MIN, self.log_odds[cx, cy] + self.L_FREE)

            # Mark hit cell as occupied
            if self.in_bounds(hcx, hcy):
                self.log_odds[hcx, hcy] = min(
                    self.L_MAX, self.log_odds[hcx, hcy] + self.L_OCC)

        # Keep robot footprint clear (avoids self-blocking)
        clear_r = int(0.30 / self.res)
        for ddx in range(-clear_r, clear_r + 1):
            for ddy in range(-clear_r, clear_r + 1):
                if ddx * ddx + ddy * ddy <= clear_r * clear_r:
                    cx, cy = rx + ddx, ry + ddy
                    if self.in_bounds(cx, cy):
                        self.log_odds[cx, cy] = min(
                            self.log_odds[cx, cy], -0.5)

    # ── derived properties ─────────────────────────────────────

    def probability_grid(self) -> np.ndarray:
        """Return [0,1] probability grid: 1.0 = definitely occupied."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
      
    def inflated_mask(self):
        return self.obstacle_mask()

    def obstacle_mask(self) -> np.ndarray:
        return self.probability_grid() > 0.65

    def explored_ratio(self, threshold=EXPLORED_THRESH) -> float:
        """
        Fraction of cells that have been observed at least once.
        A cell is considered explored when its log-odds has moved
        away from the initial unknown value (-1.0) by > threshold.
        """
        explored = np.abs(self.log_odds - (-1.0)) > threshold
        return float(np.sum(explored)) / float(self.w * self.h)

    def to_ros_msg(self, stamp=None, frame_id="map") -> OccupancyGrid:
        grid             = OccupancyGrid()
        grid.header.frame_id = frame_id
        if stamp is not None:
            grid.header.stamp = stamp
        grid.info.resolution = self.res
        grid.info.width      = self.w
        grid.info.height     = self.h
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        prob     = self.probability_grid()
        ros_data = (prob * 100).astype(np.int8)
        grid.data = ros_data.flatten(order='F').tolist()
        return grid


# ══════════════════════════════════════════════════════════════
#  Internal pose tracker  (from odometry)
# ══════════════════════════════════════════════════════════════
class _PoseTracker:

    def __init__(self):
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

    def reset(self):
        self.x = self.y = self.yaw = 0.0

    def update(self, odom: Odometry):
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    @property
    def pose(self) -> Tuple[float, float, float]:
        return self.x, self.y, self.yaw


# ══════════════════════════════════════════════════════════════
#  Public SLAM Module
# ══════════════════════════════════════════════════════════════
class SLAMModule:
    """
    Drop-in SLAM module for VLN-15.

    Example
    -------
    slam = SLAMModule()
    slam.reset()

    for step in episode:
        obs    = {"depth": scan_msg, "odom": odom_msg, "action": "forward"}
        result = slam.update(obs)

        x, y, yaw       = result.pose_est
        grid            = result.occupancy_grid   # np.ndarray [0,1]
        explored        = result.explored_ratio   # 0.0 – 1.0
        map_ros         = result.map_msg          # publish to /map
    """

    def __init__(self,
                 width_m: float      = MAP_WIDTH_M,
                 height_m: float     = MAP_HEIGHT_M,
                 resolution: float   = MAP_RESOLUTION):

        self._map   = _OccupancyMap(width_m, height_m, resolution)
        self._pose  = _PoseTracker()
        self._step  = 0
        self._t0    = time.time()

    # ── public interface ───────────────────────────────────────

    def reset(self):
        """
        Reset SLAM state to initial conditions.
        Call at the start of every new episode.
        """
        self._map.reset()
        self._pose.reset()
        self._step = 0
        self._t0   = time.time()

    def update(self, observation: dict) -> SLAMResult:
        """
        Process one observation step and return updated SLAM state.

        Parameters
        ----------
        observation : dict with keys:
            "depth"  : sensor_msgs/LaserScan  (required)
            "odom"   : nav_msgs/Odometry      (required)
            "action" : str                    (optional, for logging)

        Returns
        -------
        SLAMResult with pose_est, occupancy_grid, explored_ratio, map_msg
        """
        scan: LaserScan  = observation.get("depth")
        odom: Odometry   = observation.get("odom")
        action: str      = observation.get("action", "unknown")

        if scan is None or odom is None:
            raise ValueError(
                "slam.update() requires observation dict with "
                "'depth' (LaserScan) and 'odom' (Odometry) keys."
            )

        # 1. Update pose from odometry
        self._pose.update(odom)
        x, y, yaw = self._pose.pose

        # 2. Update occupancy map from LiDAR scan
        self._map.update(scan, x, y, yaw)

        # 3. Compute outputs
        prob_grid      = self._map.probability_grid()
        explored       = self._map.explored_ratio()
        map_ros        = self._map.to_ros_msg(
            stamp=odom.header.stamp, frame_id="map")

        self._step += 1

        return SLAMResult(
            pose_est       = (x, y, yaw),
            occupancy_grid = prob_grid,
            explored_ratio = explored,
            map_msg        = map_ros,
            step           = self._step,
            elapsed_s      = time.time() - self._t0,
        )

    # ── convenience properties ─────────────────────────────────

    @property
    def pose(self) -> Tuple[float, float, float]:
        """Current pose estimate (x, y, yaw)."""
        return self._pose.pose

    @property
    def occupancy_grid(self) -> np.ndarray:
        """Current probability grid as numpy array."""
        return self._map.probability_grid()

    @property
    def explored_ratio(self) -> float:
        """Fraction of map explored so far."""
        return self._map.explored_ratio()

    @property
    def step(self) -> int:
        """Number of update steps since last reset."""
        return self._step

    def get_map_msg(self, stamp=None, frame_id="map") -> OccupancyGrid:
        """Get current map as a ROS2 OccupancyGrid message."""
        return self._map.to_ros_msg(stamp=stamp, frame_id=frame_id)
