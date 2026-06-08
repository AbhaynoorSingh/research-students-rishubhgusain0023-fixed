#!/usr/bin/env python3
# coding=utf-8
"""
rrt_virtual_mover.py
--------------------
Yahboom ROSMASTER X3 PLUS — ROS2 Humble

RRT (Rapidly-exploring Random Tree) path planner + virtual robot mover.
Builds its own occupancy map from /scan (same as lidar_slam_planner.py),
plans a collision-free path using RRT, then virtually moves the robot
along that path.

How it works:
  1. Subscribes to /scan  → builds occupancy map in real time
  2. Subscribes to /odom_raw or /odom → tracks robot pose
  3. When a /goal arrives → runs RRT to find a path
  4. Virtually moves the robot along the RRT path:
       - broadcasts map -> base_footprint TF  (RViz shows robot moving)
       - publishes /odom_raw                  (feeds other nodes)
       - publishes /rrt_path                  (visualise in RViz)
       - publishes /virtual_cmd_vel           (what velocity would be sent)

Run order:
  Terminal 1:  ros2 launch ydlidar_ros2_driver ydlidar.launch.py   (LiDAR)
  Terminal 2:  python3 rrt_virtual_mover.py
  Terminal 3:  ros2 topic pub /goal geometry_msgs/Point "{x: 2.0, y: 1.5, z: 0.0}" --once

RViz:
  Fixed Frame : map
  Add         : /map, /rrt_path, /virtual_pose, /rrt_tree (MarkerArray)

RRT algorithm:
  - Samples random points in free space
  - Extends tree toward sample using step size STEP_SIZE
  - Checks collision along each new edge using the occupancy map
  - Uses RRT* rewiring for shorter paths (optional, see USE_RRT_STAR)
  - Goal bias: 10% of samples aim directly at the goal
"""

import math
import time
import random
import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSReliabilityPolicy,
                        QoSHistoryPolicy, QoSDurabilityPolicy)
import numpy as np
from scipy.ndimage import binary_dilation

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import (
    PoseStamped, Twist, TransformStamped,
    Quaternion, Point
)
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import TransformBroadcaster


# ───────────────────────── CONFIG ─────────────────────────────
# ───────────────────────── CONFIG ─────────────────────────────
# RRT parameters
MAX_ITERATIONS   = 5000
STEP_SIZE        = 0.30
GOAL_BIAS        = 0.10    # kept only as fallback
GOAL_TOLERANCE   = 0.5
INFLATION_M      = 0.20
USE_RRT_STAR     = True
RRT_STAR_RADIUS  = 1.0

# Map parameters
MAP_WIDTH_M      = 20.0
MAP_HEIGHT_M     = 20.0
MAP_RESOLUTION   = 0.05

# HMA-RRT* parameters  ← single authoritative block
CORRIDOR_FACTOR      = 0.4
MIN_CORRIDOR_WIDTH   = 2.0
GOAL_REGION_RADIUS   = 1.5
CORRIDOR_SAMPLE_RATE = 0.80
GOAL_SAMPLE_RATE     = 0.15
RANDOM_SAMPLE_RATE   = 0.05


# Grid-based sampling (Component 1 upgrade — Section 3.1)
GRID_N           = 10      # 10×10 grid over entire map
GRID_DELTA       = 0.1     # attenuation factor for selection count
GRID_P_BASE      = 0.05    # base probability every cell starts with

# Virtual movement
ROBOT_SPEED_MPS  = 0.3
WAYPOINT_TOLERANCE = 0.10
PUBLISH_HZ       = 20.0
# ──────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════
#  Occupancy Map 
# ══════════════════════════════════════════════════════════════
class OccupancyMap:
    L_OCC  =  0.85
    L_FREE = -0.40
    L_MAX  =  3.5
    L_MIN  = -3.5

    def __init__(self, width_m=MAP_WIDTH_M, height_m=MAP_HEIGHT_M,
                 resolution=MAP_RESOLUTION):
        self.res      = resolution
        self.w        = int(width_m  / resolution)
        self.h        = int(height_m / resolution)
        self.log_odds = np.full((self.w, self.h), -1.0, dtype=np.float32)
        self.origin_x = -width_m  / 2.0
        self.origin_y = -height_m / 2.0

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

    def _bresenham(self, x0, y0, x1, y1):
        dx, dy = abs(x1-x0), abs(y1-y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            yield x0, y0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x0 += sx
            if e2 < dx:
                err += dx; y0 += sy

    def update(self, scan, robot_x, robot_y, robot_yaw):
        rx, ry = self.world_to_cell(robot_x, robot_y)
        if not self.in_bounds(rx, ry):
            return
        angle = scan.angle_min
        for r in scan.ranges:
            angle += scan.angle_increment
            if r < scan.range_min or r > scan.range_max or \
               math.isnan(r) or math.isinf(r):
                continue
            hit_angle = robot_yaw + angle
            hx = robot_x + r * math.cos(hit_angle)
            hy = robot_y + r * math.sin(hit_angle)
            hcx, hcy = self.world_to_cell(hx, hy)
            for cx, cy in self._bresenham(rx, ry, hcx, hcy):
                if not self.in_bounds(cx, cy):
                    break
                self.log_odds[cx, cy] = max(
                    self.L_MIN, self.log_odds[cx, cy] + self.L_FREE)
            if self.in_bounds(hcx, hcy):
                self.log_odds[hcx, hcy] = min(
                    self.L_MAX, self.log_odds[hcx, hcy] + self.L_OCC)
        # Keep robot footprint clear
        clear_r = int(0.30 / self.res)
        for ddx in range(-clear_r, clear_r + 1):
            for ddy in range(-clear_r, clear_r + 1):
                if ddx*ddx + ddy*ddy <= clear_r*clear_r:
                    cx, cy = rx+ddx, ry+ddy
                    if self.in_bounds(cx, cy):
                        self.log_odds[cx, cy] = min(
                            self.log_odds[cx, cy], -0.5)

    def obstacle_mask(self):
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        return prob > 0.65

    def inflated_mask(self, inflation_m=INFLATION_M):
        cells = max(1, int(inflation_m / self.res))
        struct = np.ones((cells*2+1, cells*2+1), dtype=bool)
        return binary_dilation(self.obstacle_mask(), structure=struct)

    def to_ros_msg(self, frame_id="map"):
        grid = OccupancyGrid()
        grid.header.frame_id = frame_id
        grid.info.resolution = self.res
        grid.info.width      = self.w
        grid.info.height     = self.h
        grid.info.origin.position.x = self.origin_x
        grid.info.origin.position.y = self.origin_y
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        ros_data = (prob * 100).astype(np.int8)
        grid.data = ros_data.flatten(order='F').tolist()
        return grid

# ============================================================
# Quad RRT
# ============================================================
class QuadTreeNode:
    def __init__(self, x, y, node_index):
        self.x = x
        self.y = y
        self.node_index = node_index


class QuadTree:
    MAX_POINTS = 8

    def __init__(self, x, y, w, h, depth=0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.depth = depth

        self.points = []

        self.divided = False

        self.nw = None
        self.ne = None
        self.sw = None
        self.se = None

    def contains(self, x, y):
        return (
            self.x <= x < self.x + self.w and
            self.y <= y < self.y + self.h
        )

    def subdivide(self):
        hw = self.w / 2
        hh = self.h / 2

        self.nw = QuadTree(self.x, self.y, hw, hh, self.depth + 1)
        self.ne = QuadTree(self.x + hw, self.y, hw, hh, self.depth + 1)
        self.sw = QuadTree(self.x, self.y + hh, hw, hh, self.depth + 1)
        self.se = QuadTree(self.x + hw, self.y + hh, hw, hh, self.depth + 1)

        self.divided = True

    def insert(self, point):
        if not self.contains(point.x, point.y):
            return False

        if len(self.points) < self.MAX_POINTS:
            self.points.append(point)
            return True

        if not self.divided:
            self.subdivide()

        return (
            self.nw.insert(point) or
            self.ne.insert(point) or
            self.sw.insert(point) or
            self.se.insert(point)
        )

    def query_radius(self, x, y, radius, found=None):
        if found is None:
            found = []

        if not self._intersects_circle(x, y, radius):
            return found

        r2 = radius * radius

        for p in self.points:
            dx = p.x - x
            dy = p.y - y
            if dx*dx + dy*dy <= r2:
                found.append(p.node_index)

        if self.divided:
            self.nw.query_radius(x, y, radius, found)
            self.ne.query_radius(x, y, radius, found)
            self.sw.query_radius(x, y, radius, found)
            self.se.query_radius(x, y, radius, found)

        return found

    def nearest(self, x, y, best=None):

    # Check points in current node
        for p in self.points:
            d = (p.x - x)**2 + (p.y - y)**2

            if best is None or d < best[0]:
             best = (d, p.node_index)

    # If subdivided → search children intelligently
        if self.divided:

            children = [self.nw, self.ne, self.sw, self.se]

            # Sort children by proximity to query point
            children.sort(
                key=lambda child:
                child.distance_to_boundary(x, y)
            )

            for child in children:

                # If this quadrant cannot contain closer point → skip
                if best is not None:
                    min_possible_dist = child.distance_to_boundary(x, y)

                    if min_possible_dist > best[0]:
                        continue

                best = child.nearest(x, y, best)

        return best

    def _intersects_circle(self, x, y, r):
        nearest_x = max(self.x, min(x, self.x + self.w))
        nearest_y = max(self.y, min(y, self.y + self.h))

        dx = x - nearest_x
        dy = y - nearest_y

        return dx*dx + dy*dy <= r*r

    def distance_to_boundary(self, x, y):
        """
        Minimum squared distance from point (x,y)
        to this QuadTree region.
        """

        dx = 0.0
        dy = 0.0

        if x < self.x:
            dx = self.x - x
        elif x > self.x + self.w:
            dx = x - (self.x + self.w)

        if y < self.y:
            dy = self.y - y
        elif y > self.y + self.h:
            dy = y - (self.y + self.h)

        return dx*dx + dy*dy    




# ══════════════════════════════════════════════════════════════
#  RRT / RRT* Planner
# ══════════════════════════════════════════════════════════════
class RRTNode:
    __slots__ = ("x", "y", "parent", "cost")

    def __init__(self, x, y, parent=None, cost=0.0):
        self.x      = x
        self.y      = y
        self.parent = parent   # index into node list
        self.cost   = cost     # RRT* path cost from root


class QuadRRTPlanner:

    def __init__(self, occ_map: OccupancyMap):
        self.map = occ_map
        self.quadtree = None
        self.latest_nodes = []
        print("[INFO] QuadRRTPlanner initialized")
        # HMA-RRT* metrics
        self.plan_time = 0.0
        self.total_nodes = 0

        self.raw_path_length = 0.0
        self.smoothed_path_length = 0.0

        self.goal_samples = 0
        self.corridor_samples = 0
        self.global_samples = 0

        self.rewire_count = 0
        # Grid sampling state — reset each plan() call
        self.grid_probs  = {}   # normalized probability per (i,j) cell
        self.grid_counts = {}   # how many times each cell has been sampled
        self.grid_nc     = 0    # obstacle count intersecting start-goal line




    # ══════════════════════════════════════════════════════
#  Component 1 — Grid-Based Dynamic Sampling (Paper §3.1)
# ══════════════════════════════════════════════════════

    def _point_to_line_dist(self, px, py, x1, y1, x2, y2):
        """
        Perpendicular distance from point (px,py) to the
        finite line segment (x1,y1)→(x2,y2).
        Used in Equation 9 to compute d_ij for every grid cell.
        """
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1)*dx + (py - y1)*dy) / (dx*dx + dy*dy)
        t = max(0.0, min(1.0, t))
        return math.hypot(px - (x1 + t*dx), py - (y1 + t*dy))
    

    def _obstacle_ratio_in_cell(self, cx, cy, cw, ch, obs_mask):
        """
        Fraction of a grid cell covered by obstacle pixels.
        Returns A_ij used in Equation 11.
        cx, cy = world-coord bottom-left corner of cell
        cw, ch = cell width and height in metres
        """
        # Convert cell world corners to map pixel indices
        x0 = int((cx - self.map.origin_x) / self.map.res)
        y0 = int((cy - self.map.origin_y) / self.map.res)
        x1 = int((cx + cw - self.map.origin_x) / self.map.res)
        y1 = int((cy + ch - self.map.origin_y) / self.map.res)

        # Clamp to map bounds
        x0 = max(0, x0);  y0 = max(0, y0)
        x1 = min(obs_mask.shape[0] - 1, x1)
        y1 = min(obs_mask.shape[1] - 1, y1)

        if x1 <= x0 or y1 <= y0:
            return 0.0

        region = obs_mask[x0:x1, y0:y1]
        return float(np.sum(region)) / float(region.size + 1e-6)


    def _count_line_obstacle_intersections(self, sx, sy, gx, gy, obs_mask):
        """
        Count how many distinct obstacle clusters the straight
        start→goal line passes through.
        This gives n_c in Equation 7, used to compute σ_k (Eq. 8).
        """
        steps = int(math.hypot(gx - sx, gy - sy) / self.map.res)
        steps = max(steps, 1)
        prev_hit = False
        nc = 0
        for k in range(steps + 1):
            t  = k / steps
            wx = sx + t * (gx - sx)
            wy = sy + t * (gy - sy)
            cx, cy = self.map.world_to_cell(wx, wy)
            if self.map.in_bounds(cx, cy) and obs_mask[cx, cy]:
                if not prev_hit:
                    nc += 1       # entering a new obstacle cluster
                prev_hit = True
            else:
                prev_hit = False
        return nc
    
    def _init_sampling_grid(self, sx, sy, gx, gy, obs_mask):
        """
        Build the full n×n probability table once per plan() call.
        Implements Equations 7–13 from Section 3.1 of the paper.

        After this call, self.grid_probs[(i,j)] holds the normalized
        probability of sampling from cell (i,j).
        self.grid_counts is reset to zero for all cells.
        """
        n     = GRID_N
        ox    = self.map.origin_x          # world x of map left edge
        oy    = self.map.origin_y          # world y of map bottom edge
        mw    = self.map.w * self.map.res  # total map width  in metres
        mh    = self.map.h * self.map.res  # total map height in metres
        cw    = mw / n                     # single cell width  in metres
        ch    = mh / n                     # single cell height in metres

        L = math.hypot(gx - sx, gy - sy)  # start-goal line length

        # ── Eq. 7: count obstacle clusters on direct line ──
        nc    = self._count_line_obstacle_intersections(sx, sy, gx, gy, obs_mask)
        nmax  = max(1, nc)             # at least 1 to avoid div-by-zero

        # ── Eq. 8: steepness of distance-bias function ──
        # When nc is large (many obstacles on line), sigma_k → 0
        # meaning the distance falloff is gentle → samples spread wide
        # When nc is 0 (clear line), sigma_k = 1 → tight corridor
        sigma_k = 1.0 if nc == 0 else (1.0 - nc / (nmax + 1))

        raw_probs = {}
        self.grid_counts = {}

        for i in range(n):
            for j in range(n):
                # World coord of this cell's centre
                cell_cx = ox + (i + 0.5) * cw
                cell_cy = oy + (j + 0.5) * ch

                # ── Eq. 9: normalised perpendicular distance to line ──
                d_ij = self._point_to_line_dist(
                    cell_cx, cell_cy, sx, sy, gx, gy) / (L / 2.0 + 1e-6)

                # ── Eq. 10: distance-bias probability ──
                # Sigmoid: cells near the line get ~1.0, far cells get ~0.0
                P_line = 1.0 / (1.0 + math.exp(sigma_k * (d_ij - 1.0)))

                # ── Eq. 11: obstacle-area penalty ──
                # High obstacle ratio → low probability
                A_ij  = self._obstacle_ratio_in_cell(
                    ox + i*cw, oy + j*ch, cw, ch, obs_mask)
                P_area = math.exp(-A_ij)

                # ── Eq. 12: combined probability (no prior selections yet) ──
                # C_ij = 0 at start, so e^(-delta*0) = 1 → no attenuation yet
                C_ij  = 0
                P_ij  = (GRID_P_BASE + P_line * math.exp(-GRID_DELTA * C_ij)) \
                        * P_area

                raw_probs[(i, j)]    = max(P_ij, 1e-9)
                self.grid_counts[(i, j)] = 0

        # ── Eq. 13: normalise so all cells sum to 1.0 ──
        total = sum(raw_probs.values())
        self.grid_probs = {k: v / total for k, v in raw_probs.items()}

    def _sample_from_grid(self):
    # ── Explicit goal bias FIRST (replaces old GOAL_BIAS) ──
        if random.random() < GOAL_SAMPLE_RATE:   # 15%
            theta  = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, GOAL_REGION_RADIUS)
            self.goal_samples += 1
            return (
                self._plan_gx + radius * math.cos(theta),
                self._plan_gy + radius * math.sin(theta)
            )

        # ── Remaining 85% → grid-based sampling ──
        n  = GRID_N
        ox = self.map.origin_x
        oy = self.map.origin_y
        cw = (self.map.w * self.map.res) / n
        ch = (self.map.h * self.map.res) / n

        keys    = list(self.grid_probs.keys())
        weights = [self.grid_probs[k] for k in keys]
        chosen  = random.choices(keys, weights=weights, k=1)[0]
        i, j    = chosen

        self.grid_counts[(i, j)] += 1
        self.grid_probs[(i, j)]  *= math.exp(-GRID_DELTA)

        total = sum(self.grid_probs.values())
        if total < 0.5:
            self.grid_probs = {k: v/total for k, v in self.grid_probs.items()}

        # Check if this cell is near goal — count accordingly
        cell_cx = ox + (i + 0.5) * cw
        cell_cy = oy + (j + 0.5) * ch
        if math.hypot(cell_cx - self._plan_gx,
                    cell_cy - self._plan_gy) <= GOAL_REGION_RADIUS:
            self.goal_samples += 1
        else:
            self.corridor_samples += 1

        rx = ox + (i + random.random()) * cw
        ry = oy + (j + random.random()) * ch
        return rx, ry

    # ── public API ─────────────────────────────────────────────

    def adaptive_sample(
            self,
        sx, sy,
        gx, gy,
        wx_min, wx_max,
        wy_min, wy_max,
        nodes
        ):
        """
        HMA-RRT* Dynamic Region-Based Sampling

        Samples mostly inside a corridor between
        start and goal.
        """

        r = random.random()

        # ===================================================
        # Goal Region Sampling
        # ===================================================
        if r < GOAL_SAMPLE_RATE:

            theta = random.uniform(
                0.0,
                2.0 * math.pi
            )

            radius = random.uniform(
                0.0,
                GOAL_REGION_RADIUS
            )
            self.goal_samples += 1

            return (
                gx + radius * math.cos(theta),
                gy + radius * math.sin(theta)
            )

        # ===================================================
        # Corridor Sampling
        # ===================================================
        elif r < GOAL_SAMPLE_RATE + CORRIDOR_SAMPLE_RATE:

            dx = gx - sx
            dy = gy - sy

            dist = math.hypot(dx, dy)

            corridor_width = max(
                MIN_CORRIDOR_WIDTH,
                dist * CORRIDOR_FACTOR
            )

            t = random.random()

            base_x = sx + t * dx
            base_y = sy + t * dy

            if dist > 0.001:

                nx = -dy / dist
                ny = dx / dist

                offset = random.uniform(
                    -corridor_width / 2.0,
                    corridor_width / 2.0
                )

                sample_x = base_x + offset * nx
                sample_y = base_y + offset * ny

            else:

                sample_x = sx
                sample_y = sy

            sample_x = max(
                wx_min,
                min(sample_x, wx_max)
            )

            sample_y = max(
                wy_min,
                min(sample_y, wy_max)
            )
            self.corridor_samples += 1

            return sample_x, sample_y

        # ===================================================
        # Global Random Sampling
        # ===================================================
        else:
            self.global_samples += 1
            return (
                random.uniform(wx_min, wx_max),
                random.uniform(wy_min, wy_max)
            )

    def path_length(self, path):
        if len(path) < 2:
            return 0.0

        total = 0.0

        for i in range(len(path)-1):
            total += math.hypot(
                path[i+1][0] - path[i][0],
                path[i+1][1] - path[i][1]
            )

        return total


    def plan(self, start_world, goal_world):
        """
        Run RRT (or RRT* if USE_RRT_STAR=True).
        Returns list of (x, y) world-coord waypoints, or [] on failure.
        """
        self.goal_samples = 0
        self.corridor_samples = 0
        self.global_samples = 0

        self.rewire_count = 0

        self.plan_time = 0.0
        self.total_nodes = 0

        self.raw_path_length = 0.0
        self.smoothed_path_length = 0.0  

        obs = self.map.inflated_mask()

        start_time = time.time()

        # Map bounds in world coordinates
        wx_min = self.map.origin_x
        wx_max = self.map.origin_x + self.map.w * self.map.res

        wy_min = self.map.origin_y
        wy_max = self.map.origin_y + self.map.h * self.map.res

        # Create QuadTree
        self.quadtree = QuadTree(
            wx_min,
            wy_min,
            wx_max - wx_min,
            wy_max - wy_min
        )
        
        sx, sy = start_world
        gx, gy = goal_world

        self._plan_gx = gx
        self._plan_gy = gy  

        # Validate start / goal
        scx, scy = self.map.world_to_cell(sx, sy)
        gcx, gcy = self.map.world_to_cell(gx, gy)

        if not self.map.in_bounds(scx, scy):
            return []
        if not self.map.in_bounds(gcx, gcy) or obs[gcx, gcy]:
            gcx, gcy = self._free_near(gcx, gcy, obs)
            if gcx is None:
                return []
            gx, gy = self.map.cell_to_world(gcx, gcy)

        # Map bounds in world coords
        wx_min = self.map.origin_x
        wx_max = self.map.origin_x + self.map.w * self.map.res
        wy_min = self.map.origin_y
        wy_max = self.map.origin_y + self.map.h * self.map.res

        nodes = [RRTNode(sx, sy, parent=None, cost=0.0)]
        self.quadtree.insert(
        QuadTreeNode(sx, sy, 0)
        )
        goal_node_idx = None
        # ── Component 1: build grid probability table once ──
        self._init_sampling_grid(sx, sy, gx, gy, obs)
        

        for _ in range(MAX_ITERATIONS):
            # Sample
            
           # Sample — Component 1: grid-based dynamic sampling
            rx, ry = self._sample_from_grid()

            # Nearest node
            nearest_idx = self.quadtree.nearest(rx, ry)[1]
            nearest     = nodes[nearest_idx]

            # Steer
            nx, ny = self._steer(nearest.x, nearest.y, rx, ry)

            # Collision check
            if not self._collision_free(nearest.x, nearest.y, nx, ny, obs):
                continue

            new_cost = nearest.cost + math.hypot(nx - nearest.x,
                                                  ny - nearest.y)

            if USE_RRT_STAR:
                # RRT*: find neighbours and choose best parent
                new_node, parent_idx = self._choose_parent(
                    nodes, nx, ny, new_cost, obs)
                new_idx = len(nodes)
                nodes.append(new_node)
                self.quadtree.insert(
                QuadTreeNode(nx, ny, len(nodes)-1)
                )   
                # Rewire
                self._rewire(nodes, new_idx, obs)
            else:
                new_node = RRTNode(nx, ny, parent=nearest_idx, cost=new_cost)
                nodes.append(new_node)
                self.quadtree.insert(
                QuadTreeNode(nx, ny, len(nodes)-1)
                )


            # Goal check
            if ( math.hypot(nx-gx, ny-gy) <= GOAL_TOLERANCE and self._collision_free(nx, ny, gx, gy, obs)):
                goal_node_idx = len(nodes) - 1
                break

        if goal_node_idx is None:
            return []


        path = self._extract_path(nodes, goal_node_idx)
        raw_length = self.path_length(path)
        # Step 1: averaging smooth
        path = self.smooth_path(path)

        # Step 2: shortcut smooth (NEW)
        path = self.shortcut_smooth(path)
        self.latest_nodes = nodes
        elapsed = time.time() - start_time
        smooth_length = self.path_length(path)
        self.plan_time = elapsed
        self.total_nodes = len(nodes)

        self.raw_path_length = raw_length
        self.smoothed_path_length = smooth_length
        print(f"[RRT*] Time: {elapsed:.3f}s | Nodes: {len(nodes)}")

        print(
            f"[Sampling Stats] "
            f"Goal={self.goal_samples}, "
            f"Corridor={self.corridor_samples}, "
            f"Global={self.global_samples}"
        )
        print(
        f"[Optimization] "
        f"{raw_length:.2f}m -> {smooth_length:.2f}m"
        )
        print(f"[RRT*] Rewires: {self.rewire_count}")

        return path

    def get_tree_edges(self, nodes):
        """Return list of (x0,y0,x1,y1) for RViz tree visualisation."""
        edges = []
        for i, node in enumerate(nodes):
            if node.parent is not None:
                p = nodes[node.parent]
                edges.append((p.x, p.y, node.x, node.y))
        return edges

    # ── internal helpers ───────────────────────────────────────

    def _nearest(self, nodes, rx, ry):
        dists = [(math.hypot(n.x - rx, n.y - ry), i)
                 for i, n in enumerate(nodes)]
        return min(dists)[1]

    def _steer(self, fx, fy, tx, ty):
        d = math.hypot(tx - fx, ty - fy)
        if d <= STEP_SIZE:
            return tx, ty
        ratio = STEP_SIZE / d
        return fx + ratio * (tx - fx), fy + ratio * (ty - fy)

      

    def _collision_free(self, x0, y0, x1, y1, obs):
        """Check line segment for collisions using Bresenham."""
        cx0, cy0 = self.map.world_to_cell(x0, y0)
        cx1, cy1 = self.map.world_to_cell(x1, y1)
        for cx, cy in self.map._bresenham(cx0, cy0, cx1, cy1):
            if not self.map.in_bounds(cx, cy):
                return False
            if obs[cx, cy]:
                return False
        return True

    def _choose_parent(self, nodes, nx, ny, default_cost, obs):
        """RRT*: pick parent that gives lowest cost."""
        best_parent = None
        best_cost   = float('inf')
        nearby = self.quadtree.query_radius(
        nx, ny, RRT_STAR_RADIUS
        )

        radius = max(0.5,min(RRT_STAR_RADIUS,2.0 * math.sqrt(math.log(len(nodes)+1)/(len(nodes)+1))))
        for i in nearby:
            node = nodes[i]
            d = math.hypot(node.x - nx, node.y - ny)

            if d > radius:
                continue
            if not self._collision_free(node.x, node.y, nx, ny, obs):
                continue
            c = node.cost + d
            if c < best_cost:
                best_cost   = c
                best_parent = i
        if best_parent is None:
            # Fall back to nearest
            best_parent = self._nearest(nodes, nx, ny)
            p = nodes[best_parent]
            best_cost = p.cost + math.hypot(p.x - nx, p.y - ny)
        return RRTNode(nx, ny, parent=best_parent, cost=best_cost), best_parent

    def _rewire(self, nodes, new_idx, obs):
        """RRT*: check if routing through new_node shortens neighbour costs."""
        new_node = nodes[new_idx]
        nearby = self.quadtree.query_radius(
        new_node.x,
        new_node.y,
        RRT_STAR_RADIUS
        )

        radius = max(0.5,min(RRT_STAR_RADIUS,2.0 * math.sqrt(math.log(len(nodes)+1)/(len(nodes)+1))))
        for i in nearby:
            node = nodes[i]
            if i == new_idx or i == new_node.parent:
                continue
            d = math.hypot(node.x - new_node.x, node.y - new_node.y)

            if d > radius:
                continue
            new_cost = new_node.cost + d
            if new_cost < node.cost and \
               self._collision_free(new_node.x, new_node.y,
                                    node.x, node.y, obs):
                node.parent = new_idx
                self.rewire_count += 1
                node.cost   = new_cost
                self._update_children_costs(nodes, i)

    def _update_children_costs(self, nodes, parent_idx):
        """
        Recursively update costs of all descendants
        after a rewire operation.
        """

        parent = nodes[parent_idx]

        for i, node in enumerate(nodes):

            if node.parent == parent_idx:

                edge_cost = math.hypot(
                    node.x - parent.x,
                    node.y - parent.y
                )

                node.cost = parent.cost + edge_cost

                # Recursively update grandchildren
                self._update_children_costs(nodes, i)

    def _extract_path(self, nodes, goal_idx):
        path = []
        idx  = goal_idx
        while idx is not None:
            path.append((nodes[idx].x, nodes[idx].y))
            idx = nodes[idx].parent
        path.reverse()
        return path

    def _free_near(self, gcx, gcy, obs, search=20):
        """Find nearest free cell to a blocked goal cell."""
        for r in range(1, search):
            for ddx in range(-r, r+1):
                for ddy in range(-r, r+1):
                    nx, ny = gcx+ddx, gcy+ddy
                    if self.map.in_bounds(nx, ny) and not obs[nx, ny]:
                        return nx, ny
        return None, None

    @staticmethod
    def smooth_path(waypoints, iterations=50):
        """Simple path smoothing by averaging neighbours."""
        if len(waypoints) < 3:
            return waypoints
        pts = list(waypoints)
        for _ in range(iterations):
            for i in range(1, len(pts) - 1):
                pts[i] = (
                    (pts[i-1][0] + pts[i][0] + pts[i+1][0]) / 3.0,
                    (pts[i-1][1] + pts[i][1] + pts[i+1][1]) / 3.0,
                )
        return pts
   
    def shortcut_smooth(self, path, iterations=50):
        """Remove unnecessary waypoints by connecting distant nodes directly."""
        if len(path) < 3:
            return path

        import random
        new_path = list(path)

        obs = self.map.inflated_mask()
        for _ in range(iterations):
            i = random.randint(0, len(new_path) - 2)
            j = random.randint(i + 1, len(new_path) - 1)

            x1, y1 = new_path[i]
            x2, y2 = new_path[j]
            if self._collision_free(x1, y1, x2, y2, obs):
                new_path = new_path[:i+1] + new_path[j:]
        return new_path

# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════
def quat_from_yaw(yaw):
    return Quaternion(x=0.0, y=0.0,
                      z=math.sin(yaw/2.0),
                      w=math.cos(yaw/2.0))


# ══════════════════════════════════════════════════════════════
#  ROS2 Node
# ══════════════════════════════════════════════════════════════
class RRTVirtualMoverNode(Node):

    def __init__(self):
        super().__init__("rrt_virtual_mover")

        self.occ_map = OccupancyMap()
        self.planner = QuadRRTPlanner(self.occ_map)

        # Robot pose (virtual)
        self.x   = 0.0
        self.y   = 0.0
        self.yaw = 0.0

        # Path state
        self.waypoints    = []
        self.wp_index     = 0
        self.is_moving    = False
        self.rrt_nodes    = []   # kept for tree visualisation
        self._scan_count  = 0

        self.tf_broadcaster = TransformBroadcaster(self)

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        # Subscribers
        self.create_subscription(LaserScan, '/scan',     self._scan_cb,  sensor_qos)
        self.create_subscription(Odometry,  '/odom_raw', self._odom_cb,  sensor_qos)
        self.create_subscription(Odometry,  '/odom',     self._odom_cb,  sensor_qos)
        self.create_subscription(Point,     '/goal',     self._goal_cb,  10)

        # Publishers
        self.map_pub    = self.create_publisher(OccupancyGrid,  '/map',             10)
        self.path_pub   = self.create_publisher(Path,           '/rrt_path',        10)
        self.odom_pub   = self.create_publisher(Odometry,       '/odom_raw',        10)
        self.cmdvel_pub = self.create_publisher(Twist,          '/virtual_cmd_vel', 10)
        self.pose_pub   = self.create_publisher(PoseStamped,    '/virtual_pose',    10)
        self.tree_pub   = self.create_publisher(MarkerArray,    '/rrt_tree',        10)

        # Timers
        self.create_timer(1.0 / PUBLISH_HZ, self._control_loop)
        self.create_timer(5.0,              self._publish_map)

        self.get_logger().info("RRTVirtualMoverNode ready.")
        self.get_logger().info("Waiting for /scan to build map...")
        self.get_logger().info(
            'Send goal: ros2 topic pub /goal geometry_msgs/Point '
            '"{x: 2.0, y: 1.5, z: 0.0}" --once'
        )

    # ── callbacks ──────────────────────────────────────────────

    def _scan_cb(self, msg):
        self.occ_map.update(msg, self.x, self.y, self.yaw)
        self._scan_count += 1

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(
            2.0*(q.w*q.z + q.x*q.y),
            1.0 - 2.0*(q.y*q.y + q.z*q.z)
        )

    def _goal_cb(self, msg):
        goal = (msg.x, msg.y)
        self.get_logger().info(
            f"Goal received: ({msg.x:.2f}, {msg.y:.2f}) — running RRT...")

        if self._scan_count < 10:
            self.get_logger().warn(
                "Map not built yet — only got "
                f"{self._scan_count} scans. Move robot or wait.")

        t0 = time.time()
        path = self.planner.plan((self.x, self.y), goal)
        self.plan_metrics = {
        "time": self.planner.plan_time,
        "nodes": self.planner.total_nodes,

        "raw_length": self.planner.raw_path_length,
        "smooth_length": self.planner.smoothed_path_length,

        "goal_samples": self.planner.goal_samples,
        "corridor_samples": self.planner.corridor_samples,
        "global_samples": self.planner.global_samples,

        "rewires": self.planner.rewire_count
        }
        self.rrt_nodes = self.planner.latest_nodes
        elapsed = time.time() - t0

        if not path:
            self.get_logger().error(
                f"RRT failed to find path after {elapsed:.2f}s "
                f"({MAX_ITERATIONS} iterations). "
                "Try a closer goal or check for obstacles.")
            return

        self.waypoints = path
        self.wp_index  = 1
        self.is_moving = True

        length = self._path_length()
        self.get_logger().info(
            f"RRT path found in {elapsed:.2f}s | "
            f"{len(self.waypoints)} waypoints | "
            f"{length:.2f} m"
        )

        self._publish_rrt_path()

    # ── main control loop ──────────────────────────────────────

    def _control_loop(self):
        now = self.get_clock().now().to_msg()

        if (self.is_moving and self.waypoints and
                self.wp_index < len(self.waypoints)):

            tx, ty = self.waypoints[self.wp_index]
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            if dist < WAYPOINT_TOLERANCE:
                self.wp_index += 1
                if self.wp_index >= len(self.waypoints):
                    self._on_goal_reached()
                else:
                    pct = 100.0 * self.wp_index / len(self.waypoints)
                    self.get_logger().info(
                        f"  WP {self.wp_index}/{len(self.waypoints)} "
                        f"({pct:.0f}%) → "
                        f"next ({self.waypoints[self.wp_index][0]:.2f}, "
                        f"{self.waypoints[self.wp_index][1]:.2f})"
                    )
            else:
                target_yaw = math.atan2(dy, dx)
                step = min(ROBOT_SPEED_MPS / PUBLISH_HZ, dist)
                self.x   += step * math.cos(target_yaw)
                self.y   += step * math.sin(target_yaw)
                self.yaw  = target_yaw

                twist = Twist()
                twist.linear.x  = ROBOT_SPEED_MPS
                twist.angular.z = 0.0
                self.cmdvel_pub.publish(twist)

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
        t.transform.rotation = quat_from_yaw(self.yaw)
        self.tf_broadcaster.sendTransform(t)

    def _publish_odom(self, stamp):
        msg = Odometry()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id  = "base_footprint"
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.y = self.y
        msg.pose.pose.orientation = quat_from_yaw(self.yaw)
        msg.twist.twist.linear.x  = ROBOT_SPEED_MPS if self.is_moving else 0.0
        self.odom_pub.publish(msg)

    def _publish_pose(self, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = "map"
        msg.pose.position.x = self.x
        msg.pose.position.y = self.y
        msg.pose.orientation = quat_from_yaw(self.yaw)
        self.pose_pub.publish(msg)

    def _publish_rrt_path(self):
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for wx, wy in self.waypoints:
            ps = PoseStamped()
            ps.header = path_msg.header
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation = quat_from_yaw(0.0)
            path_msg.poses.append(ps)
        self.path_pub.publish(path_msg)

    def _publish_map(self):
        ros_map = self.occ_map.to_ros_msg(frame_id="map")
        ros_map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(ros_map)

    # ── helpers ────────────────────────────────────────────────

    def _on_goal_reached(self):
        self.is_moving = False
        twist = Twist()
        self.cmdvel_pub.publish(twist)
        goal = self.waypoints[-1]
        length = self._path_length()
        self.get_logger().info(
            f"\n{'='*50}\n"
            f"  GOAL REACHED via RRT{'*' if USE_RRT_STAR else ''}\n"
            f"  Goal           : ({goal[0]:.2f}, {goal[1]:.2f})\n"
            f"  Final position : ({self.x:.3f}, {self.y:.3f})\n"
            f"  Path length    : {length:.2f} m\n"
            f"  Waypoints      : {len(self.waypoints)}\n"
            f"{'='*50}"
        )

    def _path_length(self):
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array(self.waypoints)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ─────────────────────────── MAIN ─────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = RRTVirtualMoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()