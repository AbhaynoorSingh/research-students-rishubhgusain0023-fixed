#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import numpy as np
import heapq
import math
import time
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, Quaternion, TransformStamped
from tf2_ros import TransformBroadcaster


class OccupancyMap:
    L_OCC  =  0.85
    L_FREE = -0.40
    L_MAX  =  3.5
    L_MIN  = -3.5

    def __init__(self, width_m=20.0, height_m=20.0, resolution=0.05):
        self.res  = resolution
        self.w    = int(width_m  / resolution)
        self.h    = int(height_m / resolution)
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

    def update(self, scan, robot_x, robot_y, robot_yaw):
        rx, ry = self.world_to_cell(robot_x, robot_y)
        if not self.in_bounds(rx, ry):
            return
        angle = scan.angle_min
        for r in scan.ranges:
            angle += scan.angle_increment
            if r < scan.range_min or r > scan.range_max or math.isnan(r) or math.isinf(r):
                continue
            hit_angle = robot_yaw + angle
            hx = robot_x + r * math.cos(hit_angle)
            hy = robot_y + r * math.sin(hit_angle)
            hcx, hcy = self.world_to_cell(hx, hy)
            for (cx, cy) in self._bresenham(rx, ry, hcx, hcy):
                if not self.in_bounds(cx, cy):
                    break
                self.log_odds[cx, cy] = max(self.L_MIN, self.log_odds[cx, cy] + self.L_FREE)
            if self.in_bounds(hcx, hcy):
                self.log_odds[hcx, hcy] = min(self.L_MAX, self.log_odds[hcx, hcy] + self.L_OCC)
        # Keep robot footprint clear
        clear_r = int(0.30 / self.res)
        for dx in range(-clear_r, clear_r + 1):
            for dy in range(-clear_r, clear_r + 1):
                if dx*dx + dy*dy <= clear_r*clear_r:
                    cx, cy = rx + dx, ry + dy
                    if self.in_bounds(cx, cy):
                        self.log_odds[cx, cy] = min(self.log_odds[cx, cy], -0.5)

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

    def obstacle_mask(self):
        prob = 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
        return prob > 0.65


class GridPathPlanner:
    DIRECTIONS = [
        ( 1,  0, 1.0), (-1,  0, 1.0),
        ( 0,  1, 1.0), ( 0, -1, 1.0),
        ( 1,  1, 1.414), ( 1, -1, 1.414),
        (-1,  1, 1.414), (-1, -1, 1.414),
    ]

    def __init__(self, occ_map, inflation_radius_m=0.15):
        self.occ_map = occ_map
        self.inflation_cells = max(1, int(inflation_radius_m / occ_map.res))

    def _inflate(self, mask):
        from scipy.ndimage import binary_dilation
        struct = np.ones((self.inflation_cells * 2 + 1,
                          self.inflation_cells * 2 + 1), dtype=bool)
        return binary_dilation(mask, structure=struct)

    def _heuristic(self, a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return max(dx, dy) + (1.414 - 1) * min(dx, dy)

    def plan(self, start_world, goal_world, robot_cx, robot_cy):
        occ = self.occ_map
        obstacle = self._inflate(occ.obstacle_mask())
        clear_r = int(0.30 / occ.res)
        for dx in range(-clear_r, clear_r + 1):
            for dy in range(-clear_r, clear_r + 1):
                if dx*dx + dy*dy <= clear_r*clear_r:
                    cx, cy = robot_cx + dx, robot_cy + dy
                    if occ.in_bounds(cx, cy):
                        obstacle[cx, cy] = False
        sx, sy = occ.world_to_cell(*start_world)
        gx, gy = occ.world_to_cell(*goal_world)
        if not (occ.in_bounds(sx, sy) and occ.in_bounds(gx, gy)):
            return []
        if obstacle[gx, gy]:
            found = False
            for r in range(1, 20):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = gx+dx, gy+dy
                        if occ.in_bounds(nx, ny) and not obstacle[nx, ny]:
                            gx, gy = nx, ny
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if not found:
                return []
        open_set = []
        heapq.heappush(open_set, (0.0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0.0}
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == (gx, gy):
                path_cells = []
                while current in came_from:
                    path_cells.append(current)
                    current = came_from[current]
                path_cells.append((sx, sy))
                path_cells.reverse()
                return [occ.cell_to_world(c[0], c[1]) for c in path_cells]
            for dx, dy, cost in self.DIRECTIONS:
                nb = (current[0] + dx, current[1] + dy)
                if not occ.in_bounds(*nb) or obstacle[nb[0], nb[1]]:
                    continue
                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(nb, float('inf')):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f = tentative_g + self._heuristic(nb, (gx, gy))
                    heapq.heappush(open_set, (f, nb))
        return []

    def smooth_path(self, waypoints, window=5):
        if len(waypoints) < window:
            return waypoints
        pts = np.array(waypoints)
        smoothed = []
        for i in range(len(pts)):
            lo = max(0, i - window // 2)
            hi = min(len(pts), i + window // 2 + 1)
            smoothed.append(pts[lo:hi].mean(axis=0).tolist())
        return smoothed


class PoseTracker:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

    def update_from_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.x = p.x
        self.y = p.y
        self.yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    @property
    def pose(self):
        return self.x, self.y, self.yaw


class NavigationMetrics:
    def __init__(self, goal_x, goal_y, tolerance_m=0.3):
        self.goal = np.array([goal_x, goal_y])
        self.tolerance = tolerance_m
        self.planned_dist = 0.0
        self.actual_dist = 0.0
        self._prev_pos = None
        self.reached = False
        self.step_count = 0
        self.start_time = time.time()

    def update(self, robot_x, robot_y):
        pos = np.array([robot_x, robot_y])
        if self._prev_pos is not None:
            self.actual_dist += np.linalg.norm(pos - self._prev_pos)
        self._prev_pos = pos.copy()
        self.step_count += 1
        if np.linalg.norm(pos - self.goal) < self.tolerance:
            self.reached = True

    def set_planned_path(self, waypoints):
        pts = np.array(waypoints)
        if len(pts) > 1:
            self.planned_dist = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def report(self):
        elapsed = time.time() - self.start_time
        nav_error = float(np.linalg.norm(self._prev_pos - self.goal)) if self._prev_pos is not None else 0.0
        spl = self.planned_dist / max(self.planned_dist, self.actual_dist) if self.reached and self.planned_dist > 0 else 0.0
        print("\n--- Navigation Metrics ---")
        print("  Goal Reached     : " + ("YES" if self.reached else "NO"))
        print("  Navigation Error : {:.3f} m".format(nav_error))
        print("  Planned Distance : {:.2f} m".format(self.planned_dist))
        print("  Actual Distance  : {:.2f} m".format(self.actual_dist))
        print("  SPL              : {:.3f}".format(spl))
        print("  Elapsed Time     : {:.1f} s".format(elapsed))
        print("--------------------------\n")


class LidarSLAMPlannerNode(Node):

    def __init__(self):
        super().__init__('lidar_slam_planner')

        self.occ_map  = OccupancyMap(width_m=20.0, height_m=20.0, resolution=0.05)
        self.planner  = GridPathPlanner(self.occ_map, inflation_radius_m=0.15)
        self.pose_trk = PoseTracker()
        self.metrics  = None
        self.current_path = []
        self.goal = None
        self._scan_count = 0
        self._last_replan_time = 0.0

        # TF broadcaster: publishes map -> base_link so RViz links all frames
        self.tf_broadcaster = TransformBroadcaster(self)

        # BEST_EFFORT QoS to match YDLiDAR driver
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(LaserScan, '/scan', self._scan_cb, sensor_qos)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, sensor_qos)
        self.create_subscription(Point,     '/goal', self._goal_cb, 10)

        self.map_pub  = self.create_publisher(OccupancyGrid, '/map',          10)
        self.path_pub = self.create_publisher(Path,          '/planned_path', 10)
        self.pose_pub = self.create_publisher(PoseStamped,   '/slam_pose',    10)

        self.create_timer(5.0, self._publish_map)
        self.create_timer(1.0, self._publish_path)
        self.create_timer(0.1, self._broadcast_tf)  # 10Hz TF

        self.get_logger().info("LidarSLAMPlannerNode ready.")
        self.get_logger().info("Fixed Frame in RViz should be: map")
        self.get_logger().info('Send goal: ros2 topic pub /goal geometry_msgs/Point "{x: 1.0, y: 0.5, z: 0.0}" --once')

    def _broadcast_tf(self):
        """
        Broadcast map -> base_link transform at 10Hz.
        This links the map frame to the robot frame so RViz
        can display the scan, path and map together.
        """
        x, y, yaw = self.pose_trk.pose
        now = self.get_clock().now().to_msg()

        t = TransformStamped()
        t.header.stamp    = now
        t.header.frame_id = "map"
        t.child_frame_id  = "base_link"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = math.sin(yaw / 2.0)
        t.transform.rotation.w = math.cos(yaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def _odom_cb(self, msg):
        self.pose_trk.update_from_odom(msg)
        self._publish_slam_pose()
        if self.metrics:
            x, y, _ = self.pose_trk.pose
            self.metrics.update(x, y)
            if self.metrics.reached:
                self.get_logger().info("Goal reached!")
                self.metrics.report()
                self.metrics = None
                self.goal = None
                self.current_path = []

    def _scan_cb(self, msg):
        x, y, yaw = self.pose_trk.pose
        self.occ_map.update(msg, x, y, yaw)
        self._scan_count += 1
        if self.goal and (self._scan_count % 20 == 0):
            now = time.time()
            if now - self._last_replan_time > 2.0:
                self._replan()
                self._last_replan_time = now

    def _goal_cb(self, msg):
        self.goal = (msg.x, msg.y)
        self.get_logger().info("New goal: ({:.2f}, {:.2f})".format(msg.x, msg.y))
        self.metrics = NavigationMetrics(msg.x, msg.y, tolerance_m=0.3)
        x, y, _ = self.pose_trk.pose
        self.metrics.update(x, y)
        self._replan()

    def _replan(self):
        if self.goal is None:
            return
        x, y, _ = self.pose_trk.pose
        rx, ry = self.occ_map.world_to_cell(x, y)
        raw_path = self.planner.plan((x, y), self.goal, rx, ry)
        if not raw_path:
            self.get_logger().warn("A* found no path. Will retry on next scan.")
            return
        self.current_path = self.planner.smooth_path(raw_path)
        if self.metrics:
            self.metrics.set_planned_path(self.current_path)
        self.get_logger().info("Path found: {} waypoints, {:.2f}m".format(
            len(self.current_path),
            self.metrics.planned_dist if self.metrics else 0.0))

    def _publish_map(self):
        ros_map = self.occ_map.to_ros_msg(frame_id="map")
        ros_map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(ros_map)

    def _publish_path(self):
        if not self.current_path:
            return
        path_msg = Path()
        path_msg.header.stamp    = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"
        for wx, wy in self.current_path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.0
            pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

    def _publish_slam_pose(self):
        x, y, yaw = self.pose_trk.pose
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation = Quaternion(
            x=0.0, y=0.0,
            z=math.sin(yaw / 2.0),
            w=math.cos(yaw / 2.0)
        )
        self.pose_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarSLAMPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node.metrics:
            node.metrics.report()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()