import numpy as np

UNKNOWN, FREE, OCC = -1, 0, 1

class SlamModule:
    """
    Baseline SLAM:
      - Pose update uses odom (dead-reckoning).
      - Occupancy grid starts as UNKNOWN and becomes FREE where depth is valid.
      - explored_ratio = known_cells / total_cells
    """

    def __init__(self, map_size_m=20.0, resolution=0.05):
        self.map_size_m = float(map_size_m)
        self.res = float(resolution)
        self.size = int(self.map_size_m / self.res)
        self.reset()

    def reset(self):
        self.occ = np.full((self.size, self.size), UNKNOWN, dtype=np.int8)
        self.x, self.y, self.yaw = 0.0, 0.0, 0.0

    def update(self, observation):
        """
        observation must contain:
          - observation["odom"] : (dx, dy, dyaw)  (delta in robot frame)  OR
                               : {"dx":..,"dy":..,"dyaw":..}
          - observation["depth"]: (H,W) depth array in meters (optional for baseline)
        """
        odom = observation.get("odom", None)
        if odom is not None:
            dx, dy, dyaw = self._parse_odom(odom)
            # integrate odom in world frame
            c, s = np.cos(self.yaw), np.sin(self.yaw)
            self.x += c * dx - s * dy
            self.y += s * dx + c * dy
            self.yaw += dyaw

        depth = observation.get("depth", None)
        if depth is not None:
            self._mark_explored_from_depth(depth)

    def get_outputs(self):
        explored = np.count_nonzero(self.occ != UNKNOWN)
        explored_ratio = explored / self.occ.size
        pose_est = (float(self.x), float(self.y), float(self.yaw))
        return pose_est, self.occ, float(explored_ratio)

    # ---------------- helpers ----------------

    def _parse_odom(self, odom):
        if isinstance(odom, dict):
            return float(odom.get("dx", 0.0)), float(odom.get("dy", 0.0)), float(odom.get("dyaw", 0.0))
        # assume tuple/list/np array
        return float(odom[0]), float(odom[1]), float(odom[2])

    def _world_to_grid(self, X, Y):
        cx = self.size // 2
        cy = self.size // 2
        gx = int(cx + X / self.res)
        gy = int(cy + Y / self.res)
        gx = int(np.clip(gx, 0, self.size - 1))
        gy = int(np.clip(gy, 0, self.size - 1))
        return gx, gy

    def _mark_explored_from_depth(self, depth):
        """
        Baseline: if depth has ANY valid pixels, mark the robot's current cell as FREE.
        This guarantees explored_ratio grows as the robot moves.
        Later you'll replace this with raycasting + occupied endpoints.
        """
        d = np.asarray(depth)
        if d.size == 0:
            return
        # valid depth = >0 and finite
        valid = np.isfinite(d) & (d > 0)
        if not np.any(valid):
            return

        gx, gy = self._world_to_grid(self.x, self.y)
        self.occ[gy, gx] = FREE