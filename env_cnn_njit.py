import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit

WIN_REWARD = 1e2

@njit
def is_in_wall_jit(pos, walls):
    # Border check
    if pos[0] < 0 or pos[0] > 10 or pos[1] < 0 or pos[1] > 10:
        return 2
    # Internal walls check
    for i in range(len(walls)):
        w = walls[i]
        if w[0] < pos[0] < w[1] and w[2] < pos[1] < w[3]:
            return 1
    return 0

@njit
def cross_2d_jit(v, w):
    return v[0] * w[1] - v[1] * w[0]

@njit
def segments_intersect_jit(A, B, C, D):
    r = B - A
    s = D - C
    r_cross_s = cross_2d_jit(r, s)
    
    if abs(r_cross_s) < 1e-8:
        return False

    q_minus_p = C - A
    t = cross_2d_jit(q_minus_p, s) / r_cross_s
    u = cross_2d_jit(q_minus_p, r) / r_cross_s

    return (0 <= t <= 1) and (0 <= u <= 1)

@njit
def has_line_of_sight_jit(start_pos, target_pos, walls):
    for i in range(len(walls)):
        w = walls[i]
        # Wall segments: Bottom, Right, Top, Left
        c1 = np.array([w[0], w[2]]) # BL
        c2 = np.array([w[1], w[2]]) # BR
        c3 = np.array([w[1], w[3]]) # TR
        c4 = np.array([w[0], w[3]]) # TL
        
        if (segments_intersect_jit(start_pos, target_pos, c1, c2) or
            segments_intersect_jit(start_pos, target_pos, c2, c3) or
            segments_intersect_jit(start_pos, target_pos, c3, c4) or
            segments_intersect_jit(start_pos, target_pos, c4, c1)):
            return False
    return True

@njit
def move_entity_jit(pos, target, speed, walls):
    direction = target - pos
    dist = np.linalg.norm(direction)
    if dist < speed:
        return target, True
    
    new_pos = pos + (direction / dist) * speed
    if is_in_wall_jit(new_pos, walls) == 0:
        return new_pos, False
    return pos, True

@njit
def clip_scalar(val, low, high):
    """Numba-friendly scalar clipping."""
    if val < low:
        return low
    if val > high:
        return high
    return val

@njit
def pos_to_px_jit(pos, field_size, grid_size):
    """Converts continuous coordinates to grid indices safely."""
    # We use our custom clip instead of np.clip for scalars
    x_val = (pos[0] / field_size) * grid_size
    y_val = (pos[1] / field_size) * grid_size
    
    px = int(clip_scalar(x_val, 0, grid_size - 1))
    py = int(clip_scalar(y_val, 0, grid_size - 1))
    return px, py

@njit
def get_obs_jit(grid_size, field_size, walls, harry_pos, goal_pos, 
                last_seen_filch, last_seen_cat, filch_timer, cat_timer):
    obs_map = np.zeros((4, grid_size, grid_size), dtype=np.float32)

    # Channel 0: Walls
    for i in range(len(walls)):
        w = walls[i]
        # Use the new helper here
        x1, y1 = pos_to_px_jit(np.array([w[0], w[2]]), field_size, grid_size)
        x2, y2 = pos_to_px_jit(np.array([w[1], w[3]]), field_size, grid_size)
        obs_map[0, x1:x2+1, y1:y2+1] = 1.0

    # Channel 1 & 2: Harry and Goal
    hx, hy = pos_to_px_jit(harry_pos, field_size, grid_size)
    obs_map[1, hx, hy] = 1.0
    
    gx, gy = pos_to_px_jit(goal_pos, field_size, grid_size)
    obs_map[2, gx, gy] = 1.0

    # Channel 3: Memory
    f_int = max(0.0, 1.0 - (filch_timer / 10.0))
    c_int = max(0.0, 1.0 - (cat_timer / 10.0))
    
    fx, fy = pos_to_px_jit(last_seen_filch, field_size, grid_size)
    cx, cy = pos_to_px_jit(last_seen_cat, field_size, grid_size)
    
    obs_map[3, fx, fy] = max(obs_map[3, fx, fy], f_int)
    obs_map[3, cx, cy] = max(obs_map[3, cx, cy], c_int)
    
    return obs_map

class HarryPotterEnv(gym.Env):
    def __init__(self):
        super(HarryPotterEnv, self).__init__()
        
        # Action: [dx, dy] continuous movement
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Grid parameters for the artificial map
        self.grid_size = 64
        self.field_size = 10.0
        
        # Obs: 4 Channels (Walls, Harry, Goal, Enemies Memory), 64x64 Grid
        self.observation_space = spaces.Box(low=0.0, high=1.0, 
                                            shape=(4, self.grid_size, self.grid_size), 
                                            dtype=np.float32)
        
        # Maze parameters (Same as before)
        self.max_steps = 500
        self.agent_speed = 0.4
        self.enemy_speed = 0.25
        self.smell_radius = 2.5
        self.sight_radius = 5.0
        self.catch_radius = 0.4
        self.goal_radius = 0.5
        self.time_penalty = 1e-2
        self.lose_reward = -1e2
        self.win_reward = WIN_REWARD
        self.distance_scaler = 1e1/ self.field_size
        self.distance_scaler_enemy = self.distance_scaler / (self.field_size/self.smell_radius+1)
        self.outer_bump_penalty = 1.0
        self.bump_penalty = 2.0
        
        self.walls = [
            [4.0, 6.0, 0.0, 4.0],  # Bottom wall
            [4.0, 6.0, 6.0, 10.0]  # Top wall
        ]
        self.walls_np = np.array(self.walls, dtype=np.float32)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps = 0
        
        # Initial positions
        self.harry_pos_min = np.array([1.0, 1.0])
        self.harry_pos_max = np.array([1.0, 9.0])
        self.harry_pos = np.array([np.random.uniform(self.harry_pos_min[0], self.harry_pos_max[0]), 
                                   np.random.uniform(self.harry_pos_min[1], self.harry_pos_max[1])])
        self.goal_pos_min = np.array([9.0, 1.0])
        self.goal_pos_max = np.array([9.0, 9.0])
        self.goal_pos = np.array([np.random.uniform(self.goal_pos_min[0], self.goal_pos_max[0]), 
                                   np.random.uniform(self.goal_pos_min[1], self.goal_pos_max[1])])
        
        self.filch_pos = np.array([5.0, 5.0])
        self.filch_target = self._get_random_waypoint()
        
        self.cat_pos = np.array([8.0, 2.0])
        self.cat_target = self._get_random_waypoint()
        
        # Memory tracking
        self.last_seen_filch = np.copy(self.filch_pos)
        self.filch_timer = 0.0
        self.last_seen_cat = np.copy(self.cat_pos)
        self.cat_timer = 0.0
        
        return self._get_obs(), {}

    def _get_random_waypoint(self):
        while True:
            pt = np.random.uniform(0, 10, size=(2,))
            if not self._is_in_wall(pt):
                return pt

    def _is_in_wall(self, pos):
        return is_in_wall_jit(pos, self.walls_np)

    def _move_entity(self, pos, target, speed):
        direction = target - pos
        dist = np.linalg.norm(direction)
        if dist < speed:
            return target, True # Reached
        direction = direction / dist
        new_pos = pos + direction * speed
        if not self._is_in_wall(new_pos):
            return new_pos, False
        return pos, True # Hit wall, pick new target
    
    def _cross_2d(self, v, w):
        """Calculates the 2D cross product of two vectors."""
        return v[0] * w[1] - v[1] * w[0]

    def _segments_intersect(self, A, B, C, D):
        """
        Checks if segment AB (the path) intersects segment CD (the wall).
        Uses vector projection and interval checking.
        """
        p = A              # Path start
        r = B - A          # Path vector
        q = C              # Wall start
        s = D - C          # Wall vector

        # Cross product of the two direction vectors
        r_cross_s = self._cross_2d(r, s)
        
        q_minus_p = q - p
        
        # If r_cross_s is 0, the lines are perfectly parallel (no intersection)
        if abs(r_cross_s) < 1e-8:
            return False

        # 't' is the scalar projection along the path vector (AB)
        t = self._cross_2d(q_minus_p, s) / r_cross_s
        
        # 'u' is the scalar projection along the wall vector (CD)
        u = self._cross_2d(q_minus_p, r) / r_cross_s

        # The lines cross ONLY IF the projection point falls exactly 
        # inside both vector intervals [0, 1]
        return (0 <= t <= 1) and (0 <= u <= 1)

    def _has_line_of_sight(self, start_pos, target_pos):
        """Checks if the path between start and target collides with any walls."""
        for w in self.walls:
            x_min, x_max, y_min, y_max = w
            
            # The 4 corners of the wall
            bottom_left = np.array([x_min, y_min])
            bottom_right = np.array([x_max, y_min])
            top_left = np.array([x_min, y_max])
            top_right = np.array([x_max, y_max])
            
            # Check intersection with all 4 bounding segments of the wall
            if (self._segments_intersect(start_pos, target_pos, bottom_left, bottom_right) or
                self._segments_intersect(start_pos, target_pos, bottom_right, top_right) or
                self._segments_intersect(start_pos, target_pos, top_right, top_left) or
                self._segments_intersect(start_pos, target_pos, top_left, bottom_left)):
                return False # Path hits a wall
                
        return True # Path is clear

    def step(self, action):
        self.steps += 1
        done = False
        info = {}

        # 1. Move Harry
        action = np.clip(action, -1.0, 1.0)
        move_vector = action * self.agent_speed
        new_harry = self.harry_pos + move_vector
        
        # 2. Elegant Wall Handling (Sliding)
        wall_type = self._is_in_wall(new_harry)
        if wall_type == 0: # No wall
            self.harry_pos = new_harry
            step_reward = 0.0 # Small time penalty
        elif wall_type == 1: # Hit internal wall
            # Instead of lose_reward, give a small "bump" penalty
            step_reward = -self.bump_penalty 
            # Try to slide: move only in X then only in Y
            # This helps the agent "feel" its way around corners
            if not self._is_in_wall(np.array([new_harry[0], self.harry_pos[1]])):
                self.harry_pos[0] = new_harry[0]
            elif not self._is_in_wall(np.array([self.harry_pos[0], new_harry[1]])):
                self.harry_pos[1] = new_harry[1]
        else: # Hit game border
            step_reward = -self.outer_bump_penalty

        dist_to_goal = np.linalg.norm(self.goal_pos - self.harry_pos)
        filch_dist = np.linalg.norm(self.harry_pos - self.last_seen_filch)
        cat_dist = np.linalg.norm(self.harry_pos - self.last_seen_cat)
        reward = -self.time_penalty + step_reward # Time penalty
        reward += -dist_to_goal*self.distance_scaler # Distance to goal penalty
        reward += filch_dist*self.distance_scaler_enemy # Distance to Filch penalty
        reward += cat_dist*self.distance_scaler_enemy # Distance to mrs. Norris penalty

        # 2. Move Enemies
        # Filch logic: random waypoints
        self.filch_pos, reached_f = self._move_entity(self.filch_pos, self.filch_target, self.enemy_speed)
        if reached_f:
            self.filch_target = self._get_random_waypoint()

        # Mrs. Norris logic: smell tracking
        dist_to_cat = np.linalg.norm(self.harry_pos - self.cat_pos)
        if dist_to_cat < self.smell_radius and self._has_line_of_sight(self.cat_pos, self.harry_pos):
            self.cat_target = np.copy(self.harry_pos) # Overwrite target to pursue Harry
        
        self.cat_pos, reached_c = self._move_entity(self.cat_pos, self.cat_target, self.enemy_speed * 1.1)
        if reached_c and dist_to_cat >= self.smell_radius:
            self.cat_target = self._get_random_waypoint()

        # 3. Check Captures (Collisions)
        if np.linalg.norm(self.harry_pos - self.filch_pos) < self.catch_radius or \
           np.linalg.norm(self.harry_pos - self.cat_pos) < self.catch_radius:
            reward = self.lose_reward
            done = True
            info['result'] = 'caught'

        # 4. Check Goal
        elif np.linalg.norm(self.harry_pos - self.goal_pos) < self.goal_radius:
            reward = self.win_reward
            done = True
            info['result'] = 'escaped'

        # 5. Check Timeout
        elif self.steps >= self.max_steps:
            reward = self.lose_reward
            done = True
            info['result'] = 'timeout'
            
        # THE MAGIC FIX: Symlog scaling
        # This maps -10,000 to ~ -9 and +100 to ~ +4.6
        reward = np.sign(reward) * np.log1p(np.abs(reward))

        return self._get_obs(), reward, done, False, info

    def _dist_to_segment(self, P, A, B):
        """Calculates the minimum distance between point P and line segment AB."""
        # Vector from A to B
        v = B - A
        # Vector from A to P
        w = P - A
        
        # Calculate the projection scalar 't' of point P onto the line AB
        # t = dot(w, v) / |v|^2
        v_sq = np.dot(v, v)
        if v_sq < 1e-10: 
            return np.linalg.norm(P - A) # A and B are the same point
            
        t = np.dot(w, v) / v_sq
        
        # Clamp t to the interval [0, 1] to stay on the segment
        t = np.clip(t, 0, 1)
        
        # The nearest point on the segment
        projection = A + t * v
        
        return np.linalg.norm(P - projection), projection

    def _pos_to_pixel(self, pos):
        """Converts continuous coordinates (0-10) to grid indices (0-63)."""
        # Clamp to ensure we stay within the grid
        px = int(np.clip((pos[0] / self.field_size) * self.grid_size, 0, self.grid_size - 1))
        py = int(np.clip((pos[1] / self.field_size) * self.grid_size, 0, self.grid_size - 1))
        return px, py

    

    def _get_obs(self):
        # Update memory timers first
        dist_f = np.linalg.norm(self.harry_pos - self.filch_pos)
        if dist_f < self.sight_radius and has_line_of_sight_jit(self.filch_pos, self.harry_pos, self.walls_np):
            self.last_seen_filch = np.copy(self.filch_pos)
            self.filch_timer = 0.0
        else:
            self.filch_timer = min(10.0, self.filch_timer + 0.1)

        dist_c = np.linalg.norm(self.harry_pos - self.cat_pos)
        if dist_c < self.sight_radius and has_line_of_sight_jit(self.cat_pos, self.harry_pos, self.walls_np):
            self.last_seen_cat = np.copy(self.cat_pos)
            self.cat_timer = 0.0
        else:
            self.cat_timer = min(10.0, self.cat_timer + 0.1)

        return get_obs_jit(
            self.grid_size, self.field_size, self.walls_np, 
            self.harry_pos, self.goal_pos, 
            self.last_seen_filch, self.last_seen_cat, 
            self.filch_timer, self.cat_timer
        )