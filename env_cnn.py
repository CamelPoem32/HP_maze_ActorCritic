import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
        self.lose_reward = -1e3
        self.win_reward = 1e4
        self.distance_scaler = 3e1 / (self.field_size**2)
        self.distance_scaler_enemy = self.distance_scaler / ((self.field_size/self.smell_radius+1)**2)
        
        self.walls = [
            [4.0, 6.0, 0.0, 4.0],  # Bottom wall
            [4.0, 6.0, 6.0, 10.0]  # Top wall
        ]

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
        if pos[0] < 0 or pos[0] > 10 or pos[1] < 0 or pos[1] > 10:
            return 2
        for w in self.walls:
            if w[0] < pos[0] < w[1] and w[2] < pos[1] < w[3]:
                return 1
        return 0

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
        reward = -self.time_penalty # Time penalty
        reward += -np.float32(np.linalg.norm((self.goal_pos - self.harry_pos)**2)*self.distance_scaler) # Distance to goal penalty
        reward += np.float32(np.linalg.norm((self.last_seen_filch - self.harry_pos)**2)*self.distance_scaler_enemy) # Distance to Filch penalty
        reward += np.float32(np.linalg.norm((self.last_seen_cat - self.harry_pos)**2)*self.distance_scaler_enemy) # Distance to mrs. Norris penalty
        done = False
        info = {}

        # 1. Move Harry
        action = np.clip(action, -1.0, 1.0)
        new_harry = self.harry_pos + action * self.agent_speed
        if not self._is_in_wall(new_harry):
            self.harry_pos = new_harry
        elif self._is_in_wall(new_harry) == 1:          # In wall, not in game border
            reward = self.lose_reward

            # print(np.min(self.get_walls_distance()))
            done = True
            info['result'] = 'wall'

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
        # 1. Update visibility / memory (same logic as before)
        if np.linalg.norm(self.harry_pos - self.filch_pos) < self.sight_radius and self._has_line_of_sight(self.filch_pos, self.harry_pos):
            self.last_seen_filch = np.copy(self.filch_pos)
            self.filch_timer = 0.0
        else:
            self.filch_timer = min(10.0, self.filch_timer + 0.1)

        if np.linalg.norm(self.harry_pos - self.cat_pos) < self.sight_radius and self._has_line_of_sight(self.cat_pos, self.harry_pos):
            self.last_seen_cat = np.copy(self.cat_pos)
            self.cat_timer = 0.0
        else:
            self.cat_timer = min(10.0, self.cat_timer + 0.1)

        # 2. Build the Artificial Map (4 Channels, 64x64)
        obs_map = np.zeros((4, self.grid_size, self.grid_size), dtype=np.float32)

        # Channel 0: Walls
        for w in self.walls:
            min_x, min_y = self._pos_to_pixel([w[0], w[2]])
            max_x, max_y = self._pos_to_pixel([w[1], w[3]])
            # Fill the wall area with 1.0
            obs_map[0, min_x:max_x+1, min_y:max_y+1] = 1.0

        # Channel 1: Harry
        hx, hy = self._pos_to_pixel(self.harry_pos)
        obs_map[1, hx, hy] = 1.0

        # Channel 2: Goal
        gx, gy = self._pos_to_pixel(self.goal_pos)
        obs_map[2, gx, gy] = 1.0

        # Channel 3: Enemies Memory
        # Intensity fades from 1.0 (just seen) down to 0.0 (seen 10+ seconds ago)
        filch_intensity = max(0.0, 1.0 - (self.filch_timer / 10.0))
        cat_intensity = max(0.0, 1.0 - (self.cat_timer / 10.0))
        
        fx, fy = self._pos_to_pixel(self.last_seen_filch)
        cx, cy = self._pos_to_pixel(self.last_seen_cat)
        
        obs_map[3, fx, fy] = max(obs_map[3, fx, fy], filch_intensity)
        obs_map[3, cx, cy] = max(obs_map[3, cx, cy], cat_intensity)

        return obs_map