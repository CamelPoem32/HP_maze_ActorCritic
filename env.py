import gymnasium as gym
from gymnasium import spaces
import numpy as np

class HarryPotterEnv(gym.Env):
    def __init__(self):
        super(HarryPotterEnv, self).__init__()
        
        # Action: [dx, dy] continuous movement
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Obs: [harry_x, harry_y, filch_x, filch_y, filch_timer, cat_x, cat_y, cat_timer, goal_x, goal_y]
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(10,), dtype=np.float32)
        
        # Maze parameters
        self.max_steps = 500
        self.agent_speed = 0.4
        self.enemy_speed = 0.25
        self.smell_radius = 2.5
        self.sight_radius = 3.5
        self.catch_radius = 0.4
        self.goal_radius = 0.5
        self.time_penalty = -0.1
        self.lose_reward = -1e2
        self.win_reward = 1e2
        self.distance_scaler = 5e-2
        
        # Walls defining the "loop" (AABB: [x_min, x_max, y_min, y_max])
        self.walls = [
            [4.0, 6.0, 0.0, 4.0],  # Bottom wall
            [4.0, 6.0, 6.0, 10.0]  # Top wall (leaves middle and sides open)
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
            return True
        for w in self.walls:
            if w[0] < pos[0] < w[1] and w[2] < pos[1] < w[3]:
                return True
        return False

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

    def step(self, action):
        self.steps += 1
        # reward = self.time_penalty # Time penalty
        reward = -np.float32(np.linalg.norm((self.goal_pos - self.harry_pos)**2)*self.distance_scaler) # Distance to goal penalty
        done = False
        info = {}

        # 1. Move Harry
        action = np.clip(action, -1.0, 1.0)
        new_harry = self.harry_pos + action * self.agent_speed
        if not self._is_in_wall(new_harry):
            self.harry_pos = new_harry

        # 2. Move Enemies
        # Filch logic: random waypoints
        self.filch_pos, reached_f = self._move_entity(self.filch_pos, self.filch_target, self.enemy_speed)
        if reached_f:
            self.filch_target = self._get_random_waypoint()

        # Mrs. Norris logic: smell tracking
        dist_to_cat = np.linalg.norm(self.harry_pos - self.cat_pos)
        if dist_to_cat < self.smell_radius:
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
            done = True
            info['result'] = 'timeout'

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        # Update visibility / memory
        if np.linalg.norm(self.harry_pos - self.filch_pos) < self.sight_radius:
            self.last_seen_filch = np.copy(self.filch_pos)
            self.filch_timer = 0.0
        else:
            self.filch_timer = min(10.0, self.filch_timer + 0.1)

        if np.linalg.norm(self.harry_pos - self.cat_pos) < self.sight_radius:
            self.last_seen_cat = np.copy(self.cat_pos)
            self.cat_timer = 0.0
        else:
            self.cat_timer = min(10.0, self.cat_timer + 0.1)

        obs = np.concatenate([
            self.harry_pos, 
            self.last_seen_filch, [self.filch_timer],
            self.last_seen_cat, [self.cat_timer],
            self.goal_pos
        ])
        return obs