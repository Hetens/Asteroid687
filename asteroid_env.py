"""
The agent controls a ship at the bottom of a continuous space, avoiding falling asteroids
while trying to survive for a specified number of timesteps.

This is a continuous environment where actions control velocity rather than direct movement.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class AsteroidAvoidEnv(gym.Env):
    """
    Custom Gymnasium environment for asteroid avoidance (Continuous).
    
    The ship stays at the bottom and can accelerate left/right to avoid asteroids
    that fall from the top. The goal is to survive for max_steps timesteps.
    
    Attributes:
        W (float): Environment width
        H (float): Environment height
        N_max (int): Maximum number of asteroid slots
        max_steps (int): Maximum timesteps per episode
        p_spawn (float): Probability of spawning an asteroid per timestep
        max_lives (int): Maximum number of lives
        min_asteroid_radius (float): Minimum asteroid radius
        max_asteroid_radius (float): Maximum asteroid radius
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode=None):
        """
        Initialize the AsteroidAvoidEnv.
        
        Args:
            render_mode (str, optional): Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Continuous space configuration
        self.W = 9.0  # environment width
        self.H = 12.0  # environment height
        self.ship_y = 0.5  # ship stays near bottom
        self.ship_radius = 0.4  # ship collision radius
        
        # Velocity configuration
        self.max_velocity = 0.5  # maximum ship velocity
        self.acceleration_scale = 0.1  # how much action affects velocity
        self.friction = 0.95  # velocity decay per timestep
        
        # Asteroid configuration
        self.N_max = 5  # maximum number of asteroids
        self.p_spawn = 0.35  # spawn probability per timestep (easier)
        self.max_lives = 3
        self.min_asteroid_radius = 0.3  # minimum asteroid size
        self.max_asteroid_radius = 0.6  # maximum asteroid size (smaller)
        self.asteroid_fall_speed = 0.1  # base fall speed
        
        # Episode configuration
        self.max_steps = 1000
        
        # Action space: continuous acceleration [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Observation space: 20 floats
        # 2 (ship_x, ship_velocity) + 15 (5 asteroids × 3 values: x, y, radius) + 1 (distance_to_goal) + 1 (lives) + 1 (present flags packed)
        # Actually: 2 + 5*4 (x, y, radius, present) = 22, but we'll use:
        # ship_x, ship_vx, 5*(ast_x, ast_y, ast_radius, ast_present), goal_progress, lives
        # = 2 + 20 + 2 = 24
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(24,), dtype=np.float32
        )
        
        # Initialize environment state
        self.ship_x = None
        self.ship_vx = None  # ship velocity
        self.asteroids = None
        self.steps_remaining = None
        self.lives = self.max_lives
        
        # Rendering configuration
        self.render_mode = render_mode
        self.SCALE = 40  # pixels per unit
        self.WINDOW_WIDTH = int(self.W * self.SCALE)
        self.WINDOW_HEIGHT = int(self.H * self.SCALE)
        
        # PyGame initialization (lazy)
        self.window = None
        self.clock = None
    
    def get_observation(self):
        obs = np.zeros(24, dtype=np.float32)
        
        # Ship state - better normalization
        obs[0] = self.ship_x / self.W  # [0, 1]
        obs[1] = (self.ship_vx / self.max_velocity + 1.0) / 2.0  # [-1,1] -> [0,1]
        
        # For asteroids, also add relative positions
        for i in range(self.N_max):
            idx = 2 + i * 4
            if self.asteroids[i]['present']:
                # Relative x distance to ship (helps agent understand proximity)
                relative_x = (self.asteroids[i]['x'] - self.ship_x) / self.W
                obs[idx] = (relative_x + 1.0) / 2.0  # normalize to [0,1]
                obs[idx + 1] = self.asteroids[i]['y'] / self.H
                obs[idx + 2] = (self.asteroids[i]['radius'] - self.min_asteroid_radius) / \
                            (self.max_asteroid_radius - self.min_asteroid_radius)
                obs[idx + 3] = 1.0
            else:
                obs[idx:idx+4] = 0.0
        
        obs[22] = self.steps_remaining / self.max_steps
        obs[23] = self.lives / self.max_lives
        
        return obs
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed (int, optional): Random seed for reproducibility
            options (dict, optional): Additional options (unused)
        
        Returns:
            tuple: (observation, info) for the initial state
        """
        super().reset(seed=seed)
        if self.window is not None:
            self.window.fill((0, 0, 0))
        
        # Initialize ship at center with zero velocity
        self.ship_x = self.W / 2.0
        self.ship_vx = 0.0
        
        # Clear all asteroids
        self.asteroids = [
            {'x': 0.0, 'y': 0.0, 'radius': 0.0, 'present': False} 
            for _ in range(self.N_max)
        ]
        
        # Reset episode timer
        self.steps_remaining = self.max_steps

        # Reset lives
        self.lives = self.max_lives
        
        observation = self.get_observation()
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def _move_asteroids(self):
        for asteroid in self.asteroids:
            if asteroid['present']:
                asteroid['y'] -= self.asteroid_fall_speed
                # Remove asteroids that fall off the bottom
                if asteroid['y'] + asteroid['radius'] < 0:
                    asteroid['present'] = False

    def _spawn_asteroids(self):
        # Using Poisson distribution for random asteroid spawning
        num_to_spawn = np.random.poisson(self.p_spawn)

        for _ in range(min(num_to_spawn, 2)):
            empty = [i for i, a in enumerate(self.asteroids) if not a['present']]
            
            if not empty:
                break
            
            i = self.np_random.choice(empty)
            asteroid = self.asteroids[i]
            
            # Variable asteroid size
            radius = self.np_random.uniform(self.min_asteroid_radius, self.max_asteroid_radius)
            
            # Find a valid position for the asteroid
            placed = False
            for _try in range(20):
                x = self.np_random.uniform(radius + 0.5, self.W - radius - 0.5)
                
                # Check spacing to ALL other asteroids
                valid_position = True
                for other in self.asteroids:
                    if not other['present']:
                        continue
                    
                    horizontal_gap = abs(other['x'] - x) - (other['radius'] + radius)
                    vertical_dist = abs(other['y'] - (self.H + radius))
                    
                    # If asteroids are at similar height, require good horizontal spacing
                    if vertical_dist < 2.0 and horizontal_gap < 2.0:
                        valid_position = False
                        break
                    # If far apart vertically, less spacing needed
                    elif vertical_dist >= 2.0 and horizontal_gap < 1.0:
                        valid_position = False
                        break
                
                if valid_position:
                    placed = True
                    break
            
            if placed:
                asteroid['x'] = x
                asteroid['y'] = self.H + radius
                asteroid['radius'] = radius
                asteroid['present'] = True

    def _check_collision_and_calculate_reward(self):
        collision = False
        min_distance = float('inf')
        danger_threshold = 3.0
        closest_asteroid_x = None

        for asteroid in self.asteroids:
            if asteroid['present']:
                dist = np.sqrt(
                    (asteroid['x'] - self.ship_x) ** 2 + 
                    (asteroid['y'] - self.ship_y) ** 2
                )
                edge_dist = dist - (asteroid['radius'] + self.ship_radius)
                
                if edge_dist < min_distance:
                    min_distance = edge_dist
                    closest_asteroid_x = asteroid['x']
                
                if edge_dist < 0:  # collision
                    collision = True
                    asteroid['present'] = False
                    break

        # Decrement steps remaining
        self.steps_remaining -= 1
        
        # Determine reward and termination
        if collision:
            reward = -100.0  # big penalty
            self.lives -= 1
            terminated = self.lives <= 0
        elif self.steps_remaining <= 0:
            reward = 200.0  # win bonus
            terminated = True
        else:
            # Base survival reward
            reward = 0.1
    
            # STRONG penalty for being near walls -> this is to avoid the reward hacking
            wall_distance = min(self.ship_x - self.ship_radius, 
                            (self.W - self.ship_radius) - self.ship_x)
            if wall_distance < 1.0:
                wall_penalty = -2.0 * (1.0 - wall_distance)
                reward += wall_penalty
            
            # Reward for being near center
            center_x = self.W / 2.0
            distance_from_center = abs(self.ship_x - center_x)
            center_bonus = 0.5 * (1.0 - distance_from_center / (self.W / 2.0))
            reward += center_bonus
            
            # Distance-based bonus for avoiding asteroids
            if min_distance < danger_threshold and min_distance > 0:
                distance_bonus = 1.5 * (min_distance / danger_threshold)
                reward += distance_bonus
            elif min_distance >= danger_threshold:
                reward += 1.5
            
            # Bonus for moving AWAY from closest asteroid
            if closest_asteroid_x is not None and abs(self.ship_vx) > 0.05:
                # Reward for moving in the right direction
                should_move_right = closest_asteroid_x > self.ship_x
                is_moving_right = self.ship_vx > 0
                
                if should_move_right == is_moving_right:
                    reward += 0.5
        
            terminated = False

        return reward, terminated
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Args:
            action (np.ndarray): Continuous action for acceleration [-1, 1]
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Extract scalar action
        acceleration = float(action[0])
        acceleration = np.clip(acceleration, -1.0, 1.0)
        
        # Apply acceleration to velocity
        self.ship_vx += acceleration * self.acceleration_scale
        
        # Apply friction
        self.ship_vx *= self.friction
        
        # Clamp velocity
        self.ship_vx = np.clip(self.ship_vx, -self.max_velocity, self.max_velocity)
        
        # Update position based on velocity
        self.ship_x += self.ship_vx
        
        # Clamp ship position to valid range and bounce off walls
        if self.ship_x < self.ship_radius:
            self.ship_x = self.ship_radius
            self.ship_vx = abs(self.ship_vx) * 0.5  # bounce with energy loss
        elif self.ship_x > self.W - self.ship_radius:
            self.ship_x = self.W - self.ship_radius
            self.ship_vx = -abs(self.ship_vx) * 0.5  # bounce with energy loss
        
        # Move asteroids down
        self._move_asteroids()

        # Spawn new asteroids
        self._spawn_asteroids()
        
        # Check for collision and calculate reward
        reward, terminated = self._check_collision_and_calculate_reward()
        
        truncated = False
        observation = self.get_observation()
        info = {
            "lives": self.lives,
            "steps_remaining": self.steps_remaining,
            "steps_survived": self.max_steps - self.steps_remaining
        }
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array", else None
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _world_to_pixel(self, world_x, world_y):
        """Convert world coordinates to pixel coordinates."""
        pixel_x = int(world_x * self.SCALE)
        pixel_y = int((self.H - world_y) * self.SCALE)
        return pixel_x, pixel_y
    
    def _render_frame(self):
        """
        Internal rendering method using PyGame.
        
        Returns:
            np.ndarray or None: RGB array if render_mode is "rgb_array", else None
        """
        # Initialize PyGame window if needed
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
            )
            pygame.display.set_caption("Asteroid Avoid Environment (Continuous)")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create canvas
        canvas = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        canvas.fill((10, 10, 30))  # Dark blue background
        
        # Draw ship (blue circle)
        ship_pixel_x, ship_pixel_y = self._world_to_pixel(self.ship_x, self.ship_y)
        ship_pixel_radius = int(self.ship_radius * self.SCALE)
        pygame.draw.circle(
            canvas,
            (50, 150, 255),  # Blue
            (ship_pixel_x, ship_pixel_y),
            ship_pixel_radius
        )
        # Ship outline
        pygame.draw.circle(
            canvas,
            (100, 200, 255),  # Lighter blue
            (ship_pixel_x, ship_pixel_y),
            ship_pixel_radius,
            2
        )
        
        # Draw asteroids (red circles with variable sizes)
        for asteroid in self.asteroids:
            if asteroid['present']:
                ast_pixel_x, ast_pixel_y = self._world_to_pixel(
                    asteroid['x'], asteroid['y']
                )
                ast_pixel_radius = int(asteroid['radius'] * self.SCALE)
                # Gradient-like effect based on size
                intensity = int(150 + 105 * (asteroid['radius'] - self.min_asteroid_radius) / 
                               (self.max_asteroid_radius - self.min_asteroid_radius))
                pygame.draw.circle(
                    canvas,
                    (intensity, 50, 50),  # Red with intensity based on size
                    (ast_pixel_x, ast_pixel_y),
                    ast_pixel_radius
                )
                # Outline
                pygame.draw.circle(
                    canvas,
                    (255, 100, 100),  # Lighter red
                    (ast_pixel_x, ast_pixel_y),
                    ast_pixel_radius,
                    2
                )

        # Display Game Progress on the demo window
        if not pygame.font.get_init():
            pygame.font.init()

        font = pygame.font.SysFont("Arial", 14)
        hud_lines = [
            f"Steps Remaining: {self.steps_remaining}",
            f"Lives Remaining: {self.lives}",
            f"Velocity: {self.ship_vx:.2f}",
        ]

        y_offset = 5
        for line in hud_lines:
            text_surface = font.render(line, True, (255, 255, 255))
            canvas.blit(text_surface, (5, y_offset))
            y_offset += 18

        
        if self.render_mode == "human":
            # Copy canvas to window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.flip()
            
            # Maintain framerate
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """
        Clean up resources (close PyGame window).
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
