"""
The agent controls a ship at the bottom of a grid, avoiding falling asteroids
while trying to survive for a specified number of timesteps.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces


class AsteroidAvoidEnv(gym.Env):
    """
    Custom Gymnasium environment for asteroid avoidance.
    
    The ship stays at the bottom row and can move left/right to avoid asteroids
    that fall from the top. The goal is to survive for max_steps timesteps.
    
    Attributes:
        W (int): Grid width (number of columns)
        H (int): Grid height (number of rows)
        N_max (int): Maximum number of asteroid slots
        max_steps (int): Maximum timesteps per episode
        p_spawn (float): Probability of spawning an asteroid per timestep
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    def __init__(self, render_mode=None):
        """
        Initialize the AsteroidAvoidEnv.
        
        Args:
            render_mode (str, optional): Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        # Grid configuration
        self.W = 7  # grid width
        self.H = 12  # grid height
        self.ship_y = 0  # ship always stays on bottom row
        
        # Asteroid configuration
        self.N_max = 5  # maximum number of asteroids
        self.p_spawn = 0.2  # spawn probability per timestep
        
        # Episode configuration
        self.max_steps = 300
        
        # Action space: 0 = left, 1 = stay, 2 = right
        self.action_space = spaces.Discrete(3)
        
        # Observation space: 17 floats
        # 1 (ship_x) + 15 (5 asteroids × 3 values) + 1 (distance_to_goal)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(17,), dtype=np.float32
        )
        
        # Initialize environment state
        self.ship_x = None
        self.asteroids = None
        self.steps_remaining = None
        
        # Rendering configuration
        self.render_mode = render_mode
        self.CELL_SIZE = 40  # pixels per grid cell
        self.WINDOW_WIDTH = self.W * self.CELL_SIZE
        self.WINDOW_HEIGHT = self.H * self.CELL_SIZE
        
        # PyGame initialization (lazy)
        self.window = None
        self.clock = None
    
    def get_observation(self):
        """
        Construct the observation vector from the current state.
        
        Returns:
            np.ndarray: Observation vector of shape (17,) with normalized values
        """
        obs = np.zeros(17, dtype=np.float32)
        
        # Ship state (1 value)
        obs[0] = self.ship_x / (self.W - 1)
        
        # Asteroid states (5 × 3 = 15 values)
        for i in range(self.N_max):
            idx = 1 + i * 3
            if self.asteroids[i]['present']:
                obs[idx] = self.asteroids[i]['x'] / (self.W - 1)  # x_normalized
                obs[idx + 1] = self.asteroids[i]['y'] / (self.H - 1)  # y_normalized
                obs[idx + 2] = 1.0  # present
            else:
                obs[idx] = 0.0
                obs[idx + 1] = 0.0
                obs[idx + 2] = 0.0  # not present
        
        # Goal progress (1 value)
        obs[16] = self.steps_remaining / self.max_steps
        
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
        # Seed the random number generator
        super().reset(seed=seed)
        
        # Initialize ship at center
        self.ship_x = self.W // 2
        
        # Clear all asteroids
        self.asteroids = [
            {'x': 0, 'y': 0, 'present': False} for _ in range(self.N_max)
        ]
        
        # Reset episode timer
        self.steps_remaining = self.max_steps
        
        observation = self.get_observation()
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one timestep of the environment.
        
        Args:
            action (int): Action to take (0=left, 1=stay, 2=right)
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Apply action: move ship
        if action == 0:  # move left
            self.ship_x -= 1
        elif action == 2:  # move right
            self.ship_x += 1
        # action == 1: stay (no change)
        
        # Clamp ship position to valid range
        self.ship_x = np.clip(self.ship_x, 0, self.W - 1)
        
        # Move asteroids down
        for asteroid in self.asteroids:
            if asteroid['present']:
                asteroid['y'] -= 1
                # Remove asteroids that fall off the bottom
                if asteroid['y'] < 0:
                    asteroid['present'] = False
        
        # Spawn new asteroid with probability p_spawn
        if self.np_random.random() < self.p_spawn:
            # Find first empty slot
            for asteroid in self.asteroids:
                if not asteroid['present']:
                    asteroid['x'] = self.np_random.integers(0, self.W)
                    asteroid['y'] = self.H - 1
                    asteroid['present'] = True
                    break
        
        # Check for collision
        collision = False
        for asteroid in self.asteroids:
            if (asteroid['present'] and 
                asteroid['x'] == self.ship_x and 
                asteroid['y'] == self.ship_y):
                collision = True
                break
        
        # Decrement steps remaining
        self.steps_remaining -= 1
        
        # Determine reward and termination
        if collision:
            reward = -100.0
            terminated = True
        elif self.steps_remaining <= 0:
            reward = 100.0
            terminated = True
        else:
            reward = 1.0
            terminated = False
        
        truncated = False
        observation = self.get_observation()
        info = {}
        
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
            pygame.display.set_caption("Asteroid Avoid Environment")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create canvas
        canvas = pygame.Surface((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        canvas.fill((0, 0, 0))  # Black background
        
        # Convert grid coordinates to pixel coordinates
        def grid_to_pixel(grid_x, grid_y):
            pixel_x = grid_x * self.CELL_SIZE
            pixel_y = (self.H - 1 - grid_y) * self.CELL_SIZE
            return pixel_x, pixel_y
        
        # Draw ship (blue rectangle)
        ship_pixel_x, ship_pixel_y = grid_to_pixel(self.ship_x, self.ship_y)
        pygame.draw.rect(
            canvas,
            (0, 100, 255),  # Blue
            pygame.Rect(
                ship_pixel_x, ship_pixel_y,
                self.CELL_SIZE, self.CELL_SIZE
            )
        )
        
        # Draw asteroids (red rectangles)
        for asteroid in self.asteroids:
            if asteroid['present']:
                ast_pixel_x, ast_pixel_y = grid_to_pixel(
                    asteroid['x'], asteroid['y']
                )
                pygame.draw.rect(
                    canvas,
                    (255, 50, 50),  # Red
                    pygame.Rect(
                        ast_pixel_x, ast_pixel_y,
                        self.CELL_SIZE, self.CELL_SIZE
                    )
                )
        
        # Draw grid lines
        for x in range(self.W + 1):
            pygame.draw.line(
                canvas,
                (50, 50, 50),  # Dark gray
                (x * self.CELL_SIZE, 0),
                (x * self.CELL_SIZE, self.WINDOW_HEIGHT),
                width=1
            )
        
        for y in range(self.H + 1):
            pygame.draw.line(
                canvas,
                (50, 50, 50),  # Dark gray
                (0, y * self.CELL_SIZE),
                (self.WINDOW_WIDTH, y * self.CELL_SIZE),
                width=1
            )
        
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
