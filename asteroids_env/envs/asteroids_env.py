"""
Asteroids Gymnasium Environment

A Gymnasium-compatible reinforcement learning environment based on the classic
Asteroids game. The agent controls a spaceship that must avoid and destroy asteroids.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from typing import Optional, Tuple, Dict, Any, List
import random


class AsteroidsEnv(gym.Env):
    """
    Asteroids Gymnasium Environment
    
    ## Observation Space (Agent-Centric - Option C)
    - player: [x, y, vx, vy, rotation] (5 floats)
    - shoot_ready: 0 or 1 (discrete)
    - nearest_asteroids: 5x5 array [rel_x, rel_y, vx, vy, radius] for 5 nearest
    - num_asteroids: total asteroid count
    - num_shots: active shot count
    
    ## Action Space (MultiDiscrete - Option A)
    - [rotation, thrust, shoot]
    - rotation: 0=left, 1=none, 2=right
    - thrust: 0=backward, 1=none, 2=forward
    - shoot: 0=no, 1=yes
    
    ## Rewards (Shaped - Option B)
    - Survival: +0.1 per timestep
    - Asteroid destroyed: +2, +4, or +6 based on size
    - Collision: -50 (terminal)
    - Danger penalty: up to -1.0 when very close to asteroids
    - Shot penalty: -0.05 to discourage spam
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(self, render_mode: Optional[str] = None, max_asteroids: int = 15):
        """
        Initialize the Asteroids environment.
        
        Args:
            render_mode: "human" for display, "rgb_array" for recording, None for headless
            max_asteroids: Maximum number of simultaneous asteroids (soft limit)
        """
        super().__init__()
        
        # Screen dimensions (from constants)
        self.screen_width = 1280
        self.screen_height = 720
        
        # Game constants
        self.player_radius = 20
        self.player_turn_speed = 300  # degrees per second
        self.player_speed = 200  # pixels per second
        self.shot_radius = 5
        self.shot_speed = 500
        self.shoot_cooldown_time = 0.3  # seconds
        self.asteroid_min_radius = 20
        self.asteroid_kinds = 3
        self.asteroid_spawn_rate = 0.8  # seconds between spawns
        self.max_asteroids = max_asteroids
        
        # Action space: [rotation, thrust, shoot]
        # rotation: 0=left, 1=none, 2=right
        # thrust: 0=backward, 1=none, 2=forward  
        # shoot: 0=no, 1=yes
        self.action_space = spaces.MultiDiscrete([3, 3, 2])
        
        # Observation space (agent-centric)
        self.observation_space = spaces.Dict({
            "player": spaces.Box(
                low=np.array([0, 0, -500, -500, 0], dtype=np.float32),
                high=np.array([1280, 720, 500, 500, 360], dtype=np.float32),
                shape=(5,),
                dtype=np.float32
            ),
            "shoot_ready": spaces.Discrete(2),
            "nearest_asteroids": spaces.Box(
                low=-2000, 
                high=2000, 
                shape=(5, 5),  # 5 asteroids x 5 features
                dtype=np.float32
            ),
            "num_asteroids": spaces.Discrete(100),
            "num_shots": spaces.Discrete(50),
        })
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Game state (initialized in reset)
        self.player_pos = None
        self.player_vel = None
        self.player_rotation = None
        self.shoot_cooldown = None
        self.asteroids = []  # List of dicts with pos, vel, radius
        self.shots = []  # List of dicts with pos, vel
        self.spawn_timer = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """
        Get current observation in agent-centric format.
        
        Returns:
            Dictionary observation with player state and nearby asteroids
        """
        # Player state: [x, y, vx, vy, rotation]
        player_state = np.array([
            self.player_pos[0],
            self.player_pos[1],
            self.player_vel[0],
            self.player_vel[1],
            self.player_rotation
        ], dtype=np.float32)
        
        # Shoot readiness
        shoot_ready = 1 if self.shoot_cooldown <= 0 else 0
        
        # Find 5 nearest asteroids
        nearest = np.zeros((5, 5), dtype=np.float32)
        
        if len(self.asteroids) > 0:
            # Calculate distances to all asteroids
            distances = []
            for i, ast in enumerate(self.asteroids):
                dist = np.linalg.norm(ast['pos'] - self.player_pos)
                distances.append((dist, i))
            
            # Sort by distance and take up to 5 nearest
            distances.sort()
            for j, (dist, i) in enumerate(distances[:5]):
                ast = self.asteroids[i]
                # Relative position and absolute velocity and radius
                nearest[j] = [
                    ast['pos'][0] - self.player_pos[0],  # relative x
                    ast['pos'][1] - self.player_pos[1],  # relative y
                    ast['vel'][0],  # velocity x
                    ast['vel'][1],  # velocity y
                    ast['radius']   # radius
                ]
        
        return {
            "player": player_state,
            "shoot_ready": shoot_ready,
            "nearest_asteroids": nearest,
            "num_asteroids": len(self.asteroids),
            "num_shots": len(self.shots),
        }
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Get auxiliary information for debugging.
        
        Returns:
            Dictionary with debug information
        """
        min_distance = float('inf')
        if len(self.asteroids) > 0:
            for ast in self.asteroids:
                dist = np.linalg.norm(ast['pos'] - self.player_pos)
                min_distance = min(min_distance, dist)
        
        return {
            "step_count": self.step_count,
            "num_asteroids": len(self.asteroids),
            "num_shots": len(self.shots),
            "min_asteroid_distance": min_distance if min_distance != float('inf') else -1,
            "total_reward": self.total_reward,
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        # Seed RNG
        super().reset(seed=seed)
        
        # Initialize player at center
        self.player_pos = np.array([self.screen_width / 2, self.screen_height / 2], dtype=np.float32)
        self.player_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.player_rotation = 0.0
        self.shoot_cooldown = 0.0
        
        # Clear asteroids and shots
        self.asteroids = []
        self.shots = []
        
        # Spawn initial asteroids (2-5)
        num_initial = self.np_random.integers(2, 6)
        for _ in range(num_initial):
            self._spawn_asteroid()
        
        # Reset timers and counters
        self.spawn_timer = 0.0
        self.step_count = 0
        self.total_reward = 0.0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def step(
        self, 
        action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.
        
        Args:
            action: [rotation, thrust, shoot] actions
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        dt = 1.0 / 60.0  # 60 FPS timestep
        self.step_count += 1
        reward = 0.0
        terminated = False
        
        # Decode action
        rotation_action = action[0]  # 0=left, 1=none, 2=right
        thrust_action = action[1]    # 0=backward, 1=none, 2=forward
        shoot_action = action[2]     # 0=no, 1=yes
        
        # Update player rotation
        if rotation_action == 0:  # left
            self.player_rotation -= self.player_turn_speed * dt
        elif rotation_action == 2:  # right
            self.player_rotation += self.player_turn_speed * dt
        
        # Normalize rotation to [0, 360)
        self.player_rotation = self.player_rotation % 360
        
        # Update player velocity based on thrust
        if thrust_action == 2:  # forward
            rad = np.radians(self.player_rotation)
            direction = np.array([np.sin(rad), np.cos(rad)])
            self.player_vel = direction * self.player_speed
        elif thrust_action == 0:  # backward
            rad = np.radians(self.player_rotation)
            direction = np.array([np.sin(rad), np.cos(rad)])
            self.player_vel = direction * (-self.player_speed)
        else:  # no thrust
            # Gradual deceleration
            self.player_vel *= 0.98
        
        # Update player position
        self.player_pos += self.player_vel * dt
        
        # Keep player on screen (wrap or clamp - using clamp)
        self.player_pos[0] = np.clip(self.player_pos[0], 0, self.screen_width)
        self.player_pos[1] = np.clip(self.player_pos[1], 0, self.screen_height)
        
        # Update shoot cooldown
        self.shoot_cooldown = max(0, self.shoot_cooldown - dt)
        
        # Handle shooting
        if shoot_action == 1 and self.shoot_cooldown <= 0:
            self._create_shot()
            self.shoot_cooldown = self.shoot_cooldown_time
            reward -= 0.05  # Small penalty for shooting
        
        # Update asteroids
        asteroids_to_remove = []
        for i, ast in enumerate(self.asteroids):
            ast['pos'] += ast['vel'] * dt
            
            # Remove if too far off-screen (beyond a reasonable buffer)
            # Use a larger buffer than shots since asteroids should wrap around
            buffer = 500  # pixels
            if (ast['pos'][0] < -buffer or ast['pos'][0] > self.screen_width + buffer or
                ast['pos'][1] < -buffer or ast['pos'][1] > self.screen_height + buffer):
                asteroids_to_remove.append(i)
        
        # Remove off-screen asteroids
        for i in reversed(asteroids_to_remove):
            self.asteroids.pop(i)
        
        # Update shots
        shots_to_remove = []
        for i, shot in enumerate(self.shots):
            shot['pos'] += shot['vel'] * dt
            
            # Remove if off-screen
            if (shot['pos'][0] < -100 or shot['pos'][0] > self.screen_width + 100 or
                shot['pos'][1] < -100 or shot['pos'][1] > self.screen_height + 100):
                shots_to_remove.append(i)
        
        # Remove off-screen shots
        for i in reversed(shots_to_remove):
            self.shots.pop(i)
        
        # Check collisions: player with asteroids
        for ast in self.asteroids:
            dist = np.linalg.norm(ast['pos'] - self.player_pos)
            if dist <= self.player_radius + ast['radius']:
                reward = -50  # Large penalty for collision
                terminated = True
                break
        
        # Check collisions: shots with asteroids
        if not terminated:
            asteroids_to_remove = []
            shots_to_remove = []
            asteroids_to_add = []
            
            for i, shot in enumerate(self.shots):
                for j, ast in enumerate(self.asteroids):
                    dist = np.linalg.norm(shot['pos'] - ast['pos'])
                    if dist <= self.shot_radius + ast['radius']:
                        # Hit!
                        reward += ast['radius'] / 10  # 2, 4, or 6 points
                        
                        shots_to_remove.append(i)
                        asteroids_to_remove.append(j)
                        
                        # Split asteroid if not minimum size
                        if ast['radius'] > self.asteroid_min_radius:
                            new_asteroids = self._split_asteroid(ast)
                            asteroids_to_add.extend(new_asteroids)
                        
                        break  # Shot can only hit one asteroid
            
            # Remove hit shots and asteroids
            for i in reversed(sorted(set(shots_to_remove))):
                self.shots.pop(i)
            for j in reversed(sorted(set(asteroids_to_remove))):
                self.asteroids.pop(j)
            
            # Add new split asteroids
            self.asteroids.extend(asteroids_to_add)
        
        # Survival reward
        if not terminated:
            reward += 0.1
        
        # Danger penalty (distance-based)
        if not terminated and len(self.asteroids) > 0:
            min_dist = float('inf')
            for ast in self.asteroids:
                dist = np.linalg.norm(ast['pos'] - self.player_pos)
                min_dist = min(min_dist, dist)
            
            if min_dist < 100:
                danger_penalty = (100 - min_dist) / 100  # 0 to 1
                reward -= danger_penalty
        
        # Spawn new asteroids
        self.spawn_timer += dt
        if self.spawn_timer >= self.asteroid_spawn_rate and len(self.asteroids) < self.max_asteroids:
            self._spawn_asteroid()
            self.spawn_timer = 0.0
        
        self.total_reward += reward
        
        observation = self._get_obs()
        info = self._get_info()
        truncated = False  # Handled by TimeLimit wrapper via registration
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def _create_shot(self):
        """Create a new shot from player's current position and rotation."""
        rad = np.radians(self.player_rotation)
        direction = np.array([np.sin(rad), np.cos(rad)])
        
        shot = {
            'pos': self.player_pos.copy(),
            'vel': direction * self.shot_speed
        }
        self.shots.append(shot)
    
    def _spawn_asteroid(self):
        """Spawn a new asteroid at a random edge of the screen."""
        # Choose random edge: 0=left, 1=right, 2=top, 3=bottom
        edge = self.np_random.integers(0, 4)
        
        # Random position on that edge
        if edge == 0:  # left
            pos = np.array([-self.asteroid_min_radius * 2, 
                           self.np_random.uniform(0, self.screen_height)], dtype=np.float32)
            base_vel = np.array([1, 0], dtype=np.float32)
        elif edge == 1:  # right
            pos = np.array([self.screen_width + self.asteroid_min_radius * 2,
                           self.np_random.uniform(0, self.screen_height)], dtype=np.float32)
            base_vel = np.array([-1, 0], dtype=np.float32)
        elif edge == 2:  # top
            pos = np.array([self.np_random.uniform(0, self.screen_width),
                           -self.asteroid_min_radius * 2], dtype=np.float32)
            base_vel = np.array([0, 1], dtype=np.float32)
        else:  # bottom
            pos = np.array([self.np_random.uniform(0, self.screen_width),
                           self.screen_height + self.asteroid_min_radius * 2], dtype=np.float32)
            base_vel = np.array([0, -1], dtype=np.float32)
        
        # Random speed and angle variation
        speed = self.np_random.uniform(40, 100)
        angle_variation = self.np_random.uniform(-30, 30)
        
        # Rotate base velocity
        rad = np.radians(angle_variation)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        vel = rotation_matrix @ base_vel * speed
        
        # Random size
        kind = self.np_random.integers(1, self.asteroid_kinds + 1)
        radius = self.asteroid_min_radius * kind
        
        asteroid = {
            'pos': pos,
            'vel': vel,
            'radius': radius
        }
        self.asteroids.append(asteroid)
    
    def _split_asteroid(self, asteroid: Dict) -> List[Dict]:
        """
        Split an asteroid into two smaller ones.
        
        Args:
            asteroid: The asteroid to split
            
        Returns:
            List of two new smaller asteroids
        """
        new_radius = asteroid['radius'] - self.asteroid_min_radius
        random_angle = self.np_random.uniform(20, 50)
        
        # Rotation matrices
        rad1 = np.radians(random_angle)
        rad2 = np.radians(-random_angle)
        
        cos1, sin1 = np.cos(rad1), np.sin(rad1)
        cos2, sin2 = np.cos(rad2), np.sin(rad2)
        
        rot1 = np.array([[cos1, -sin1], [sin1, cos1]])
        rot2 = np.array([[cos2, -sin2], [sin2, cos2]])
        
        vel1 = rot1 @ asteroid['vel'] * 1.2
        vel2 = rot2 @ asteroid['vel'] * 1.2
        
        asteroid1 = {
            'pos': asteroid['pos'].copy(),
            'vel': vel1,
            'radius': new_radius
        }
        asteroid2 = {
            'pos': asteroid['pos'].copy(),
            'vel': vel2,
            'radius': new_radius
        }
        
        return [asteroid1, asteroid2]
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame using pygame."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Asteroids Gymnasium Environment")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.screen_width, self.screen_height))
        canvas.fill((0, 0, 0))  # Black background
        
        # Draw asteroids
        for ast in self.asteroids:
            pygame.draw.circle(
                canvas,
                (200, 200, 200),  # Light gray
                ast['pos'].astype(int),
                int(ast['radius']),
                2
            )
        
        # Draw shots
        for shot in self.shots:
            pygame.draw.circle(
                canvas,
                (255, 255, 0),  # Yellow
                shot['pos'].astype(int),
                self.shot_radius
            )
        
        # Draw player (triangle)
        rad = np.radians(self.player_rotation)
        forward = np.array([np.sin(rad), np.cos(rad)])
        right = np.array([np.sin(rad + np.pi/2), np.cos(rad + np.pi/2)])
        
        tip = self.player_pos + forward * self.player_radius
        left_base = self.player_pos - forward * self.player_radius - right * self.player_radius * 0.7
        right_base = self.player_pos - forward * self.player_radius + right * self.player_radius * 0.7
        
        pygame.draw.polygon(
            canvas,
            (0, 200, 255),  # Cyan
            [tip, left_base, right_base],
            2
        )
        
        # Draw UI info
        if self.render_mode == "human":
            font = pygame.font.Font(None, 24)
            info_text = f"Asteroids: {len(self.asteroids)} | Shots: {len(self.shots)} | Reward: {self.total_reward:.1f}"
            text_surface = font.render(info_text, True, (255, 255, 255))
            canvas.blit(text_surface, (10, 10))
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
