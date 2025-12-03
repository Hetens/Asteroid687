# Asteroids Game

A classic arcade-style Asteroids game implemented in Python using Pygame.

## Description

This is a modern recreation of the classic Asteroids arcade game where players control a spaceship and must survive by shooting incoming asteroids while navigating through space.

## Features

- Smooth spaceship controls with rotation and thrust
- Shooting mechanics with cooldown system
- Asteroids that split into smaller pieces when shot
- Collision detection between ship, asteroids, and shots
- Progressive difficulty as asteroids continuously spawn

## Installation

1. Ensure you have Python installed on your system
2. Clone this repository
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## How to Play

Run the game using:

```bash
python main.py
```

### Controls

- **↑ (Up Arrow)**: Move forward
- **↓ (Down Arrow)**: Move backward
- **← (Left Arrow)**: Rotate counter-clockwise
- **→ (Right Arrow)**: Rotate clockwise
- **Space**: Shoot

### Game Rules

- Navigate your triangular spaceship through space
- Shoot asteroids to break them into smaller pieces
- Large asteroids break into medium ones, medium into small ones
- Avoid colliding with asteroids
- Game ends if your ship collides with an asteroid

## Technical Details

The game is built using:

- Python 3
- Pygame 2.6.1
- Object-oriented programming principles
- Vector-based movement and collision detection

## Project Structure

- `main.py` - Game initialization and main loop
- `player.py` - Player spaceship logic
- `asteroid.py` - Asteroid object implementation
- `shot.py` - Projectile mechanics
- `asteroidfield.py` - Asteroid spawning and management
- `circleshape.py` - Base class for circular objects
- `constants.py` - Game constants and configuration
