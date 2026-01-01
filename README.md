# Project Overview
This project implements a robot navigation system using Deep Reinforcement Learning (DRL) within a ROS2 (Robot Operating System) and Gazebo simulation environment.
The goal of the agent ("Simple Robot - Avoider") is not just to go from point A to point B. It has to complete a task with several steps, move through a messy room, and most importantly avoid a moving enemy robot (“Chaser”) that patrols the area.
The agent is trained using the Soft Actor-Critic (SAC) algorithm, allowing it to learn continuous control policies (linear and angular velocity) from scratch based on Lidar sensor data.

The Environment
The simulation takes place in a closed 9x9 meter room containing:
 - Static Obstacles: Several boxes placed to block direct paths,
 - The Avoider: Main robot equipped with a 360-degree Lidar sensor,
 - The Chaser: A hostile robot that moves in a predefined, fixed circular path, creating a dynamic threat,
 - Objectives:
  1. Start at the Base (green square),
  2. Collect Checkpoint 1 (CP1),
  3. Collect Checkpoint 2 (CP2),
  4. Return safely to Base.

# Map Visualization:
# ZDJECIE MAPY

# Key Features:
Continuous Control: The robot controls its speed and turning angle smoothly, rather than choosing from a discrete list of moves (like "turn left", "go forward").
Dynamic Avoidance: Unlike standard pathfinding, the robot learns to predict and avoid a moving adversary (the Chaser).
Multi-Objective Mission: The state machine handles sequential goals (CP1 -> CP2 -> Base), requiring the robot to adapt its pathing logic during the episode.
Custom Gym Environment: robot_env.py handles sensor processing, reward calculation, and simulation resetting.
Robust Reward System:
 * +600 for Mission Complete,
 * +100 for collecting a Checkpoint,
 * -100 for collisions (Walls/Boxes/Chaser),
 * Dense Reward: Constant feedback based on distance to the current target to guide and fasten the learning process.

# The Algorithm (Soft Actor-Critic)
I use SAC, an off-policy actor-critic deep RL algorithm. It was chosen because it maximizes the trade-off between expected reward and entropy (randomness), encouraging the robot to explore the environment thoroughly during the early stages of training. Also, while doing research on similar works, this algorithm was used the most and proposed as the best one in this case. Avoider (main robot) has buffer experience of 1,000,000 time steps, what helps him learn more efficiently, not only based on current run, but also on a lot experience gained in the past (comparing different routes, strategies, speeds, etc.).
Input (Observation Space): 76 dimensions
 - 72 Lidar rays (normalized distance)
 - Relative coordinates to the current target (Distance & Angle)
 - Relative coordinates to the Chaser robot
Output (Action Space): 2 continuous values
 - Linear Velocity (Forward/Backward)
 - Angular Velocity (Turn Left/Right)

# Training Results (after 2500 Episodes)
The final model was trained over 2500 episodes. Below is the report on the agent's performance progression.

Learning Curve
Initially, the robot struggled with collisions, often hitting walls or the Chaser. Around episode 800-1000, the agent began to consistently understand the concept of checkpoints. By episode 2200+, the success rate increased significantly as the robot learned to balance speed with safety.

Final Performance Stats
After training, the model achieves the following metrics:
Success Rate: ~50% (Mission Complete)
Collision Rate: Reduced significantly compared to early episodes
Average Steps: The robot learned to take optimized paths, reducing the time needed to complete the loop
![results](https://github.com/user-attachments/assets/a701f32b-e57f-4fa6-a7db-250a0719b4b6)


# Demo Run
Here is a demonstration of the fully trained agent navigating the environment, collecting all checkpoints, and successfully avoiding the Chaser.
# VIDEO RUN

# How to Run
Prerequisites:
 - Ubuntu 20.04 / 22.04
 - ROS2 (Humble or Foxy)
 - Gazebo
 - Python 3 + PyTorch

Installation:
1. Clone the repository to your ROS2 workspace:
`` cd ~/ros2_ws/src
git clone https://github.com/makowski-michal/ROS2-Gazebo-Avoider.git
cd ..
colcon build
source install/setup.bash ``
2. To start training the robot from scratch (or resume from a checkpoint):
`` python3 train.py ``

# Future Improvements:
1. Training: More training time to improve the success rate to 80-90%.
2. Complex Chaser AI: Currently, the chaser moves in a circle. Future versions could implement a chaser that actively hunts the player.
3. Real World Transfer: Porting the trained policy to a physical TurtleBot or similar platform.
