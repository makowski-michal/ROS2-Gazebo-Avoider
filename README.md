# Project Overview
**This project implements a robot navigation system using Deep Reinforcement Learning (DRL) within a ROS2 (Robot Operating System) and Gazebo simulation environment.**<br>
The goal of the agent ("Simple Robot - Avoider") is not just to go from point A to point B. It has to complete a task with several steps, move through a messy room, and most importantly avoid a moving enemy robot (“Chaser”) that patrols the area.<br>
The agent is trained using the Soft Actor-Critic (SAC) algorithm, allowing it to learn continuous control policies (linear and angular velocity) from scratch based on Lidar sensor data.<br>

# The Environment
The simulation takes place in a closed 9x9 meter room containing:<br>
 - Static Obstacles: Several boxes placed to block direct paths,<br>
 - The Avoider: Main robot equipped with a 360-degree Lidar sensor,<br>
 - The Chaser: A hostile robot that moves in a predefined, fixed circular path, creating a dynamic threat,<br>
 - Objectives:
   1. Start at the Base (green square),
   2. Collect Checkpoint 1 (CP1),
   3. Collect Checkpoint 2 (CP2),
   4. Return safely to Base.

# Map Visualization:
![current_map](https://github.com/user-attachments/assets/a95ab80c-39db-4573-8dbf-76541be47dd7)


# Key Features:
1. Continuous Control: The robot controls its speed and turning angle smoothly, rather than choosing from a discrete list of moves (like "turn left", "go forward").<br>
2. Dynamic Avoidance: Unlike standard pathfinding, the robot learns to predict and avoid a moving adversary (the Chaser).<br>
3. Multi-Objective Mission: The state machine handles sequential goals (CP1 -> CP2 -> Base), requiring the robot to adapt its pathing logic during the episode.<br>
4. Custom Gym Environment: robot_env.py handles sensor processing, reward calculation, and simulation resetting.<br>
5. Robust Reward System:
   * +600 for Mission Complete,<br>
   * +100 for collecting a Checkpoint,<br>
   * -100 for collisions (Walls/Boxes/Chaser),<br>
6. Dense Reward: Constant feedback based on distance to the current target to guide and fasten the learning process,<br>
7. Possibility of stopping and resuming trainings thanks to .pth files saved every 100 episodes.<br>

# The Algorithm (Soft Actor-Critic)
I used SAC, an off-policy actor-critic deep RL algorithm. It was chosen because it maximizes the trade-off between expected reward and entropy (randomness), encouraging the robot to explore the environment thoroughly during the early stages of training. Also, while doing research on similar works, this algorithm was used the most and proposed as the best one in this case. Avoider (main robot) has buffer experience of 1,000,000 time steps, which helps him learn more efficiently, not only based on current run, but also on a lot of experience gained in the past (comparing different routes, strategies, speeds, etc.).

**Input (Observation Space): 76 dimensions**
- 72 Lidar rays (normalized distance)
- Relative coordinates to the current target (Distance & Angle)
- Relative coordinates to the Chaser robot

**Output (Action Space): 2 continuous values**
- Linear Velocity (Forward/Backward)
- Angular Velocity (Turn Left/Right)

# Training Results (after 2500 Episodes)
**The final model was trained over 2500 episodes. Below is the report on the agent's performance progression.**<br>

**Learning Curve**<br>
Initially, the robot struggled with collisions, often hitting walls or the Chaser. Around episode 800-1000, the agent began to consistently understand the concept of checkpoints. By episode 2200+, the success rate increased significantly as the robot learned to balance speed with safety.<br>

**Final Performance Stats**<br>
After training, the model achieves the following metrics:<br>
Success Rate: ~60-75% (Mission Complete) during long periods of training<br>
Collision Rate: Reduced significantly compared to early episodes<br>
Average Steps: The robot learned to take optimized paths, reducing the time needed to complete the loop <br>

![results2](https://github.com/user-attachments/assets/6e3ad926-e52f-43cb-a2ca-0a5a3c84c5fe)

# Video of Successful Mission Run
Here is a sped-up demonstration of the fully trained agent navigating the environment, collecting all checkpoints, and successfully avoiding the Chaser.

https://github.com/user-attachments/assets/31775b05-3f3d-4113-a604-2e59881efd08

# Video of Robot's Crash and Mission Restart

https://github.com/user-attachments/assets/18c7c541-13a8-4cb3-87ba-e3b5ba7a2669

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
3. Tougher Area: Adding second chaser robot.
4. Different Scenarios: Implementing different maps, with different placements of static boxes and different chasers' fixed paths.
