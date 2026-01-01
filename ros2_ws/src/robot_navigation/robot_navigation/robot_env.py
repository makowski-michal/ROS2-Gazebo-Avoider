#!/usr/bin/env python3
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import yaml
import time
import math


class RobotEnv(gym.Env, Node):
    def __init__(self, config_path="config.yaml"):
        gym.Env.__init__(self)
        Node.__init__(self, 'robot_env_node')
        
        with open(config_path, 'r') as f: # load config file
            self.cfg = yaml.safe_load(f)
        
        env = self.cfg['environment'] # load environment settings
        self.max_steps = env['max_steps']
        self.base_pos = np.array(env['base_position'], dtype=np.float32)
        self.cp1_pos = np.array(env['checkpoint_1'], dtype=np.float32)
        self.cp2_pos = np.array(env['checkpoint_2'], dtype=np.float32)
        self.cp_radius = env['checkpoint_radius']
        self.collision_threshold = env['collision_threshold']
        
        rw = self.cfg['rewards'] # load reward values
        self.R_CP1 = rw['checkpoint_collected']
        self.R_CP2 = rw['checkpoint_collected']
        self.R_MISSION = rw['mission_complete']
        self.R_PROGRESS = rw['progress_scale']
        self.R_COLLISION = rw['collision_penalty']
        
        obs_cfg = self.cfg['observation'] # load observation settings
        self.lidar_samples = obs_cfg['lidar_samples']
        self.lidar_max = obs_cfg['lidar_max_range']
        
        # Observation: lidar (72) + dx_cp1, dy_cp1 + dx_cp2, dy_cp2 + dx_base, dy_base + dx_chaser, dy_chaser + cp1_flag + cp2_flag = 82
        self.observation_space = spaces.Box( # define observation space shape
            low=-np.inf, high=np.inf,
            shape=(self.lidar_samples + 10,),
            dtype=np.float32
        )
        
        act = self.cfg['action'] # define action space
        self.action_space = spaces.Box(
            low=np.array([act['min_linear_speed'], act['min_angular_speed']], dtype=np.float32),
            high=np.array([act['max_linear_speed'], act['max_angular_speed']], dtype=np.float32),
            dtype=np.float32
        )
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10) # create publishers and subscribers
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
        
        self.create_subscription(Odometry, '/chaser_robot/odom', self.chaser_odom_cb, 10) # chaser robot topics
        self.chaser_cmd_pub = self.create_publisher(Twist, '/chaser_robot/cmd_vel', 10)
        
        # chaser movement parameters
        self.chaser_target_radius = 1.75
        self.chaser_angular_velocity = -0.3  # minus = counter clock wise
        self.chaser_linear_velocity = abs(self.chaser_angular_velocity) * self.chaser_target_radius
        self.chaser_enabled = self.cfg['chaser']['enabled']
        self.chaser_start_pos = np.array([1.75, 0.0], dtype=np.float32)
        
        self.set_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state') # gazebo service for teleport
        service_ready = self.set_state_client.wait_for_service(timeout_sec=5.0)
        if not service_ready:
            self.get_logger().error('Gazebo service not available')
        
        self.robot_pos = np.zeros(2, dtype=np.float32) # robot state
        self.robot_turn = 0.0
        self.lidar_data = None
        self.robot_odom_data = None
        
        self.chaser_pos = np.zeros(2, dtype=np.float32) # chaser state
        self.chaser_collision_radius = 0.55
        
        self.step_count = 0 # episode state
        self.episode_count = 0
        self.cp1_collected = False
        self.cp2_collected = False
        self.prev_dist_to_target = None
        self.total_reward = 0.0

        self.min_steps_for_cp1 = 10 # step limits before counting success, prevents bugs
        self.min_steps_for_cp2 = 20
        self.min_steps_for_success = 30

        self.success_count = 0 # statistics
        self.collision_count = 0
        self.timeout_count = 0
        self.cp1_collected_count = 0
        self.cp2_collected_count = 0
        
        self.episode_rewards = [] # logs for rewards
        self.pending_futures = []
    
    def lidar_cb(self, msg):  # store lidar readings
        self.lidar_data = np.array(msg.ranges, dtype=np.float32)
        self.lidar_data = np.nan_to_num(self.lidar_data, nan=self.lidar_max, posinf=self.lidar_max)
        self.lidar_data = np.clip(self.lidar_data, 0.0, self.lidar_max)
    
    def odom_cb(self, msg): # store robot position and heading
        self.robot_odom_data = msg
        pos = msg.pose.pose.position
        self.robot_pos = np.array([pos.x, pos.y], dtype=np.float32)
        
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_turn = math.atan2(siny, cosy)
    
    def chaser_odom_cb(self, msg): # store chaser position
        pos = msg.pose.pose.position
        self.chaser_pos = np.array([pos.x, pos.y], dtype=np.float32)
    
    def _cleanup_futures(self): # remove finished async service calls
        self.pending_futures = [f for f in self.pending_futures if not f.done()]
    
    def _send_teleport_async(self, name, x, y, z, yaw): # send async teleport command to gazebo
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = float(z)
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.twist.linear.x = 0.0
        req.state.twist.linear.y = 0.0
        req.state.twist.linear.z = 0.0
        req.state.twist.angular.x = 0.0
        req.state.twist.angular.y = 0.0
        req.state.twist.angular.z = 0.0
        
        future = self.set_state_client.call_async(req)
        self.pending_futures.append(future)
        return future
    
    def _do_reset(self): # reset robot and chaser state
        self._cleanup_futures()
        
        for _ in range(10):
            self.send_cmd(0, 0)
            self.send_chaser_cmd(0, 0)
        
        dx = self.cp1_pos[0] - self.base_pos[0]
        dy = self.cp1_pos[1] - self.base_pos[1]
        robot_yaw = math.atan2(dy, dx)
        
        self._send_teleport_async('simple_robot', self.base_pos[0], self.base_pos[1], 0.1, robot_yaw)
        self._send_teleport_async('chaser_robot', 1.75, 0.0, 0.15, -math.pi / 2.0)
        
        end_time = time.time() + 0.5
        while time.time() < end_time:
            rclpy.spin_once(self, timeout_sec=0.02)
            self.send_cmd(0, 0)
            self.send_chaser_cmd(0, 0)
    
    def reset(self, seed=None, options=None): # reset environment at episode start
        super().reset(seed=seed)
        
        self.chaser_enabled = False
        self._do_reset()
        
        self.step_count = 0
        self.cp1_collected = False
        self.cp2_collected = False
        self.total_reward = 0.0
        self.episode_count += 1
        self.prev_dist_to_target = np.linalg.norm(self.robot_pos - self.cp1_pos)
        
        self.chaser_enabled = self.cfg['chaser']['enabled']
        
        return self.get_obs(), {}
    
    def step(self, action): # one simulation step
        self.step_count += 1
        
        self.send_cmd(action[0], action[1])
        self.control_chaser()
        
        time.sleep(0.05)
        rclpy.spin_once(self, timeout_sec=0.01)
        
        obs = self.get_obs()
        reward = self.get_reward()
        
        terminated, info = self._check_done()
        truncated = (self.step_count >= self.max_steps)
        
        if info.get('collision', False):
            reward += self.R_COLLISION
         
        if info.get('success', False):
            reward += self.R_MISSION
        
        self.total_reward += reward
        
        if terminated or truncated:
            for _ in range(5):
                self.send_cmd(0, 0)
                self.send_chaser_cmd(0, 0)
                rclpy.spin_once(self, timeout_sec=0.01)
            
            self.chaser_enabled = False
            self.episode_rewards.append(self.total_reward)
            self._print_summary(terminated, truncated, info)
        
        return obs, reward, terminated, truncated, info
    
    def send_cmd(self, linear, angular): # send movement command to robot
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_pub.publish(cmd)
    
    def send_chaser_cmd(self, linear, angular): # send movement command to chaser
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.chaser_cmd_pub.publish(cmd)
    
    def control_chaser(self): # control chaser logic
        if self.chaser_enabled:
            self.send_chaser_cmd(self.chaser_linear_velocity, self.chaser_angular_velocity)
        else:
            self.send_chaser_cmd(0.0, 0.0)
    
    def get_obs(self): # build observation vector
        # lidar readings
        if self.lidar_data is None or len(self.lidar_data) == 0:
            lidar = np.ones(self.lidar_samples, dtype=np.float32) * self.lidar_max
        else:
            indices = np.linspace(0, len(self.lidar_data)-1, self.lidar_samples, dtype=int)
            lidar = self.lidar_data[indices]
        
        lidar_norm = lidar / self.lidar_max
        
        # choose active target
        if not self.cp1_collected:
            target = self.cp1_pos
        elif not self.cp2_collected:
            target = self.cp2_pos
        else:
            target = self.base_pos
            
        # calculate distance and angle to target
        dx = target[0] - self.robot_pos[0]
        dy = target[1] - self.robot_pos[1]
        
        dist = np.linalg.norm([dx, dy])
        
        global_angle = math.atan2(dy, dx)
        relative_angle = global_angle - self.robot_turn
        relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi
        
        # calculate vector to chaser
        dx_chaser = self.chaser_pos[0] - self.robot_pos[0]
        dy_chaser = self.chaser_pos[1] - self.robot_pos[1]
        
        # combine observation values
        obs = np.concatenate([
            lidar_norm,
            [dist / 10.0, relative_angle],
            [dx_chaser / 10.0, dy_chaser / 10.0]
        ]).astype(np.float32)
        
        return obs
    
    def get_reward(self): # calculate reward value
        reward = 0.0
        
        if not self.cp1_collected:
            current_target = self.cp1_pos
        elif not self.cp2_collected:
            current_target = self.cp2_pos
        else:
            current_target = self.base_pos
        
        current_dist = np.linalg.norm(self.robot_pos - current_target)
        
        if self.prev_dist_to_target is not None:
            delta = self.prev_dist_to_target - current_dist
            progress_reward = delta * self.R_PROGRESS
            progress_reward = np.clip(progress_reward, -1.0, 1.0)
            reward += progress_reward
        
        self.prev_dist_to_target = current_dist
        
        if not self.cp1_collected and self.step_count >= self.min_steps_for_cp1:
            dist_cp1 = np.linalg.norm(self.robot_pos - self.cp1_pos)
            if dist_cp1 < self.cp_radius:
                self.cp1_collected = True
                reward += self.R_CP1
                self.prev_dist_to_target = np.linalg.norm(self.robot_pos - self.cp2_pos)
        
        if self.cp1_collected and not self.cp2_collected and self.step_count >= self.min_steps_for_cp2:
            dist_cp2 = np.linalg.norm(self.robot_pos - self.cp2_pos)
            if dist_cp2 < self.cp_radius:
                self.cp2_collected = True
                reward += self.R_CP2
                self.prev_dist_to_target = np.linalg.norm(self.robot_pos - self.base_pos)
        reward -= 0.05 # small time penalty to encourage faster behavior
        return reward
    
    def _check_done(self): # check if episode should end
        terminated = False
        info = {}
        
        if self.cp1_collected and self.cp2_collected and self.step_count >= self.min_steps_for_success: # check mission success
            dist_base = np.linalg.norm(self.robot_pos - self.base_pos)
            if dist_base < self.cp_radius:
                terminated = True
                info['success'] = True
                info['reason'] = 'mission_complete'
                self.total_reward += self.R_MISSION
                return terminated, info
        
        if self.step_count > 5: # check collisions
            dist_to_chaser = np.linalg.norm(self.robot_pos - self.chaser_pos)
            if dist_to_chaser < self.chaser_collision_radius:
                terminated = True
                info['collision'] = True
                info['reason'] = 'collision_with_chaser'
                return terminated, info
            
            if self.lidar_data is not None:
                min_dist = np.min(self.lidar_data)
                if min_dist < self.collision_threshold:
                    terminated = True
                    info['collision'] = True
                    info['reason'] = 'collision_with_obstacle'
                    return terminated, info
        
        return terminated, info
    
    def _print_summary(self, terminated, truncated, info): # print episode summary to console
        self.get_logger().info('='*60)
        self.get_logger().info(f'EPISODE {self.episode_count} SUMMARY')
        
        if info.get('success', False):
            self.success_count += 1
            self.get_logger().info('RESULT: SUCCESS - MISSION COMPLETE!')
        elif info.get('collision', False):
            self.collision_count += 1
            reason = info.get('reason', 'collision')
            self.get_logger().info(f'RESULT: COLLISION ({reason})')
        elif truncated:
            self.timeout_count += 1
            self.get_logger().info('RESULT: TIMEOUT')
        
        self.get_logger().info(f'Steps: {self.step_count}/{self.max_steps}')
        self.get_logger().info(f'Total reward: {self.total_reward:.2f}')
        
        cp1_status = "YES" if self.cp1_collected else "NO"
        cp2_status = "YES" if self.cp2_collected else "NO"
        self.get_logger().info(f'CP1 collected: {cp1_status}, CP2 collected: {cp2_status}')
        
        if not self.cp1_collected:
            target = self.cp1_pos
            target_name = "CP1"
        elif not self.cp2_collected:
            target = self.cp2_pos
            target_name = "CP2"
        else:
            target = self.base_pos
            target_name = "BASE"
        
        dist_to_target = np.linalg.norm(self.robot_pos - target)
        self.get_logger().info(f'Distance to {target_name}: {dist_to_target:.2f}m')
        
        dist_to_chaser = np.linalg.norm(self.robot_pos - self.chaser_pos)
        self.get_logger().info(f'Distance to chaser: {dist_to_chaser:.2f}m')
        
        if self.cp1_collected:
            self.cp1_collected_count += 1
        if self.cp2_collected:
            self.cp2_collected_count += 1
        
        total_episodes = self.episode_count
        if total_episodes > 0:
            success_rate = (self.success_count / total_episodes) * 100.0
            collision_rate = (self.collision_count / total_episodes) * 100.0
            timeout_rate = (self.timeout_count / total_episodes) * 100.0
            cp1_collection_rate = (self.cp1_collected_count / total_episodes) * 100.0
            cp2_collection_rate = (self.cp2_collected_count / total_episodes) * 100.0

            self.get_logger().info(f'--- OVERALL STATS ({total_episodes} episodes) ---')
            self.get_logger().info(f'Success rate: {success_rate:.1f}%')
            self.get_logger().info(f'CP1 collection rate: {cp1_collection_rate:.1f}%')
            self.get_logger().info(f'CP2 collection rate: {cp2_collection_rate:.1f}%')
            self.get_logger().info(f'Collision rate: {collision_rate:.1f}%')
            self.get_logger().info(f'Timeout rate: {timeout_rate:.1f}%')
        
        self.get_logger().info('='*60)
    
    def close(self): # stop robot and chaser and destroy node
        self.send_cmd(0, 0)
        self.send_chaser_cmd(0, 0)
        self.destroy_node()
