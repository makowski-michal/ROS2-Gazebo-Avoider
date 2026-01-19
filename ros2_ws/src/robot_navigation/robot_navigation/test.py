#!/usr/bin/env python3
import rclpy
import numpy as np
import torch
import os
import time
from robot_env import RobotEnv
from sac_agent import SAC

def test_agent(model_name, num_episodes=10):
    print(f"--- TESTING MODEL: {model_name} ---")
    
    if not rclpy.ok():
        rclpy.init()
    
    env = RobotEnv(config_path="config.yaml")
    
    state_dim = 76  
    action_dim = 2
    max_action = 1.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sac = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        load_model=True,
        save_directory="./models",
        model_name=model_name
    )
    
    stats = {
        "success": 0,
        "collision_chaser": 0,
        "collision_obstacle": 0,
        "timeout": 0,
        "cp1": 0,
        "cp2": 0,
        "rewards": []
    }
    
    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = sac.act(obs, sample=False) # no exploration
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                obs = next_obs
                
                if terminated or truncated:
                    done = True
                    
                    if info.get('success', False):
                        stats["success"] += 1
                    elif info.get('collision', False):
                        reason = info.get('reason', 'unknown')
                        if 'chaser' in reason:
                            stats["collision_chaser"] += 1
                        else:
                            stats["collision_obstacle"] += 1
                    elif truncated:
                        stats["timeout"] += 1
            
            if env.cp1_collected: stats["cp1"] += 1
            if env.cp2_collected: stats["cp2"] += 1
            stats["rewards"].append(episode_reward)

    except KeyboardInterrupt:
        print("\nTesting intrrupted.")
    
    finally:
        print(f"\n{'='*50}")
        print(f"TEST SUMMARIZATION ({num_episodes} EPISODES)")
        print(f"{'='*50}")

        total = max(1, len(stats['rewards']))
        success_rate = (stats['success'] / total) * 100
        col_obs_rate = (stats['collision_obstacle'] / total) * 100
        timeout_rate = (stats['timeout'] / total) * 100
        cp1_rate = (stats['cp1'] / total) * 100
        cp2_rate = (stats['cp2'] / total) * 100
        avg_reward = np.mean(stats['rewards']) if stats['rewards'] else 0.0

        print(f"SUCCESS RATE:      {success_rate:.1f}%")
        print(f"Collision Rate:     {col_obs_rate:.1f}%")
        print(f"Timeout Rate:       {timeout_rate:.1f}%")
        print(f"---")
        print(f"Collected CP1 Rate:                {cp1_rate:.1f}%")
        print(f"Collected CP2 Rate:                {cp2_rate:.1f}%")
        print(f"AVG Reward:            {avg_reward:.2f}")
        print(f"{'='*50}")
        
        env.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    MODEL_TO_TEST = "sac_3checkpoint_ep2500"
    
    EPISODES = 10
    
    test_agent(MODEL_TO_TEST, num_episodes=EPISODES)
