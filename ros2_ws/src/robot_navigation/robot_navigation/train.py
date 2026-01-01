#!/usr/bin/env python3
import rclpy
import numpy as np
import torch
import time
import os
from robot_env import RobotEnv
from sac_agent import SAC
from replay_buffer import ReplayBuffer


def main():
    # starts the ros system if it is not already running    
    if not rclpy.ok():
        rclpy.init()
    
    # makes folders if they do not exist
    os.makedirs("models", exist_ok=True)
    
    # finds the latest saved model checkpoint
    def find_latest_checkpoint():
        """Znajduje najnowszy checkpoint w ./models/"""
        import glob
        
        # this looks for files with model checkpoints
        checkpoints = glob.glob("./models/sac_3checkpoint_ep*_actor.pth")
        # if there are no checkpoints, return nothing
        if not checkpoints:
            return None, 0
        
        # extract the episode number from each filename
        episodes = []
        for ckpt in checkpoints:
            try:
                ep_num = int(ckpt.split("ep")[1].split("_")[0])
                episodes.append((ckpt.replace("_actor.pth", ""), ep_num))
            except:
                continue
        
        if not episodes: # if this fails, then there are no valid files
            return None, 0
        
        # get the newest checkpoint (largest episode number)
        latest = max(episodes, key=lambda x: x[1])
        model_name = os.path.basename(latest[0])
        return model_name, latest[1]
    
    resume_model_name, start_episode = find_latest_checkpoint() # try to resume training from the last checkpoint
    resume_training = (resume_model_name is not None)
    resume_buffer_path = "./models/replay_buffer_3cp.pkl"
    
    if resume_training:
        print(f"Found checkpoint: {resume_model_name} at episode {start_episode}")
    else:
        print("No checkpoint found, starting from zero")
    
    state_dim = 76  # 72 lidar + 2*cp1 + 2*cp2 + 2*base + 2*chaser + 2 flags
    action_dim = 2
    max_action = 1.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}") # creates the robot environment
    
    env = RobotEnv(config_path="config.yaml")
    
    sac = SAC( # this creates the sac agent that learns to control the robot
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        max_action=max_action,
        discount=0.99,
        init_temperature=0.1,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        critic_tau=0.005,
        actor_update_frequency=1,
        critic_target_update_frequency=2,
        learnable_temperature=True,
        save_every=100,
        load_model=resume_training,
        save_directory="./models",
        model_name=resume_model_name if resume_training else "sac_3checkpoint",
        load_directory="./models",
    )
    
    replay_buffer = ReplayBuffer(buffer_size=1000000, random_seed=42) # this creates a memory buffer to store past robot experiences
    
    # loads the memory buffer if we resume training
    if resume_training:
        replay_buffer.load(resume_buffer_path)
    
    max_episodes = 10000
    max_steps = 2000
    batch_size = 256
    train_every = 2
    training_iterations = 200
    start_training_steps = 5000
    
    total_steps = replay_buffer.size() # total steps already done (for resume)
    episode = start_episode
    
    print(f"Config: max_episodes={max_episodes}, batch_size={batch_size}")
    print(f"Training starts after {start_training_steps} steps")
    print(f"Train every {train_every} episodes for {training_iterations} iterations")
    print(f"Starting from episode {episode}, total_steps {total_steps}, buffer_size {replay_buffer.size()}")
    
    try: # main training loop over episodes
        while episode < max_episodes:
            obs, _ = env.reset() # reset the environment to start a fresh episode
            
            for step in range(max_steps): # loop over steps inside one episode
                if total_steps < start_training_steps: # before enough data is collected, use random actions
                    action = env.action_space.sample()
                else:
                    action = sac.act(obs, sample=True) # later use the trained model to choose actions
                
                next_obs, reward, terminated, truncated, info = env.step(action) # apply action in the environment and get feedback
                
                done = 1.0 if (terminated or truncated) else 0.0 # done tells if the episode is finished
                replay_buffer.add(obs, action, reward, done, next_obs) # store the experience in the replay buffer
                
                obs = next_obs # move to the next state
                total_steps += 1
                
                # once we have enough data, train a little bit every step
                if total_steps >= start_training_steps:
                    sac.train(replay_buffer, iterations=1, batch_size=batch_size)
                
                if terminated or truncated: # stop episode if robot is done
                    break
            
            episode += 1
            
            # save model every 100 episodes
            if episode % 100 == 0:
                sac.save(f"sac_3checkpoint_ep{episode}", "./models")
                replay_buffer.save("./models/replay_buffer_3cp.pkl")
        
    except KeyboardInterrupt: # this runs when user stops the program manually
        print("\nTraining interrupted")
        sac.save("sac_3checkpoint_interrupted", "./models")
        replay_buffer.save("./models/replay_buffer_3cp_interrupted.pkl")
    
    finally: # always close environment and shutdown ros
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
