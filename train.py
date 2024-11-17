import os
import random; random.seed(0)
import numpy as np; np.random.seed(0)

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import torch
import clip

is_cuda = torch.cuda.is_available()

if is_cuda: device = torch.device('cuda')
else: device = torch.device('cpu')


# Import Resnet-based Clip model
clip_model, preprocess = clip.load("RN50", device=device) 

def process_inputs(frame, instruction):
    
    image = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
    
    text = clip.tokenize([instruction]).to(device)
    
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return image_features, text_features

def random_policy(controller, action_space, instruction, num_steps):

    for step in range(num_steps):
    
        action = random.choice(action_space)
        
        if action == "MoveAgent":
            controller.step(
                action="MoveAgent",
                ahead=random.uniform(-0.25, 0.25), # Move forward/backward
                right=random.uniform(-0.25, 0.25), # Move left/right
                returnToStart=False
            )
      
        elif action == "RotateAgent":
            controller.step(
                action="RotateAgent",
                degrees=random.choice([30, -30]), # Rotate left/right
                returnToStart=False
            )
        elif action == "MoveArm":
           
            random_position = {
                "x": random.uniform(-0.5, 0.5),
                "y": random.uniform(0.0, 1.0),
                "z": random.uniform(-0.5, 0.5),
            }
            controller.step(
                action="MoveArm",
                position=random_position,
                coordinateSpace="armBase",
                restrictMovement=False,
                returnToStart=False
            )
            
        elif action == "MoveArmBase":
            controller.step(
                action="MoveArmBase",
                y=random.uniform(0.0, 1.0), # Move arm base up/down
                returnToStart=False
            )
            
        elif action == "LookUp":
            controller.step(action="LookUp")
            
        elif action == "LookDown":
            controller.step(action="LookDown")
        
        current_frame = controller.last_event.frame
        image_features, text_features = process_inputs(current_frame, instruction)
        
        similarity = torch.cosine_similarity(image_features, text_features)
        print(f"Step {step + 1}: Action={action}, Instruction: \"{instruction}\", Similarity={similarity.item():.4f}")
        
        plt.imshow(controller.last_event.frame)
        plt.title("Agent's View")
        plt.axis("off")
        plt.show()
        
def train_ppo(controller, ppo_agent, action_space, state_dim, num_episodes=1000, max_timesteps=200):
    
    """
    Train the PPO agent in the AI2-THOR environment.

    Parameters:
    - controller: AI2-THOR controller
    - ppo_agent: Instance of PPOAgent
    - action_space: List of possible actions
    - state_dim: Dimensionality of the state representation
    - num_episodes: Number of training episodes
    - max_timesteps: Maximum timesteps per episode
    """

    for episode in range(num_episodes):
    
        controller.reset("FloorPlan20")
        state = np.zeros(state_dim) # Replace with actual state representation logic
        total_reward = 0

        for timestep in range(max_timesteps):
            action_idx = ppo_agent.select_action(state)
            action = action_space[action_idx]

            # Perform the action in AI2-THOR
            event = controller.step(action=action)
            next_state = np.zeros(state_dim) # Replace with actual state representation logic
            reward = 0 # Replace with appropriate reward function
            done = False # Replace with terminal state logic

            # Store in memory
            ppo_agent.memory.rewards.append(reward)
            ppo_agent.memory.is_terminals.append(done)

            # Update state
            state = next_state
            total_reward += reward

            # Break if done
            if done:break

        # PPO Update
        if (episode + 1) % ppo_agent.update_timestep == 0:
            ppo_agent.update()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")
