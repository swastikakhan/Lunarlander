# Lunarlander
This project implements an agent to solve the [LunarLander-v3](https://www.gymlibrary.dev/environments/box2d/lunar_lander/) environment using Particle Swarm Optimization (PSO). A simple linear policy maps game state to action, and PSO optimizes this policy to maximize reward. It has a linear policy, it maps observation vector (8-dimensions) to action logits (4-dimensions) using a simple linear transformation.


## 📁 **File Overview**

 best_policy.npy - Saved policy weights (parameters) found using PSO. 
 
 evaluate_agent.py - Evaluates any agent's average reward over 100 episodes.
 
 evaluate.bat - Runs evaluation script with your policy and weights
 
 my_policy.py - Defines how the agent selects an action using trained policy weights.
 
 train_agent.py - Training script: runs PSO and saves best policy.
 
 play_lunar_lander.py - Play the game manually using keyboard controls (W, A, D, S).
