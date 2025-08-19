#!/usr/bin/env python3
"""
Simple visual demonstration showing Fetch robot moving with actual control.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

def simple_fetch_demo():
    """Simple demo that moves the Fetch robot arm."""
    print("🤖 Simple Fetch Robot Movement Demo")
    
    # Create environment with visual rendering
    env = gym.make("FetchPickAndPlace-v4", render_mode="human")
    obs, info = env.reset()
    
    print("✓ Environment initialized with visual rendering")
    print("📺 You should see the Fetch robot in a window")
    
    # Define some basic movements
    movements = [
        [0.2, 0.0, 0.0, 0.0, "Move arm joint 1 forward"],
        [-0.2, 0.3, 0.0, 0.0, "Move joints 1&2"],
        [0.0, -0.3, 0.2, 0.0, "Move joints 2&3"],
        [0.0, 0.0, -0.2, 0.0, "Return joint 3"],
        [0.0, 0.0, 0.0, 1.0, "Open gripper"],
        [0.0, 0.0, 0.0, -1.0, "Close gripper"],
        [0.0, 0.0, 0.0, 0.0, "Stop movement"]
    ]
    
    print("\n🎯 Starting movement sequence...")
    
    for i, (action1, action2, action3, gripper, description) in enumerate(movements):
        print(f"{i+1}. {description}")
        
        # Create action array
        action = np.array([action1, action2, action3, gripper], dtype=np.float32)
        
        # Apply action for multiple steps to see the movement
        for step in range(30):  # 30 steps per movement
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.05)  # Small delay for smooth animation
        
        # Pause between movements
        time.sleep(0.5)
        
        # Check if episode ended
        if terminated or truncated:
            print("   Episode ended, resetting...")
            obs, info = env.reset()
    
    print("\n🎉 Movement demonstration complete!")
    print("   The robot arm moved through different joint configurations")
    
    # Keep environment open for observation
    print("\n⏸️  Demo will continue with random actions for 10 seconds...")
    
    # Random movements for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        time.sleep(0.1)
        
        if terminated or truncated:
            obs, info = env.reset()
    
    print("🔚 Demo complete! Closing environment...")
    env.close()

if __name__ == "__main__":
    simple_fetch_demo()