#!/usr/bin/env python3
"""
Interactive demonstration of Claude's Direct Control System with live robot movement.
This shows Claude's mathematical control in action with visual feedback.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import json
from dcs.core.fetch_claude_controller import FetchClaudeController

def interactive_claude_demo():
    """Interactive demo showing Claude's direct control with actual robot movement."""
    print("=" * 60)
    print("🧠 Claude's Direct Control System - LIVE DEMO")
    print("=" * 60)
    print("This demo shows Claude's mathematical control with actual robot movement")
    
    # Create environment with visual rendering
    env = gym.make("FetchPickAndPlace-v4", render_mode="human")
    obs, info = env.reset()
    
    # Initialize Claude's controller
    claude = FetchClaudeController(
        env.unwrapped.model,
        env.unwrapped.data,
        verbose=True
    )
    
    print("✅ Visual environment ready - you should see the Fetch robot!")
    print("🧠 Claude's control system initialized")
    
    # Get initial state
    current_state = claude.get_current_state()
    ee_pos = current_state['end_effector']['position']
    print(f"📍 Initial position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")
    
    # Demo sequences
    print("\n🎭 Starting Claude's Control Demonstrations...")
    
    demonstrations = [
        {
            "name": "Gripper Control",
            "action": "gripper_demo",
            "description": "Claude directly calculates gripper movements"
        },
        {
            "name": "Simple Movement",
            "action": "simple_movement",
            "description": "Claude calculates IK for small position changes"
        },
        {
            "name": "Path Planning",
            "action": "path_demo", 
            "description": "Claude plans and executes a multi-waypoint path"
        }
    ]
    
    for i, demo in enumerate(demonstrations):
        print(f"\n{i+1}. 🎯 {demo['name']}")
        print(f"   {demo['description']}")
        input("   Press Enter to start this demonstration...")
        
        if demo["action"] == "gripper_demo":
            gripper_demonstration(claude, env)
        elif demo["action"] == "simple_movement":
            simple_movement_demo(claude, env)
        elif demo["action"] == "path_demo":
            path_planning_demo(claude, env)
        
        print(f"   ✅ {demo['name']} demonstration complete!")
        time.sleep(1)
    
    print("\n🎉 All Claude demonstrations complete!")
    print("   Claude's Direct Control System provides:")
    print("   • Mathematical precision in control")
    print("   • Real-time inverse kinematics")
    print("   • Explicit reasoning for each action")
    print("   • No learning required - instant control")
    
    # Save session data
    claude.save_session("interactive_demo_session.json")
    print(f"\n📁 Session data saved to interactive_demo_session.json")
    
    print("\n⏸️  Press Enter to close the demo...")
    input()
    env.close()

def gripper_demonstration(claude, env):
    """Demonstrate Claude's gripper control."""
    print("   🤖 Claude analyzing gripper state...")
    
    # Get current gripper state
    state = claude.get_current_state()
    gripper_open = state['joints']['gripper_open']
    
    print(f"   Current gripper: {'Open' if gripper_open else 'Closed'}")
    
    # Open gripper
    print("   💭 Claude: Opening gripper with calculated positions...")
    success, msg = claude.control_gripper(True)
    apply_gripper_action(env, 1.0)  # Open gripper
    time.sleep(2)
    
    # Close gripper  
    print("   💭 Claude: Closing gripper with calculated positions...")
    success, msg = claude.control_gripper(False)
    apply_gripper_action(env, -1.0)  # Close gripper
    time.sleep(2)

def simple_movement_demo(claude, env):
    """Demonstrate simple movement calculation."""
    print("   🎯 Claude calculating small position adjustment...")
    
    # Get current position
    current_pos, _ = claude.ik_solver.get_current_ee_pose()
    print(f"   Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    
    # Calculate small movement
    target_pos = current_pos + np.array([0.05, 0.02, -0.03])  # Small adjustment
    print(f"   Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Claude calculates the movement
    success, joint_solution, error = claude.ik_solver.solve_ik(target_pos)
    
    if success:
        print(f"   ✅ Claude solved IK with error: {error:.6f}")
        print("   💭 Claude: Applying calculated joint angles...")
        
        # Apply the calculated joint solution gradually
        current_joints = claude.ik_solver.get_current_joint_angles()
        for step in range(30):
            alpha = step / 29.0
            interpolated = (1 - alpha) * current_joints[:3] + alpha * joint_solution[:3]
            
            # Convert to action space
            action = np.zeros(4)
            action[:3] = np.clip(interpolated, -1, 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(0.1)
    else:
        print(f"   ❌ Claude's IK calculation failed with error: {error:.6f}")

def path_planning_demo(claude, env):
    """Demonstrate path planning calculation."""
    print("   🗺️ Claude planning multi-waypoint path...")
    
    # Define a simple path within the robot's reach
    current_pos, _ = claude.ik_solver.get_current_ee_pose()
    
    waypoints = [
        current_pos + np.array([0.02, 0.05, 0.0]),
        current_pos + np.array([0.0, 0.05, 0.02]),
        current_pos + np.array([-0.02, 0.0, 0.02]),
        current_pos  # Return to start
    ]
    
    print(f"   💭 Claude: Calculating path through {len(waypoints)} waypoints...")
    
    for i, waypoint in enumerate(waypoints):
        print(f"   Waypoint {i+1}: [{waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f}]")
        
        # Claude calculates IK for this waypoint
        success, joint_solution, error = claude.ik_solver.solve_ik(waypoint)
        
        if success:
            print(f"   ✅ IK solved for waypoint {i+1} (error: {error:.6f})")
            
            # Move to waypoint smoothly
            current_joints = claude.ik_solver.get_current_joint_angles()
            for step in range(20):  # Smooth movement
                alpha = step / 19.0
                interpolated = (1 - alpha) * current_joints[:3] + alpha * joint_solution[:3]
                
                action = np.zeros(4)
                action[:3] = np.clip(interpolated, -1, 1)
                
                obs, reward, terminated, truncated, info = env.step(action)
                time.sleep(0.05)
        else:
            print(f"   ❌ Waypoint {i+1} unreachable (error: {error:.6f})")
        
        time.sleep(0.5)

def apply_gripper_action(env, gripper_value):
    """Apply gripper action to environment."""
    for _ in range(20):
        action = np.array([0.0, 0.0, 0.0, gripper_value])
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.05)

if __name__ == "__main__":
    interactive_claude_demo()