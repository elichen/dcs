#!/usr/bin/env python3
"""
Simple demonstration of Claude's Direct Control System.

This shows how Claude directly controls the Fetch robot through
mathematical calculations without needing abstract AI interpretation.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import json

from dcs.core.fetch_claude_controller import FetchClaudeController

def demonstrate_claude_direct_control():
    """Demonstrate Claude's direct mathematical control of the Fetch robot."""
    
    print("=" * 60)
    print("🤖 Claude's Direct Mathematical Control Demonstration")
    print("=" * 60)
    
    # Initialize environment (no rendering for demo)
    env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array")
    obs, info = env.reset()
    
    # Initialize Claude's control system
    print("\n🧠 Initializing Claude's direct control system...")
    claude = FetchClaudeController(
        env.unwrapped.model,
        env.unwrapped.data,
        verbose=True
    )
    
    # Show current state
    print("\n📊 Current Robot State:")
    state = claude.get_current_state()
    current_pos = state['end_effector']['position']
    print(f"   End-effector position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
    print(f"   Gripper state: {'Open' if state['joints']['gripper_open'] else 'Closed'}")
    print(f"   Execution state: {state['control']['execution_state']}")
    
    # Demonstrate direct mathematical control
    print("\n🎯 Claude's Direct Control Demonstration:")
    print("   (Claude directly calculates IK solutions and path planning)")
    
    # Example 1: Direct movement calculation
    print("\n1️⃣ Direct Movement Control:")
    target_pos = [0.8, 0.0, 0.4]  # Safe position within workspace
    print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    success, msg = claude.move_to_position(target_pos, path_type="arc")
    print(f"   Result: {'✅ Success' if success else '❌ Failed'} - {msg}")
    
    # Example 2: Gripper control
    print("\n2️⃣ Direct Gripper Control:")
    success, msg = claude.control_gripper(True)  # Open gripper
    print(f"   Open gripper: {'✅ Success' if success else '❌ Failed'} - {msg}")
    
    success, msg = claude.control_gripper(False)  # Close gripper  
    print(f"   Close gripper: {'✅ Success' if success else '❌ Failed'} - {msg}")
    
    # Example 3: Pick sequence planning
    print("\n3️⃣ Complete Pick Sequence Planning:")
    object_pos = [0.6, 0.0, 0.05]  # Safe object position
    print(f"   Object position: [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")
    
    success, msg = claude.pick_object(object_pos)
    print(f"   Pick planning: {'✅ Success' if success else '❌ Failed'} - {msg}")
    
    # Show final state and action history
    print("\n📈 Final System State:")
    final_state = claude.get_current_state()
    print(f"   Actions taken: {final_state['control']['actions_taken']}")
    print(f"   Execution state: {final_state['control']['execution_state']}")
    
    action_history = claude.get_action_history()
    if action_history:
        print(f"\n📝 Action History ({len(action_history)} actions):")
        for i, action in enumerate(action_history[-3:]):  # Show last 3 actions
            if 'target' in action:
                print(f"   {i+1}. Waypoint {action['waypoint']}: {action['target']}")
            else:
                print(f"   {i+1}. {action.get('action', 'Unknown')}: {action}")
    
    # Communication demonstration
    print("\n🔗 Communication Layer Demonstration:")
    print("   Claude's control system uses JSON files for real-time communication:")
    
    # Create example command
    command_example = {
        "timestamp": 1755628600.0,
        "action": "move_to",
        "params": {
            "position": [0.5, 0.2, 0.4],
            "path_type": "arc"
        },
        "message": "Move end-effector using mathematical path planning"
    }
    
    with open("example_command.json", "w") as f:
        json.dump(command_example, f, indent=2)
    
    # Save session data
    claude.save_session("demo_session.json")
    
    print("   ✅ Created example_command.json - shows how Claude receives commands")
    print("   ✅ Created demo_session.json - shows complete session data")
    
    print(f"\n🎉 Demonstration Complete!")
    print("Key Points:")
    print("• Claude directly calculates inverse kinematics solutions")
    print("• Claude explicitly plans collision-free paths")  
    print("• Claude provides mathematical reasoning for each action")
    print("• JSON files enable real-time bidirectional communication")
    print("• No abstract AI interpretation needed - pure mathematical control")
    
    env.close()

if __name__ == "__main__":
    demonstrate_claude_direct_control()