#!/usr/bin/env python3
"""
Interactive demonstration of Claude's direct control over Fetch robot.

This creates a real-time interface where Claude can demonstrate
path planning, IK solving, and manipulation tasks with visualization.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import json
import time
import os
from typing import Dict, Any, List

from dcs.core.fetch_claude_controller import FetchClaudeController


class FetchInteractiveDemo:
    """
    Interactive demonstration system for Claude's Fetch robot control.
    
    Provides real-time visualization and command interface for demonstrating
    Claude's direct control capabilities.
    """
    
    def __init__(self, render_mode="human", command_file="claude_commands.json", state_file="robot_state.json"):
        """
        Initialize the interactive demo.
        
        Args:
            render_mode: MuJoCo rendering mode
            command_file: File for receiving commands from Claude
            state_file: File for sending state to Claude
        """
        self.render_mode = render_mode
        self.command_file = command_file
        self.state_file = state_file
        
        # Initialize environment
        print("🤖 Initializing Fetch robot environment...")
        self.env = gym.make("FetchPickAndPlace-v4", render_mode=render_mode)
        self.obs, self.info = self.env.reset()
        
        # Initialize Claude's controller
        print("🧠 Initializing Claude's control system...")
        self.claude = FetchClaudeController(
            self.env.unwrapped.model,
            self.env.unwrapped.data,
            verbose=True
        )
        
        # Demo state
        self.running = True
        self.demo_step = 0
        self.last_command_time = 0
        
        # Create communication files
        self._initialize_communication_files()
        
        print("✅ Interactive demo ready!")
        print(f"   - Command file: {command_file}")
        print(f"   - State file: {state_file}")
        print(f"   - Render mode: {render_mode}")
    
    def _initialize_communication_files(self):
        """Initialize JSON communication files."""
        # Create empty command file
        initial_command = {
            "timestamp": time.time(),
            "action": "initialize",
            "params": {},
            "message": "Demo initialized, waiting for commands..."
        }
        
        with open(self.command_file, 'w') as f:
            json.dump(initial_command, f, indent=2)
        
        # Create initial state file
        self._update_state_file()
    
    def _update_state_file(self):
        """Update the state file with current robot information."""
        # Get current state from Claude's controller
        robot_state = self.claude.get_current_state()
        
        # Add environment information
        full_state = {
            "timestamp": time.time(),
            "demo_step": self.demo_step,
            "robot": robot_state,
            "environment": {
                "observation_shape": self.obs.shape if hasattr(self.obs, 'shape') else str(type(self.obs)),
                "info": self.info
            },
            "status": "ready" if self.claude.execution_state == "idle" else self.claude.execution_state
        }
        
        # Add object and target positions if available
        try:
            if hasattr(self.env.unwrapped, 'goal'):
                full_state["environment"]["goal_position"] = self.env.unwrapped.goal.tolist()
            
            # Try to get object position from observation
            if isinstance(self.obs, dict) and 'achieved_goal' in self.obs:
                full_state["environment"]["object_position"] = self.obs['achieved_goal'].tolist()
                full_state["environment"]["desired_goal"] = self.obs['desired_goal'].tolist()
        except:
            pass
        
        with open(self.state_file, 'w') as f:
            json.dump(full_state, f, indent=2)
    
    def _read_command(self) -> Dict[str, Any]:
        """Read command from the command file."""
        try:
            if not os.path.exists(self.command_file):
                return None
            
            with open(self.command_file, 'r') as f:
                command = json.load(f)
            
            # Check if command is new
            command_time = command.get('timestamp', 0)
            if command_time > self.last_command_time:
                self.last_command_time = command_time
                return command
        except (json.JSONDecodeError, FileNotFoundError):
            pass
        
        return None
    
    def _execute_command(self, command: Dict[str, Any]) -> tuple[bool, str]:
        """
        Execute a command from Claude.
        
        Args:
            command: Command dictionary with action and parameters
            
        Returns:
            (success, result_message)
        """
        action = command.get('action', '')
        params = command.get('params', {})
        
        print(f"🎯 Executing command: {action}")
        print(f"   Parameters: {params}")
        
        try:
            if action == "move_to":
                target_pos = params.get('position', [0.5, 0, 0.4])
                path_type = params.get('path_type', 'arc')
                success, msg = self.claude.move_to_position(target_pos, path_type=path_type)
                return success, msg
            
            elif action == "control_gripper":
                open_gripper = params.get('open', True)
                success, msg = self.claude.control_gripper(open_gripper)
                return success, msg
            
            elif action == "pick_object":
                object_pos = params.get('position', [0.6, 0, 0.05])
                success, msg = self.claude.pick_object(object_pos)
                return success, msg
            
            elif action == "place_object":
                target_pos = params.get('position', [0.4, 0.2, 0.05])
                success, msg = self.claude.place_object(target_pos)
                return success, msg
            
            elif action == "pick_and_place":
                object_pos = params.get('object_position', [0.6, 0, 0.05])
                target_pos = params.get('target_position', [0.4, 0.2, 0.05])
                success, msg = self.claude.execute_pick_and_place(object_pos, target_pos)
                return success, msg
            
            elif action == "reset":
                self.claude.reset_controller()
                self.obs, self.info = self.env.reset()
                return True, "Environment and controller reset"
            
            elif action == "get_state":
                # State is automatically updated, just confirm
                return True, "State information updated"
            
            elif action == "demonstrate":
                # Run a pre-programmed demonstration
                return self._run_demonstration()
            
            elif action == "stop":
                self.running = False
                return True, "Demo stopped"
            
            else:
                return False, f"Unknown action: {action}"
        
        except Exception as e:
            return False, f"Error executing {action}: {str(e)}"
    
    def _run_demonstration(self) -> tuple[bool, str]:
        """Run a pre-programmed demonstration sequence."""
        print("🎭 Running demonstration sequence...")
        
        # Get object and goal positions from environment
        try:
            if isinstance(self.obs, dict):
                object_pos = self.obs['achieved_goal'][:3].tolist()
                goal_pos = self.obs['desired_goal'][:3].tolist()
            else:
                # Default positions for demo
                object_pos = [0.6, 0.1, 0.05]
                goal_pos = [0.4, -0.1, 0.05]
            
            print(f"   Object at: {object_pos}")
            print(f"   Goal at: {goal_pos}")
            
            # Execute pick and place
            success, msg = self.claude.execute_pick_and_place(object_pos, goal_pos)
            return success, f"Demonstration completed: {msg}"
        
        except Exception as e:
            return False, f"Demonstration failed: {str(e)}"
    
    def run_interactive_loop(self):
        """Run the main interactive loop."""
        print("🔄 Starting interactive demo loop...")
        print("   - Modify claude_commands.json to send commands")
        print("   - Monitor robot_state.json for current state")
        print("   - Press Ctrl+C to stop")
        
        try:
            while self.running:
                # Update state file
                self._update_state_file()
                
                # Check for new commands
                command = self._read_command()
                if command:
                    success, result = self._execute_command(command)
                    
                    # Log result
                    if success:
                        print(f"✅ Command successful: {result}")
                    else:
                        print(f"❌ Command failed: {result}")
                
                # Render environment
                if self.render_mode == "human":
                    self.env.render()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                self.demo_step += 1
        
        except KeyboardInterrupt:
            print("\n🛑 Demo stopped by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up...")
        if hasattr(self, 'env'):
            self.env.close()
        
        # Save final session
        if hasattr(self, 'claude'):
            self.claude.save_session("demo_session.json")
    
    def create_example_commands(self):
        """Create example command files for Claude to use."""
        examples = {
            "move_example.json": {
                "timestamp": time.time(),
                "action": "move_to",
                "params": {
                    "position": [0.5, 0.2, 0.4],
                    "path_type": "arc"
                },
                "message": "Move end-effector to specified position using arc path"
            },
            
            "pick_example.json": {
                "timestamp": time.time(),
                "action": "pick_object",
                "params": {
                    "position": [0.6, 0.1, 0.05]
                },
                "message": "Pick up object at specified position"
            },
            
            "place_example.json": {
                "timestamp": time.time(),
                "action": "place_object", 
                "params": {
                    "position": [0.4, -0.1, 0.05]
                },
                "message": "Place object at specified position"
            },
            
            "demo_example.json": {
                "timestamp": time.time(),
                "action": "demonstrate",
                "params": {},
                "message": "Run complete pick-and-place demonstration"
            },
            
            "gripper_example.json": {
                "timestamp": time.time(),
                "action": "control_gripper",
                "params": {
                    "open": True
                },
                "message": "Open or close the gripper"
            }
        }
        
        for filename, example in examples.items():
            with open(filename, 'w') as f:
                json.dump(example, f, indent=2)
        
        print(f"📁 Created {len(examples)} example command files")


def main():
    """Main function to run the interactive demo."""
    print("=" * 60)
    print("🤖 Claude's Direct Fetch Robot Control Demo")
    print("=" * 60)
    
    # Create demo
    demo = FetchInteractiveDemo(
        render_mode="human",
        command_file="claude_commands.json",
        state_file="robot_state.json"
    )
    
    # Create example commands
    demo.create_example_commands()
    
    print("\n📖 How to use:")
    print("1. The demo creates claude_commands.json and robot_state.json")
    print("2. Claude writes commands to claude_commands.json")
    print("3. The demo executes commands and updates robot_state.json")
    print("4. Claude can read robot_state.json to see current status")
    print("5. Use the example command files as templates")
    
    # Run interactive loop
    demo.run_interactive_loop()


if __name__ == "__main__":
    main()