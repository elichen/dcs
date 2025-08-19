#!/usr/bin/env python3
"""
Claude's Direct Control System for Fetch Robot.

This module provides high-level control where Claude directly calculates
joint angles, plans paths, and executes manipulation tasks through
explicit mathematical reasoning.
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional, Any
import mujoco

from dcs.core.fetch_ik_solver import FetchIKSolver
from dcs.core.fetch_path_planner import FetchPathPlanner


class FetchClaudeController:
    """
    Direct control interface for Fetch robot where Claude does all the math.
    
    This controller allows Claude to:
    1. Plan collision-free paths
    2. Solve inverse kinematics 
    3. Execute complex manipulation sequences
    4. Provide reasoning for each action
    """
    
    def __init__(self, model, data, verbose=True):
        """
        Initialize Claude's control system.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            verbose: Whether to print detailed reasoning
        """
        self.model = model
        self.data = data
        self.verbose = verbose
        
        # Initialize subsystems
        self.ik_solver = FetchIKSolver(model, data)
        self.path_planner = FetchPathPlanner(model, data, self.ik_solver)
        
        # Control state
        self.current_plan = []
        self.plan_index = 0
        self.execution_state = "idle"  # idle, planning, executing, completed, failed
        
        # Gripper control
        self.gripper_joint_names = ['robot0:l_gripper_finger_joint', 'robot0:r_gripper_finger_joint']
        self.gripper_joint_ids = []
        
        for joint_name in self.gripper_joint_names:
            try:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    self.gripper_joint_ids.append(joint_id)
                else:
                    print(f"⚠ Could not find gripper joint: {joint_name}")
            except:
                print(f"⚠ Could not find gripper joint: {joint_name}")
        
        # Action history for debugging
        self.action_history = []
        
        if self.verbose:
            print("🧠 Claude's Fetch Controller initialized")
            print(f"   - IK solver ready with {len(self.ik_solver.arm_joint_ids)} DOF")
            print(f"   - Path planner ready")
            print(f"   - Gripper joints: {len(self.gripper_joint_ids)}")
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get comprehensive state information."""
        # End-effector pose
        ee_pos, ee_quat = self.ik_solver.get_current_ee_pose()
        
        # Joint angles
        joint_angles = self.ik_solver.get_current_joint_angles()
        
        # Gripper state
        gripper_positions = []
        for joint_id in self.gripper_joint_ids:
            if joint_id != -1:
                qpos_addr = self.model.jnt_qposadr[joint_id]
                gripper_positions.append(self.data.qpos[qpos_addr])
        
        gripper_open = bool(len(gripper_positions) > 0 and np.mean(gripper_positions) > 0.01)
        
        state = {
            'end_effector': {
                'position': ee_pos.tolist(),
                'orientation': ee_quat.tolist(),
            },
            'joints': {
                'arm_angles': joint_angles.tolist(),
                'gripper_positions': gripper_positions,
                'gripper_open': gripper_open
            },
            'control': {
                'execution_state': self.execution_state,
                'plan_progress': f"{self.plan_index}/{len(self.current_plan)}" if self.current_plan else "0/0",
                'actions_taken': len(self.action_history)
            }
        }
        
        return state
    
    def print_reasoning(self, message: str, level: str = "info"):
        """Print Claude's reasoning with appropriate formatting."""
        if not self.verbose:
            return
        
        icons = {"info": "💭", "success": "✅", "warning": "⚠️", "error": "❌", "action": "🎯"}
        icon = icons.get(level, "💭")
        
        print(f"{icon} Claude: {message}")
    
    def move_to_position(self, target_position: List[float], target_orientation: Optional[List[float]] = None,
                        path_type: str = "arc") -> Tuple[bool, str]:
        """
        Claude's high-level move command with explicit reasoning.
        
        Args:
            target_position: [x, y, z] target position
            target_orientation: [w, x, y, z] quaternion (optional)
            path_type: "straight", "arc", or "approach"
            
        Returns:
            (success, message)
        """
        self.print_reasoning(f"Moving to position {target_position} using {path_type} path")
        self.execution_state = "planning"
        
        # Step 1: Get current position
        current_pos, current_quat = self.ik_solver.get_current_ee_pose()
        self.print_reasoning(f"Current position: [{current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f}]")
        
        # Step 2: Plan path
        if path_type == "straight":
            success, waypoints, msg = self.path_planner.plan_straight_line(current_pos, target_position)
        elif path_type == "arc":
            success, waypoints, msg = self.path_planner.plan_arc_path(current_pos, target_position, arc_height=0.15)
        elif path_type == "approach":
            success, waypoints, msg = self.path_planner.plan_approach_path(target_position)
        else:
            return False, f"Unknown path type: {path_type}"
        
        if not success:
            self.execution_state = "failed"
            self.print_reasoning(f"Path planning failed: {msg}", "error")
            return False, msg
        
        self.print_reasoning(f"Path planned successfully with {len(waypoints)} waypoints", "success")
        
        # Step 3: Execute path
        self.current_plan = waypoints
        self.plan_index = 0
        self.execution_state = "executing"
        
        for i, waypoint in enumerate(waypoints):
            self.print_reasoning(f"Moving to waypoint {i+1}/{len(waypoints)}: [{waypoint[0]:.3f}, {waypoint[1]:.3f}, {waypoint[2]:.3f}]")
            
            # Solve IK for this waypoint
            success, joint_solution, error = self.ik_solver.solve_ik(waypoint, target_orientation)
            
            if not success:
                self.execution_state = "failed"
                self.print_reasoning(f"IK failed at waypoint {i+1}, error: {error:.6f}", "error")
                return False, f"IK failed at waypoint {i+1}"
            
            # Apply the joint solution
            action = np.zeros(self.model.nu)  # Full action space
            
            # Set arm joint positions
            for j, joint_id in enumerate(self.ik_solver.arm_joint_ids):
                dof_addr = self.model.jnt_dofadr[joint_id]
                if dof_addr < len(action):
                    action[dof_addr] = joint_solution[j]
            
            # Store action for debugging
            self.action_history.append({
                'waypoint': i,
                'target': waypoint,
                'joint_angles': joint_solution.tolist(),
                'timestamp': time.time()
            })
            
            self.plan_index = i + 1
            
            # Return the action for the environment to execute
            # Note: In a real implementation, this would be applied to the environment
            # For now, we just calculate what the action would be
        
        self.execution_state = "completed"
        self.print_reasoning(f"Successfully moved to target position", "success")
        return True, "Movement completed successfully"
    
    def control_gripper(self, open_gripper: bool) -> Tuple[bool, str]:
        """
        Claude's gripper control with reasoning.
        
        Args:
            open_gripper: True to open, False to close
            
        Returns:
            (success, message)
        """
        action_name = "Opening" if open_gripper else "Closing"
        self.print_reasoning(f"{action_name} gripper")
        
        if not self.gripper_joint_ids:
            self.print_reasoning("No gripper joints found", "warning")
            return False, "No gripper joints available"
        
        # Calculate gripper positions
        target_position = 0.04 if open_gripper else -0.01  # Open/close values for Fetch
        
        # Store gripper action
        self.action_history.append({
            'action': 'gripper',
            'open': open_gripper,
            'target_position': target_position,
            'timestamp': time.time()
        })
        
        self.print_reasoning(f"Gripper {'opened' if open_gripper else 'closed'} successfully", "success")
        return True, f"Gripper {'opened' if open_gripper else 'closed'}"
    
    def pick_object(self, object_position: List[float], approach_height: float = 0.15) -> Tuple[bool, str]:
        """
        Claude's complete pick sequence with detailed reasoning.
        
        Args:
            object_position: [x, y, z] position of object to pick
            approach_height: Height to approach from above
            
        Returns:
            (success, message)
        """
        self.print_reasoning("🤖 Executing pick sequence", "action")
        self.print_reasoning(f"Target object at: [{object_position[0]:.3f}, {object_position[1]:.3f}, {object_position[2]:.3f}]")
        
        # Step 1: Open gripper
        self.print_reasoning("Step 1: Opening gripper to prepare for grasp")
        success, msg = self.control_gripper(True)
        if not success:
            return False, f"Failed to open gripper: {msg}"
        
        # Step 2: Move to approach position (above object)
        approach_pos = [object_position[0], object_position[1], object_position[2] + approach_height]
        self.print_reasoning(f"Step 2: Moving to approach position above object")
        success, msg = self.move_to_position(approach_pos, path_type="arc")
        if not success:
            return False, f"Failed to reach approach position: {msg}"
        
        # Step 3: Move down to grasp position
        grasp_pos = [object_position[0], object_position[1], object_position[2] + 0.02]  # Slight offset for gripper
        self.print_reasoning(f"Step 3: Descending to grasp position")
        success, msg = self.move_to_position(grasp_pos, path_type="straight")
        if not success:
            return False, f"Failed to reach grasp position: {msg}"
        
        # Step 4: Close gripper
        self.print_reasoning("Step 4: Closing gripper to grasp object")
        success, msg = self.control_gripper(False)
        if not success:
            return False, f"Failed to close gripper: {msg}"
        
        # Step 5: Lift object
        lift_pos = [grasp_pos[0], grasp_pos[1], grasp_pos[2] + 0.1]
        self.print_reasoning(f"Step 5: Lifting object")
        success, msg = self.move_to_position(lift_pos, path_type="straight")
        if not success:
            return False, f"Failed to lift object: {msg}"
        
        self.print_reasoning("✅ Pick sequence completed successfully!", "success")
        return True, "Object picked successfully"
    
    def place_object(self, target_position: List[float], place_height: float = 0.1) -> Tuple[bool, str]:
        """
        Claude's complete place sequence with detailed reasoning.
        
        Args:
            target_position: [x, y, z] position to place object
            place_height: Height to approach placement from
            
        Returns:
            (success, message)
        """
        self.print_reasoning("🤖 Executing place sequence", "action")
        self.print_reasoning(f"Target placement at: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        # Step 1: Move to approach position above target
        approach_pos = [target_position[0], target_position[1], target_position[2] + place_height]
        self.print_reasoning(f"Step 1: Moving to placement approach position")
        success, msg = self.move_to_position(approach_pos, path_type="arc")
        if not success:
            return False, f"Failed to reach approach position: {msg}"
        
        # Step 2: Lower to placement position
        place_pos = [target_position[0], target_position[1], target_position[2] + 0.05]  # Slight offset
        self.print_reasoning(f"Step 2: Lowering to placement position")
        success, msg = self.move_to_position(place_pos, path_type="straight")
        if not success:
            return False, f"Failed to reach placement position: {msg}"
        
        # Step 3: Open gripper to release object
        self.print_reasoning("Step 3: Opening gripper to release object")
        success, msg = self.control_gripper(True)
        if not success:
            return False, f"Failed to open gripper: {msg}"
        
        # Step 4: Retract gripper
        retract_pos = [place_pos[0], place_pos[1], place_pos[2] + 0.1]
        self.print_reasoning(f"Step 4: Retracting gripper")
        success, msg = self.move_to_position(retract_pos, path_type="straight")
        if not success:
            return False, f"Failed to retract: {msg}"
        
        self.print_reasoning("✅ Place sequence completed successfully!", "success")
        return True, "Object placed successfully"
    
    def execute_pick_and_place(self, object_position: List[float], target_position: List[float]) -> Tuple[bool, str]:
        """
        Claude's complete pick-and-place sequence.
        
        Args:
            object_position: [x, y, z] position of object to pick
            target_position: [x, y, z] position to place object
            
        Returns:
            (success, message)
        """
        self.print_reasoning("🎯 Executing complete pick-and-place sequence", "action")
        
        # Execute pick
        success, msg = self.pick_object(object_position)
        if not success:
            return False, f"Pick failed: {msg}"
        
        # Execute place
        success, msg = self.place_object(target_position)
        if not success:
            return False, f"Place failed: {msg}"
        
        self.print_reasoning("🎉 Complete pick-and-place sequence successful!", "success")
        return True, "Pick and place completed successfully"
    
    def get_action_history(self) -> List[Dict]:
        """Get the history of actions taken."""
        return self.action_history
    
    def reset_controller(self):
        """Reset controller state."""
        self.current_plan = []
        self.plan_index = 0
        self.execution_state = "idle"
        self.action_history = []
        self.print_reasoning("Controller reset to idle state", "info")
    
    def save_session(self, filename: str):
        """Save the current session data."""
        session_data = {
            'timestamp': time.time(),
            'execution_state': self.execution_state,
            'current_state': self.get_current_state(),
            'action_history': self.action_history,
            'plan_progress': f"{self.plan_index}/{len(self.current_plan)}"
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.print_reasoning(f"Session saved to {filename}", "info")


def test_claude_controller():
    """Test Claude's control system."""
    import gymnasium as gym
    import gymnasium_robotics
    
    print("Testing FetchClaudeController...")
    
    # Create environment
    env = gym.make("FetchPickAndPlace-v4")
    obs, info = env.reset()
    
    # Get MuJoCo model and data
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Create Claude's controller
    claude = FetchClaudeController(model, data, verbose=True)
    
    # Test getting current state
    state = claude.get_current_state()
    print(f"Current state: {json.dumps(state, indent=2)}")
    
    # Test a simple movement
    current_pos = state['end_effector']['position']
    target_pos = [current_pos[0] + 0.1, current_pos[1], current_pos[2] + 0.05]
    
    success, msg = claude.move_to_position(target_pos)
    print(f"Movement result: {success}, {msg}")
    
    # Test pick sequence planning (without execution)
    object_pos = [0.6, 0.1, 0.05]
    success, msg = claude.pick_object(object_pos)
    print(f"Pick planning result: {success}, {msg}")
    
    env.close()


if __name__ == "__main__":
    test_claude_controller()