#!/usr/bin/env python3
"""
Fetch Robot Control Algorithms - Shared control logic for CLI tools.

This module contains reusable control algorithms that can be used
by individual CLI tools instead of being centralized in FetchSession.
"""

import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any
from .fetch_protocol import (
    create_raw_step_command, create_get_state_command, create_set_joint_command,
    StateHelper, ProtocolHelper, clamp_action_vector
)


class ProportionalController:
    """Proportional controller for position control."""
    
    def __init__(self, k_p: float = 10.0, max_velocity: float = 1.0, tolerance: float = 0.005):
        """
        Initialize proportional controller.
        
        Args:
            k_p: Proportional gain
            max_velocity: Maximum velocity output
            tolerance: Position tolerance for considering target reached
        """
        self.k_p = k_p
        self.max_velocity = max_velocity
        self.tolerance = tolerance
    
    def compute_action(self, current_pos: np.ndarray, target_pos: np.ndarray, 
                      maintain_grip: bool = False) -> Tuple[np.ndarray, float, bool]:
        """
        Compute control action to reach target position.
        
        Args:
            current_pos: Current gripper position [x, y, z]
            target_pos: Target position [x, y, z]
            maintain_grip: Whether to maintain current grip state
            
        Returns:
            (action_vector, distance_error, reached_target)
        """
        error = target_pos - current_pos
        distance = np.linalg.norm(error)
        reached = distance < self.tolerance
        
        if reached:
            velocity = np.zeros(3)
        else:
            velocity = np.clip(error * self.k_p, -self.max_velocity, self.max_velocity)
        
        grip_action = 0.0  # Maintain grip by default
        if maintain_grip:
            grip_action = -1.0  # Keep gripper closed
        
        action = np.array([velocity[0], velocity[1], velocity[2], grip_action], dtype=np.float32)
        
        return action, distance, reached


class GripperController:
    """Controller for gripper open/close operations."""
    
    def __init__(self, steps: int = 25, settling_time: float = 0.5):
        """
        Initialize gripper controller.
        
        Args:
            steps: Number of steps to execute for gripper action
            settling_time: Time to wait after action for gripper to settle
        """
        self.steps = steps
        self.settling_time = settling_time
    
    def generate_action_sequence(self, open_gripper: bool) -> List[np.ndarray]:
        """
        Generate action sequence for gripper control.
        
        Args:
            open_gripper: True to open, False to close
            
        Returns:
            List of action vectors
        """
        grip_action = 1.0 if open_gripper else -1.0
        action = np.array([0.0, 0.0, 0.0, grip_action], dtype=np.float32)
        return [action] * self.steps


class TrajectoryPlanner:
    """Planner for generating smooth trajectories."""
    
    def __init__(self, num_waypoints: int = 20):
        """
        Initialize trajectory planner.
        
        Args:
            num_waypoints: Number of waypoints in generated trajectories
        """
        self.num_waypoints = num_waypoints
    
    def plan_linear_path(self, start_pos: np.ndarray, end_pos: np.ndarray) -> List[np.ndarray]:
        """
        Plan a linear path between two points.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            
        Returns:
            List of waypoints
        """
        waypoints = []
        for i in range(self.num_waypoints):
            t = i / (self.num_waypoints - 1)
            waypoint = start_pos + t * (end_pos - start_pos)
            waypoints.append(waypoint)
        return waypoints
    
    def plan_arc_path(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                     arc_height: float = 0.15) -> List[np.ndarray]:
        """
        Plan an arc path between two points.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            arc_height: Height of the arc above the straight line
            
        Returns:
            List of waypoints
        """
        waypoints = []
        for i in range(self.num_waypoints):
            t = i / (self.num_waypoints - 1)
            
            # Linear interpolation
            linear_point = start_pos + t * (end_pos - start_pos)
            
            # Add arc height (parabolic)
            arc_z_offset = 4 * arc_height * t * (1 - t)  # Parabola: peak at t=0.5
            
            waypoint = linear_point.copy()
            waypoint[2] += arc_z_offset
            waypoints.append(waypoint)
            
        return waypoints
    
    def plan_wave_trajectory(self, center_pos: np.ndarray, cycles: int = 3, 
                           radius: float = 0.04) -> List[np.ndarray]:
        """
        Plan a wave trajectory for the robot.
        
        Args:
            center_pos: Center position for the wave motion
            cycles: Number of wave cycles
            radius: Radius of the wave motion
            
        Returns:
            List of waypoints for the wave motion
        """
        waypoints = []
        
        # Wave between 11 o'clock and 1 o'clock positions
        angle_start = 11 * np.pi / 6  # 11 o'clock
        angle_end = np.pi / 6         # 1 o'clock
        
        points_per_swing = 10
        
        for cycle in range(cycles):
            # Forward sweep: 11 to 1 o'clock
            for i in range(points_per_swing):
                t = i / (points_per_swing - 1)
                # Handle wrap-around from 11 to 1 o'clock
                if t <= 0.5:
                    angle = angle_start + t * 2 * (2*np.pi + angle_end - angle_start)
                else:
                    angle = (angle_start + (t-0.5) * 2 * (2*np.pi + angle_end - angle_start)) % (2*np.pi)
                
                x = center_pos[0] + radius * np.cos(angle)
                y = center_pos[1] + radius * np.sin(angle)
                z = center_pos[2]
                
                waypoints.append(np.array([x, y, z]))
            
            # Return sweep: 1 to 11 o'clock
            for i in range(points_per_swing):
                t = i / (points_per_swing - 1)
                angle = angle_end + t * (angle_start - angle_end + 2*np.pi)
                angle = angle % (2*np.pi)
                
                x = center_pos[0] + radius * np.cos(angle)
                y = center_pos[1] + radius * np.sin(angle)
                z = center_pos[2]
                
                waypoints.append(np.array([x, y, z]))
        
        return waypoints


class MovementExecutor:
    """Executes movement commands using low-level protocol."""
    
    def __init__(self, session_manager, session_id: str):
        """
        Initialize movement executor.
        
        Args:
            session_manager: SessionManager instance
            session_id: Target session ID
        """
        self.session_manager = session_manager
        self.session_id = session_id
        self.controller = ProportionalController()
    
    def move_to_position(self, target_pos: np.ndarray, max_steps: int = 50, 
                        maintain_grip: bool = False, step_delay: float = 0.02) -> Tuple[bool, str]:
        """
        Move gripper to target position using proportional control.
        
        Args:
            target_pos: Target position [x, y, z]
            max_steps: Maximum number of control steps
            maintain_grip: Whether to maintain gripper closed
            step_delay: Delay between control steps
            
        Returns:
            (success, message)
        """
        for step in range(max_steps):
            # Get current state
            state = self.session_manager.get_session_state(self.session_id)
            if not state:
                return False, "Could not get session state"
            
            current_pos = StateHelper.extract_gripper_position(state)
            if current_pos is None:
                return False, "Could not extract gripper position"
            
            # Compute control action
            action, distance, reached = self.controller.compute_action(
                current_pos, target_pos, maintain_grip
            )
            
            if reached:
                return True, f"Reached target (error: {distance:.4f}m)"
            
            # Send action command
            command = create_raw_step_command(action.tolist(), steps=1)
            success, message = self.session_manager.send_command(self.session_id, command)
            
            if not success:
                return False, f"Step {step} failed: {message}"
            
            time.sleep(step_delay)
        
        # Check final distance
        state = self.session_manager.get_session_state(self.session_id)
        if state:
            current_pos = StateHelper.extract_gripper_position(state)
            if current_pos is not None:
                final_distance = np.linalg.norm(target_pos - current_pos)
                return final_distance < 0.01, f"Stopped at distance: {final_distance:.4f}m"
        
        return False, "Could not determine final position"
    
    def execute_trajectory(self, waypoints: List[np.ndarray], maintain_grip: bool = False,
                          step_delay: float = 0.02) -> Tuple[bool, str]:
        """
        Execute a trajectory defined by waypoints.
        
        Args:
            waypoints: List of position waypoints
            maintain_grip: Whether to maintain gripper closed
            step_delay: Delay between waypoints
            
        Returns:
            (success, message)
        """
        for i, waypoint in enumerate(waypoints):
            success, message = self.move_to_position(
                waypoint, max_steps=15, maintain_grip=maintain_grip, 
                step_delay=step_delay
            )
            if not success:
                return False, f"Failed at waypoint {i}: {message}"
        
        return True, f"Successfully executed trajectory with {len(waypoints)} waypoints"
    
    def control_gripper(self, open_gripper: bool) -> Tuple[bool, str]:
        """
        Control gripper open/close.
        
        Args:
            open_gripper: True to open, False to close
            
        Returns:
            (success, message)
        """
        gripper_controller = GripperController()
        action_sequence = gripper_controller.generate_action_sequence(open_gripper)
        
        for action in action_sequence:
            command = create_raw_step_command(action.tolist(), steps=1)
            success, message = self.session_manager.send_command(self.session_id, command)
            if not success:
                return False, f"Gripper control failed: {message}"
            time.sleep(0.02)
        
        # Settling time
        time.sleep(gripper_controller.settling_time)
        
        return True, f"Gripper {'opened' if open_gripper else 'closed'}"
    
    def set_joint_angle(self, joint_name: str, angle: float) -> Tuple[bool, str]:
        """
        Set a specific joint angle.
        
        Args:
            joint_name: Name of the joint
            angle: Target angle in radians
            
        Returns:
            (success, message)
        """
        command = create_set_joint_command(joint_name, angle)
        return self.session_manager.send_command(self.session_id, command)


def lift_by_height(executor: MovementExecutor, height: float) -> Tuple[bool, str]:
    """
    Lift gripper by specified height while maintaining grip.
    
    Args:
        executor: MovementExecutor instance
        height: Height to lift in meters
        
    Returns:
        (success, message)
    """
    # Get current position
    state = executor.session_manager.get_session_state(executor.session_id)
    if not state:
        return False, "Could not get session state"
    
    current_pos = StateHelper.extract_gripper_position(state)
    if current_pos is None:
        return False, "Could not extract gripper position"
    
    # Calculate target position
    target_pos = current_pos + np.array([0, 0, height])
    
    # Execute movement with grip maintained
    return executor.move_to_position(target_pos, maintain_grip=True)