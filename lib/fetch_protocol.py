#!/usr/bin/env python3
"""
Fetch Robot Protocol - Low-level command types and state management.

This module defines the protocol for communicating with FetchSession
using raw, low-level commands instead of high-level actions.
"""

from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import numpy as np


class CommandType(Enum):
    """Low-level command types for FetchSession."""
    RAW_STEP = "raw_step"              # Execute single environment step
    BATCH_STEPS = "batch_steps"        # Execute multiple steps efficiently  
    GET_STATE = "get_state"            # Return full environment state
    SET_JOINT = "set_joint"            # Direct joint manipulation
    RESET = "reset"                    # Reset environment
    DIRECT_API_CALL = "direct_api_call"  # Execute Direct API method
    EMBEDDED_SCRIPT = "embedded_script"  # Execute Python code within session


class StateQuery(Enum):
    """Types of state information that can be queried."""
    FULL = "full"                      # Complete robot and environment state
    GRIPPER_POS = "gripper_position"   # Just gripper position
    GRIPPER_STATE = "gripper_state"    # Just gripper open/closed
    OBJECT_POS = "object_position"     # Just object position
    TARGET_POS = "target_position"     # Just target position
    JOINT_ANGLES = "joint_angles"      # All joint angles


def create_raw_step_command(action_vector: List[float], steps: int = 1) -> Dict[str, Any]:
    """
    Create a command to execute raw environment steps.
    
    Args:
        action_vector: [x_vel, y_vel, z_vel, grip_action] 
                      velocities in [-1, 1], grip_action: 1=open, -1=close, 0=maintain
        steps: Number of steps to execute with this action
    
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.RAW_STEP.value,
        "action_vector": action_vector,
        "steps": steps
    }


def create_batch_steps_command(action_sequence: List[List[float]], 
                              step_delay: float = 0.02) -> Dict[str, Any]:
    """
    Create a command to execute a sequence of actions efficiently.
    
    Args:
        action_sequence: List of action vectors to execute in sequence
        step_delay: Delay between steps in seconds
        
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.BATCH_STEPS.value,
        "action_sequence": action_sequence,
        "step_delay": step_delay
    }


def create_get_state_command(query_type: StateQuery = StateQuery.FULL) -> Dict[str, Any]:
    """
    Create a command to query environment state.
    
    Args:
        query_type: Type of state information to retrieve
        
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.GET_STATE.value,
        "query_type": query_type.value
    }


def create_set_joint_command(joint_name: str, angle: float) -> Dict[str, Any]:
    """
    Create a command to directly set a joint angle.
    
    Args:
        joint_name: Name of the joint (e.g., "robot0:wrist_flex_joint")
        angle: Target angle in radians
        
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.SET_JOINT.value,
        "joint_name": joint_name,
        "angle": angle
    }


def create_reset_command() -> Dict[str, Any]:
    """
    Create a command to reset the environment.
    
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.RESET.value
    }


def create_direct_api_command(method: str, *args, **kwargs) -> Dict[str, Any]:
    """
    Create a command to execute a Direct API method within the session process.
    
    Args:
        method: Name of the DirectExecutor method to call
        *args: Positional arguments for the method
        **kwargs: Keyword arguments for the method
    
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.DIRECT_API_CALL.value,
        "method": method,
        "args": list(args),
        "kwargs": kwargs
    }


def create_embedded_script_command(script_code: str, script_globals: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create a command to execute Python code within the session process.
    
    Args:
        script_code: Python code to execute
        script_globals: Global variables to make available to the script
    
    Returns:
        Command dictionary for SessionManager.send_command()
    """
    return {
        "action": CommandType.EMBEDDED_SCRIPT.value,
        "script_code": script_code,
        "script_globals": script_globals or {}
    }


class StateHelper:
    """Helper class for working with state data."""
    
    @staticmethod
    def extract_gripper_position(state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract gripper position from state dict."""
        try:
            return np.array(state.get('robot', {}).get('gripper_position', []))
        except (KeyError, TypeError):
            return None
    
    @staticmethod
    def extract_object_position(state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract object position from state dict."""
        try:
            return np.array(state.get('environment', {}).get('object_position', []))
        except (KeyError, TypeError):
            return None
    
    @staticmethod
    def extract_target_position(state: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract target position from state dict."""
        try:
            return np.array(state.get('environment', {}).get('target_position', []))
        except (KeyError, TypeError):
            return None
    
    @staticmethod
    def is_gripper_open(state: Dict[str, Any]) -> Optional[bool]:
        """Check if gripper is open from state dict."""
        try:
            return state.get('robot', {}).get('gripper_open', None)
        except (KeyError, TypeError):
            return None


class ProtocolHelper:
    """Helper class for common protocol operations."""
    
    @staticmethod
    def send_and_get_state(session_manager, session_id: str, 
                          command: Dict[str, Any]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Send a command and immediately get the updated state.
        
        Args:
            session_manager: SessionManager instance
            session_id: Target session ID
            command: Command to send
            
        Returns:
            (success, message, updated_state)
        """
        success, message = session_manager.send_command(session_id, command)
        if success:
            state = session_manager.get_session_state(session_id)
            return success, message, state
        return success, message, None
    
    @staticmethod
    def wait_for_position(session_manager, session_id: str, 
                         target_pos: np.ndarray, tolerance: float = 0.005,
                         max_attempts: int = 50) -> Tuple[bool, str, float]:
        """
        Wait for gripper to reach a target position.
        
        Args:
            session_manager: SessionManager instance
            session_id: Target session ID
            target_pos: Target position to reach
            tolerance: Distance tolerance for success
            max_attempts: Maximum attempts to check
            
        Returns:
            (reached, message, final_distance)
        """
        for attempt in range(max_attempts):
            state = session_manager.get_session_state(session_id)
            if state is None:
                return False, "Session state unavailable", float('inf')
            
            current_pos = StateHelper.extract_gripper_position(state)
            if current_pos is None:
                return False, "Could not extract gripper position", float('inf')
            
            distance = np.linalg.norm(target_pos - current_pos)
            if distance < tolerance:
                return True, f"Reached target (error: {distance:.4f}m)", distance
        
        return False, f"Failed to reach target within {max_attempts} attempts", distance


def validate_action_vector(action_vector: List[float]) -> bool:
    """
    Validate that an action vector is properly formatted.
    
    Args:
        action_vector: Action vector to validate
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(action_vector, (list, tuple, np.ndarray)):
        return False
    
    if len(action_vector) != 4:
        return False
    
    # Check that all values are numeric and within reasonable bounds
    try:
        x, y, z, grip = action_vector
        if not all(isinstance(v, (int, float)) for v in action_vector):
            return False
        # Velocities should be in [-1, 1], grip action in [-1, 1]
        if not (-1 <= x <= 1 and -1 <= y <= 1 and -1 <= z <= 1 and -1 <= grip <= 1):
            return False
        return True
    except (ValueError, TypeError):
        return False


def clamp_action_vector(action_vector: List[float]) -> np.ndarray:
    """
    Clamp action vector values to valid ranges.
    
    Args:
        action_vector: Action vector to clamp
        
    Returns:
        Clamped action vector as numpy array
    """
    action = np.array(action_vector[:4], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)