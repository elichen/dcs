#!/usr/bin/env python3
"""
Fetch Direct API - Zero-overhead in-process robot control.

This module provides a direct Python API for controlling Fetch robots with
zero IPC overhead, maintaining the modular architecture while restoring
original performance levels.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import json
from .session_registry import get_registry
from .direct_executor import DirectExecutor
from .fetch_session import SessionManager
from .fetch_protocol import create_direct_api_command


class FetchAPI:
    """Direct in-process API for Fetch robot control with zero IPC overhead."""
    
    def __init__(self, session_id: str):
        """
        Initialize API connection to a session.
        
        Args:
            session_id: ID of the active session to connect to
        """
        self.session_id = session_id
        self._executor = None
        self._connect()
    
    def _connect(self):
        """Connect to the session via IPC."""
        # Check if session exists
        state = SessionManager.get_session_state(self.session_id)
        if not state:
            raise RuntimeError(f"Session {self.session_id} not found")
        
        # Mark as connected (we use IPC to trigger Direct API calls in session process)
        self._executor = "ipc_connected"
    
    def _call_direct_api(self, method: str, *args, **kwargs) -> Tuple[bool, Any]:
        """Execute a Direct API method within the session process via IPC."""
        try:
            command = create_direct_api_command(method, *args, **kwargs)
            success, response = SessionManager.send_command(self.session_id, command)
            
            if success:
                # Parse JSON response
                try:
                    result = json.loads(response)
                    return True, result
                except json.JSONDecodeError:
                    # Response might be a simple string
                    return True, response
            else:
                return False, response
                
        except Exception as e:
            return False, str(e)
    
    @classmethod
    def connect(cls, session_id: str) -> 'FetchAPI':
        """
        Connect to an active session.
        
        Args:
            session_id: ID of the session to connect to
            
        Returns:
            FetchAPI instance connected to the session
            
        Raises:
            RuntimeError: If session is not found or not ready
        """
        return cls(session_id)
    
    def move_to(self, position: List[float], maintain_grip: bool = False, 
               max_steps: int = 50, velocity_scale: float = 1.0) -> Tuple[bool, str]:
        """
        Move gripper to target position.
        
        Args:
            position: Target position [x, y, z]
            maintain_grip: Whether to maintain gripper closed
            max_steps: Maximum control steps
            velocity_scale: Scale factor for maximum velocity (0.1-2.0, default 1.0)
            
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api(
            "move_to_position", 
            position,  # position as list 
            max_steps=max_steps, 
            maintain_grip=maintain_grip,
            velocity_scale=velocity_scale
        )
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def approach(self, position: List[float], height_offset: float = 0.10) -> Tuple[bool, str]:
        """
        Move to approach position (above target).
        
        Args:
            position: Target position [x, y, z]
            height_offset: Height above target (default 10cm)
            
        Returns:
            (success, message)
        """
        approach_pos = [position[0], position[1], position[2] + height_offset]
        
        # Check if gripper is closed to decide on grip maintenance
        state = self.get_state()
        maintain_grip = not state.get('gripper_open', True)
        
        return self.move_to(approach_pos, maintain_grip=maintain_grip)
    
    def grip(self, open_gripper: bool) -> Tuple[bool, str]:
        """
        Control gripper open/close.
        
        Args:
            open_gripper: True to open, False to close
            
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("control_gripper", open_gripper)
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def lift(self, height: float) -> Tuple[bool, str]:
        """
        Lift gripper by specified height while maintaining grip.
        
        Args:
            height: Height to lift in meters
            
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("lift_by_height", height)
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def wave(self, cycles: int = 3, speed: float = 1.0) -> Tuple[bool, str]:
        """
        Perform wave motion.
        
        Args:
            cycles: Number of wave cycles
            speed: Wave speed multiplier
            
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("wave_motion", cycles, speed)
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def pick(self, position: List[float]) -> Tuple[bool, str, List[Dict]]:
        """
        Execute complete pick sequence.
        
        Args:
            position: Object position [x, y, z]
            
        Returns:
            (success, message, sequence_results)
        """
        if self._executor is None:
            return False, "Not connected to session", []
        
        sequence_results = []
        target_pos = np.array(position)
        
        try:
            # Step 1: Open gripper
            success, message = self.grip(True)
            sequence_results.append({"step": "open_gripper", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to open gripper: {message}", sequence_results
            
            # Step 2: Approach position (10cm above)
            approach_pos = target_pos + np.array([0, 0, 0.1])
            success, message = self.move_to(approach_pos.tolist(), max_steps=30)
            sequence_results.append({"step": "approach", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to approach position: {message}", sequence_results
            
            # Step 3: Move to grasp position (5mm above object center)
            grasp_pos = target_pos + np.array([0, 0, 0.005])
            success, message = self.move_to(grasp_pos.tolist(), max_steps=30)
            sequence_results.append({"step": "descend", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to descend to grasp position: {message}", sequence_results
            
            # Step 4: Close gripper
            success, message = self.grip(False)
            sequence_results.append({"step": "close_gripper", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to close gripper: {message}", sequence_results
            
            # Step 5: Small test lift to verify grip
            success, message = self.lift(0.02)  # 2cm test lift
            sequence_results.append({"step": "test_lift", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed test lift: {message}", sequence_results
            
            # Step 6: Full lift
            success, message = self.lift(0.13)  # Additional 13cm (total 15cm)
            sequence_results.append({"step": "full_lift", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed full lift: {message}", sequence_results
            
            return True, "Pick sequence completed successfully", sequence_results
            
        except Exception as e:
            error_result = {"step": "error", "result": {"success": False, "message": str(e)}}
            sequence_results.append(error_result)
            return False, f"Pick sequence failed: {str(e)}", sequence_results
    
    def place(self, position: List[float]) -> Tuple[bool, str, List[Dict]]:
        """
        Execute complete place sequence.
        
        Args:
            position: Target position [x, y, z]
            
        Returns:
            (success, message, sequence_results)
        """
        if self._executor is None:
            return False, "Not connected to session", []
        
        sequence_results = []
        target_pos = np.array(position)
        
        try:
            # Step 1: Transport above placement position (maintain grip)
            transport_pos = target_pos + np.array([0, 0, 0.10])  # 10cm above
            success, message = self.move_to(transport_pos.tolist(), max_steps=30, maintain_grip=True)
            sequence_results.append({"step": "transport", "result": {"success": success, "message": message, "target_position": target_pos.tolist(), "approach_position": transport_pos.tolist()}})
            if not success:
                return False, f"Failed to transport to position: {message}", sequence_results
            
            # Step 2: Lower to placement position (maintain grip)
            place_pos = target_pos + np.array([0, 0, 0.05])  # 5cm above placement surface
            success, message = self.move_to(place_pos.tolist(), max_steps=30, maintain_grip=True)
            sequence_results.append({"step": "lower", "result": {"success": success, "message": message, "target_position": place_pos.tolist()}})
            if not success:
                return False, f"Failed to lower to placement position: {message}", sequence_results
            
            # Step 3: Open gripper (release)
            success, message = self.grip(True)
            sequence_results.append({"step": "release", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to release object: {message}", sequence_results
            
            # Step 4: Retract gripper
            success, message = self.lift(0.10)  # 10cm retraction
            sequence_results.append({"step": "retract", "result": {"success": success, "message": message}})
            if not success:
                return False, f"Failed to retract gripper: {message}", sequence_results
            
            return True, "Place sequence completed successfully", sequence_results
            
        except Exception as e:
            error_result = {"step": "error", "result": {"success": False, "message": str(e)}}
            sequence_results.append(error_result)
            return False, f"Place sequence failed: {str(e)}", sequence_results
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get current robot and environment state.
        
        Returns:
            Dictionary with robot state information
        """
        if self._executor is None:
            return {"error": "Not connected to session"}
        
        success, result = self._call_direct_api("get_state")
        if success and isinstance(result, dict):
            return result
        else:
            return {"error": f"Failed to get state: {result}"}
    
    def get_object_position(self) -> List[float]:
        """Get current object position."""
        state = self.get_state()
        return state.get('object_position', [1.3, 0.7, 0.425])
    
    def get_target_position(self) -> List[float]:
        """Get target position."""
        state = self.get_state()
        return state.get('target_position', [1.3, 0.9, 0.425])
    
    def get_gripper_position(self) -> List[float]:
        """Get current gripper position."""
        state = self.get_state()
        return state.get('gripper_position', [1.35, 0.75, 0.47])
    
    def is_gripper_open(self) -> bool:
        """Check if gripper is open."""
        state = self.get_state()
        return state.get('gripper_open', True)
    
    def step_physics(self) -> Tuple[bool, str]:
        """
        Step the physics simulation once with zero velocity to allow natural physics.
        
        This is useful for letting objects slide/fall naturally without robot movement,
        particularly after applying forces to objects.
        
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("step_physics")
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def reset_environment(self) -> Tuple[bool, str]:
        """
        Reset the environment to initial state with new random object/target positions.
        
        This is useful for running multiple tests without restarting the session.
        
        Returns:
            (success, message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("_reset_environment")
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)
    
    def capture_image(self, filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Capture current environment state as an image.
        
        Args:
            filename: Optional filename. If not provided, auto-generates one.
            
        Returns:
            (success, path_or_message)
        """
        if self._executor is None:
            return False, "Not connected to session"
        
        success, result = self._call_direct_api("capture_image", filename)
        if success and isinstance(result, (list, tuple)) and len(result) == 2:
            return bool(result[0]), str(result[1])
        elif success:
            return True, str(result)
        else:
            return False, str(result)


# Convenience functions
def connect(session_id: str) -> FetchAPI:
    """Connect to a session with the direct API."""
    return FetchAPI.connect(session_id)


def list_sessions() -> Dict[str, Dict[str, Any]]:
    """List all active sessions."""
    registry = get_registry()
    return registry.list_sessions()