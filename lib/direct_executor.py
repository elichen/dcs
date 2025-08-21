#!/usr/bin/env python3
"""
Direct Executor - In-process robot control with zero IPC overhead.

This module provides direct control capabilities that execute entirely within
the calling process, eliminating all IPC overhead for maximum performance.
"""

import time
import numpy as np
import mujoco
from typing import Tuple, List, Optional
from .fetch_protocol import StateHelper


class DirectExecutor:
    """Executes robot control commands directly in-process with zero IPC overhead."""
    
    def __init__(self, env, model, data, env_lock=None, cv_window_name=None, show_ui=True, gif_capture_callback=None):
        """
        Initialize direct executor.
        
        Args:
            env: MuJoCo environment
            model: MuJoCo model
            data: MuJoCo data
            env_lock: Threading lock for environment synchronization
            cv_window_name: OpenCV window name for direct rendering
            show_ui: Whether to show UI
            gif_capture_callback: Optional callback to capture frames for GIF recording
        """
        self.env = env
        self.model = model
        self.data = data
        self.env_lock = env_lock
        self.cv_window_name = cv_window_name
        self.show_ui = show_ui
        self.gif_capture_callback = gif_capture_callback
        
        # Control parameters
        self.k_p = 10.0  # Proportional gain
        self.max_velocity = 1.0
        self.tolerance = 0.005
        self.current_grip_action = 0.0
    
    def _get_gripper_position(self) -> np.ndarray:
        """Get current gripper position."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        return self.data.site_xpos[site_id].copy()
    
    def _get_object_position(self) -> np.ndarray:
        """Get current object position."""
        obs = self.env.unwrapped._get_obs()  # Get latest observation
        if isinstance(obs, dict) and 'achieved_goal' in obs:
            return obs['achieved_goal'][:3]
        return np.array([1.3, 0.7, 0.425])
    
    def _get_target_position(self) -> np.ndarray:
        """Get target position."""
        obs = self.env.unwrapped._get_obs()
        if isinstance(obs, dict) and 'desired_goal' in obs:
            return obs['desired_goal'][:3]
        return np.array([1.3, 0.9, 0.425])
    
    def _get_gripper_state(self) -> bool:
        """Get gripper open/closed state."""
        try:
            joint_names = ['robot0:l_gripper_finger_joint', 'robot0:r_gripper_finger_joint']
            positions = []
            
            for joint_name in joint_names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id != -1:
                    qpos_addr = self.model.jnt_qposadr[joint_id]
                    positions.append(self.data.qpos[qpos_addr])
            
            if positions:
                avg_position = np.mean(positions)
                return bool(avg_position > 0.01)  # True if open
        except:
            pass
        
        return True  # Default to open
    
    def _render_if_needed(self):
        """Render frame immediately for real-time animation and GIF capture."""
        try:
            import cv2
            # Render frame directly
            frame = self.env.render()
            if frame is not None:
                # Capture frame for GIF if callback is provided
                if self.gif_capture_callback:
                    self.gif_capture_callback(frame.copy())
                
                # Display frame if UI is enabled
                if self.show_ui and self.cv_window_name:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow(self.cv_window_name, frame_bgr)
                    cv2.waitKey(1)  # Non-blocking window update
        except Exception as e:
            # Silently fail to avoid disrupting robot control
            pass
    
    def move_to_position(self, target_pos: np.ndarray, max_steps: int = 50,
                        maintain_grip: bool = False, step_delay: float = 0.02,
                        velocity_scale: float = 1.0) -> Tuple[bool, str]:
        """
        Move gripper to target position using direct control.
        
        Args:
            target_pos: Target position [x, y, z]
            max_steps: Maximum number of control steps
            maintain_grip: Whether to maintain gripper closed
            step_delay: Delay between control steps
            velocity_scale: Scale factor for maximum velocity (0.1-2.0, default 1.0)
            
        Returns:
            (success, message)
        """
        target_pos = np.array(target_pos)
        
        # Use lock for thread-safe execution
        if self.env_lock:
            with self.env_lock:
                for step in range(max_steps):
                    current_pos = self._get_gripper_position()
                    error = target_pos - current_pos
                    distance = np.linalg.norm(error)
                    
                    if distance < self.tolerance:
                        return bool(True), f"Reached target (error: {distance:.4f}m)"
                    
                    # Proportional controller with velocity scaling
                    scaled_max_velocity = self.max_velocity * velocity_scale
                    velocity = np.clip(error * self.k_p, -scaled_max_velocity, scaled_max_velocity)
                    
                    # Set grip action
                    grip_action = 0.0  # Default: maintain current state
                    if maintain_grip:
                        grip_action = -1.0  # Keep gripper closed
                    
                    action = np.array([velocity[0], velocity[1], velocity[2], grip_action], dtype=np.float32)
                    
                    # Execute step directly - no IPC!
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self._render_if_needed()
                    
                    if step_delay > 0:
                        time.sleep(step_delay)
        else:
            # Fallback for backwards compatibility
            for step in range(max_steps):
                current_pos = self._get_gripper_position()
                error = target_pos - current_pos
                distance = np.linalg.norm(error)
                
                if distance < self.tolerance:
                    return True, f"Reached target (error: {distance:.4f}m)"
                
                # Proportional controller with velocity scaling
                scaled_max_velocity = self.max_velocity * velocity_scale
                velocity = np.clip(error * self.k_p, -scaled_max_velocity, scaled_max_velocity)
                
                # Set grip action
                grip_action = 0.0  # Default: maintain current state
                if maintain_grip:
                    grip_action = -1.0  # Keep gripper closed
                
                action = np.array([velocity[0], velocity[1], velocity[2], grip_action], dtype=np.float32)
                
                # Execute step directly - no IPC!
                obs, reward, terminated, truncated, info = self.env.step(action)
                self._render_if_needed()
                
                if step_delay > 0:
                    time.sleep(step_delay)
        
        # Check final distance
        final_distance = np.linalg.norm(target_pos - self._get_gripper_position())
        return bool(final_distance < 0.01), f"Stopped at distance: {final_distance:.4f}m"
    
    
    def control_gripper(self, open_gripper: bool, steps: int = 25, 
                       settling_time: float = 0.5) -> Tuple[bool, str]:
        """
        Control gripper open/close directly.
        
        Args:
            open_gripper: True to open, False to close
            steps: Number of steps to execute
            settling_time: Time to wait after action
            
        Returns:
            (success, message)
        """
        grip_action = 1.0 if open_gripper else -1.0
        
        # Use lock to ensure thread-safe env.step() calls
        if self.env_lock:
            with self.env_lock:
                for _ in range(steps):
                    action = np.array([0.0, 0.0, 0.0, grip_action], dtype=np.float32)
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    self._render_if_needed()
                    time.sleep(0.02)
        else:
            # Fallback for when no lock is provided (backwards compatibility)
            for _ in range(steps):
                action = np.array([0.0, 0.0, 0.0, grip_action], dtype=np.float32)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self._render_if_needed()
                time.sleep(0.02)
        
        # Settling time
        time.sleep(settling_time)
        
        # Update current grip state
        self.current_grip_action = grip_action
        
        return True, f"Gripper {'opened' if open_gripper else 'closed'}"
    
    def lift_by_height(self, height: float) -> Tuple[bool, str]:
        """
        Lift gripper by specified height while maintaining grip.
        
        Args:
            height: Height to lift in meters
            
        Returns:
            (success, message)
        """
        current_pos = self._get_gripper_position()
        target_pos = current_pos + np.array([0, 0, height])
        return self.move_to_position(target_pos, maintain_grip=True)
    
    def set_joint_angle(self, joint_name: str, angle: float) -> Tuple[bool, str]:
        """
        Directly set a joint angle.
        
        Args:
            joint_name: Name of the joint
            angle: Target angle in radians
            
        Returns:
            (success, message)
        """
        try:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            
            if joint_id == -1:
                return False, f"Joint '{joint_name}' not found"
            
            # Get current angle
            qpos_addr = self.model.jnt_qposadr[joint_id]
            original_angle = self.data.qpos[qpos_addr]
            
            # Set new angle with reasonable bounds
            if 'wrist_flex' in joint_name:
                clamped_angle = np.clip(angle, -2.16, 2.16)
            else:
                clamped_angle = np.clip(angle, -np.pi, np.pi)
            
            self.data.qpos[qpos_addr] = clamped_angle
            
            # Update physics
            mujoco.mj_forward(self.model, self.data)
            
            return True, f"Set {joint_name} to {clamped_angle:.3f} rad (was {original_angle:.3f} rad)"
            
        except Exception as e:
            return False, f"Failed to set joint angle: {str(e)}"
    
    def execute_trajectory(self, waypoints: List[np.ndarray], maintain_grip: bool = False,
                          step_delay: float = 0.02, max_steps_per_waypoint: int = 15) -> Tuple[bool, str]:
        """
        Execute a complete trajectory with direct control.
        
        Args:
            waypoints: List of position waypoints
            maintain_grip: Whether to maintain gripper closed
            step_delay: Delay between steps
            max_steps_per_waypoint: Max steps to reach each waypoint
            
        Returns:
            (success, message)
        """
        for i, waypoint in enumerate(waypoints):
            success, message = self.move_to_position(
                waypoint, 
                max_steps=max_steps_per_waypoint,
                maintain_grip=maintain_grip,
                step_delay=step_delay
            )
            if not success:
                return False, f"Failed at waypoint {i}: {message}"
        
        return True, f"Successfully executed trajectory with {len(waypoints)} waypoints"
    
    def wave_motion(self, cycles: int = 3, speed: float = 1.0, radius: float = 0.04) -> Tuple[bool, str]:
        """
        Execute wave motion with direct control.
        
        Args:
            cycles: Number of wave cycles
            speed: Wave speed multiplier
            radius: Wave radius in meters
            
        Returns:
            (success, message)
        """
        try:
            # Remember initial position
            initial_position = self._get_gripper_position().copy()
            
            # Set wrist orientation to point gripper up
            success, message = self.set_joint_angle("robot0:wrist_flex_joint", -1.5)
            if not success:
                return False, f"Failed to set gripper orientation: {message}"
            
            # Calculate wave center position (elevated for clock motion)
            center_pos = initial_position.copy()
            center_pos[2] = max(initial_position[2], 0.65)
            
            # Move to wave starting position
            success, message = self.move_to_position(center_pos, max_steps=30)
            if not success:
                return False, f"Failed to reach wave starting position: {message}"
            
            # Generate wave trajectory
            waypoints = self._generate_wave_trajectory(center_pos, cycles, radius)
            
            # Execute wave trajectory with direct control - no IPC!
            for i, waypoint in enumerate(waypoints):
                success, message = self.move_to_position(
                    waypoint, max_steps=15, step_delay=0.02
                )
                if not success:
                    return False, f"Wave failed at waypoint {i}: {message}"
                
                # Control wave speed
                time.sleep(0.1 / speed)
            
            # Restore original wrist orientation
            success, message = self.set_joint_angle("robot0:wrist_flex_joint", 0.0)
            if not success:
                return False, f"Failed to restore wrist orientation: {message}"
            
            # Return to initial position
            success, message = self.move_to_position(initial_position, max_steps=50)
            if not success:
                return False, f"Failed to return to initial position: {message}"
            
            return True, f"🕐 Completed {cycles} clock hand waves at speed {speed}x!"
            
        except Exception as e:
            return False, f"Wave motion failed: {str(e)}"
    
    def _generate_wave_trajectory(self, center_pos: np.ndarray, cycles: int, radius: float) -> List[np.ndarray]:
        """Generate wave trajectory waypoints."""
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
    
    def capture_image(self, filename: Optional[str] = None) -> Tuple[bool, str]:
        """
        Capture current environment state as an image.
        
        Args:
            filename: Optional filename. If not provided, auto-generates one.
            
        Returns:
            (success, path_or_message)
        """
        try:
            import cv2
            import os
            from pathlib import Path
            import time
            
            # Create captures directory
            captures_dir = Path.cwd() / "captures"
            captures_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if filename is None:
                timestamp = int(time.time())
                filename = f"capture_{timestamp}.png"
            
            # Ensure .png extension
            if not filename.endswith('.png'):
                filename += '.png'
            
            filepath = captures_dir / filename
            
            # Render current frame
            frame = self.env.render()
            if frame is None:
                return False, "Failed to render frame"
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Save image
            success = cv2.imwrite(str(filepath), frame_bgr)
            
            if success:
                return True, str(filepath)
            else:
                return False, "Failed to save image"
                
        except Exception as e:
            return False, f"Image capture failed: {str(e)}"
    
    def get_state(self) -> dict:
        """Get current robot and environment state."""
        return {
            'gripper_position': self._get_gripper_position().tolist(),
            'gripper_open': bool(self._get_gripper_state()),
            'object_position': self._get_object_position().tolist(),
            'target_position': self._get_target_position().tolist(),
        }