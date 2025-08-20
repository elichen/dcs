#!/usr/bin/env python3
"""
Fetch Robot Session Management for Unix-style CLI tools.

This module provides shared session management allowing multiple CLI tools
to interact with a single persistent robot environment.
"""

import os
import json
import time
import uuid
import signal
import threading
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import mujoco
import cv2
from .fetch_protocol import CommandType, StateQuery, validate_action_vector, clamp_action_vector

class FetchSession:
    """Manages a persistent Fetch robot session with shared state."""
    
    def __init__(self, session_id: str, show_ui: bool = True):
        self.session_id = session_id
        self.show_ui = show_ui
        self.session_dir = Path(tempfile.gettempdir()) / "fetch-sessions"
        self.session_dir.mkdir(exist_ok=True)
        
        # Session files
        self.state_file = self.session_dir / f"{session_id}.json"
        self.lock_file = self.session_dir / f"{session_id}.lock"
        self.pid_file = self.session_dir / f"{session_id}.pid"
        
        # Environment
        self.env = None
        self.obs = None
        self.info = None
        
        # State
        self.running = True
        self.last_action_time = time.time()
        self.current_grip_action = 0.0  # Track current desired grip state
        
        # UI setup for clean display
        self.cv_window_name = f'Fetch Robot - Session {session_id}'
        
        # Initialize environment
        self._init_environment()
        
        # Write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._cleanup_handler)
        signal.signal(signal.SIGINT, self._cleanup_handler)
        
        print(f"✅ Session {session_id} started")
        print(f"📍 State file: {self.state_file}")
        if show_ui:
            print("🖥️  Clean UI window opened (no overlay)")
    
    def _init_environment(self):
        """Initialize the Gymnasium environment."""
        # Always use rgb_array mode to avoid overlay, then display with OpenCV if needed
        self.env = gym.make("FetchPickAndPlace-v4", render_mode="rgb_array", max_episode_steps=1000)
        self.obs, self.info = self.env.reset()
        
        # Get MuJoCo model and data
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        # Setup OpenCV window if UI is requested
        if self.show_ui:
            cv2.namedWindow(self.cv_window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.cv_window_name, 800, 600)
            self._render_frame()  # Show initial frame
        
        # Update state
        self._update_state()
    
    def _render_frame(self):
        """Render and display frame without overlay using OpenCV."""
        if not self.show_ui:
            return
            
        frame = self.env.render()
        if frame is not None:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow(self.cv_window_name, frame_bgr)
            cv2.waitKey(1)  # Non-blocking window update
    
    def _cleanup_handler(self, signum, frame):
        """Handle cleanup on signal."""
        print(f"\n🛑 Session {self.session_id} received signal {signum}, cleaning up...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up session resources."""
        self.running = False
        
        # Remove session files
        for file_path in [self.state_file, self.lock_file, self.pid_file]:
            try:
                if file_path.exists():
                    file_path.unlink()
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
        
        # Close OpenCV window
        if self.show_ui:
            cv2.destroyAllWindows()
        
        # Close environment
        if self.env:
            self.env.close()
        
        print(f"🧹 Session {self.session_id} cleaned up")
    
    def _get_gripper_position(self):
        """Get current gripper position."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        return self.data.site_xpos[site_id].copy()
    
    def _get_object_position(self):
        """Get current object position."""
        if isinstance(self.obs, dict) and 'achieved_goal' in self.obs:
            return self.obs['achieved_goal'][:3]
        return np.array([1.3, 0.7, 0.425])
    
    def _get_target_position(self):
        """Get target position."""
        if isinstance(self.obs, dict) and 'desired_goal' in self.obs:
            return self.obs['desired_goal'][:3]
        return np.array([1.3, 0.9, 0.425])
    
    def _get_gripper_state(self):
        """Get gripper open/closed state."""
        # Check gripper joint positions
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
                return avg_position > 0.01  # True if open
        except:
            pass
        
        return True  # Default to open
    
    def _get_joint_angles(self):
        """Get all joint angles."""
        joint_angles = {}
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if joint_name and joint_name.startswith('robot0:'):
                qpos_addr = self.model.jnt_qposadr[i]
                joint_angles[joint_name] = float(self.data.qpos[qpos_addr])
        return joint_angles

    def _update_state(self):
        """Update the shared state file."""
        state = {
            'session_id': self.session_id,
            'timestamp': time.time(),
            'robot': {
                'gripper_position': self._get_gripper_position().tolist(),
                'gripper_open': bool(self._get_gripper_state()),
                'joint_angles': self._get_joint_angles(),
            },
            'environment': {
                'object_position': self._get_object_position().tolist(),
                'target_position': self._get_target_position().tolist(),
            },
            'status': 'running' if self.running else 'stopped'
        }
        
        # Write state atomically
        temp_file = self.state_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self.state_file)
    
    def execute_action(self, action_type: str, **kwargs) -> Tuple[bool, str]:
        """Execute a low-level robot action using the new protocol."""
        self.last_action_time = time.time()
        
        try:
            if action_type == CommandType.RAW_STEP.value:
                return self._execute_raw_step(kwargs.get('action_vector'), kwargs.get('steps', 1))
            elif action_type == CommandType.BATCH_STEPS.value:
                return self._execute_batch_steps(kwargs.get('action_sequence'), kwargs.get('step_delay', 0.02))
            elif action_type == CommandType.GET_STATE.value:
                return self._get_detailed_state(kwargs.get('query_type', StateQuery.FULL.value))
            elif action_type == CommandType.SET_JOINT.value:
                return self._set_joint_angle(kwargs.get('joint_name'), kwargs.get('angle'))
            elif action_type == CommandType.RESET.value:
                return self._reset_environment()
            # Backward compatibility for legacy commands
            elif action_type in ['move', 'move_with_grip', 'gripper', 'lift', 'wave']:
                return False, f"Legacy action '{action_type}' no longer supported. Use low-level protocol commands."
            else:
                return False, f"Unknown action type: {action_type}"
        
        except Exception as e:
            return False, f"Action failed: {str(e)}"
        
        finally:
            # Always update state after action
            self._update_state()
    
    def _execute_raw_step(self, action_vector: List[float], steps: int = 1) -> Tuple[bool, str]:
        """Execute raw environment step(s) with given action vector."""
        if action_vector is None:
            return False, "No action vector provided"
        
        if not validate_action_vector(action_vector):
            return False, "Invalid action vector format"
        
        # Clamp action to safe ranges
        action = clamp_action_vector(action_vector)
        
        try:
            for step in range(steps):
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                self._render_frame()
                
                # Update grip state tracking
                if action[3] != 0:
                    self.current_grip_action = action[3]
                
                if steps > 1:  # Only add delay for multi-step actions
                    time.sleep(0.02)
            
            return True, f"Executed {steps} step(s) with action {action.tolist()}"
            
        except Exception as e:
            return False, f"Step execution failed: {str(e)}"
    
    def _execute_batch_steps(self, action_sequence: List[List[float]], step_delay: float = 0.02) -> Tuple[bool, str]:
        """Execute a batch of action steps efficiently."""
        if not action_sequence:
            return False, "No action sequence provided"
        
        try:
            for i, action_vector in enumerate(action_sequence):
                if not validate_action_vector(action_vector):
                    return False, f"Invalid action vector at index {i}: {action_vector}"
                
                action = clamp_action_vector(action_vector)
                self.obs, reward, terminated, truncated, self.info = self.env.step(action)
                self._render_frame()
                
                # Update grip state tracking
                if action[3] != 0:
                    self.current_grip_action = action[3]
                
                time.sleep(step_delay)
            
            return True, f"Executed {len(action_sequence)} action steps"
            
        except Exception as e:
            return False, f"Batch execution failed: {str(e)}"
    
    def _get_detailed_state(self, query_type: str = "full") -> Tuple[bool, str]:
        """Get detailed environment state based on query type."""
        try:
            if query_type == StateQuery.GRIPPER_POS.value:
                pos = self._get_gripper_position().tolist()
                return True, json.dumps({"gripper_position": pos})
            
            elif query_type == StateQuery.GRIPPER_STATE.value:
                open_state = self._get_gripper_state()
                return True, json.dumps({"gripper_open": bool(open_state)})
            
            elif query_type == StateQuery.OBJECT_POS.value:
                pos = self._get_object_position().tolist()
                return True, json.dumps({"object_position": pos})
            
            elif query_type == StateQuery.TARGET_POS.value:
                pos = self._get_target_position().tolist()
                return True, json.dumps({"target_position": pos})
            
            elif query_type == StateQuery.JOINT_ANGLES.value:
                angles = self._get_joint_angles()
                return True, json.dumps({"joint_angles": angles})
            
            else:  # StateQuery.FULL or default
                state = {
                    'gripper_position': self._get_gripper_position().tolist(),
                    'gripper_open': bool(self._get_gripper_state()),
                    'object_position': self._get_object_position().tolist(),
                    'target_position': self._get_target_position().tolist(),
                    'joint_angles': self._get_joint_angles(),
                    'timestamp': time.time()
                }
                return True, json.dumps(state)
                
        except Exception as e:
            return False, f"State query failed: {str(e)}"
    
    def _set_joint_angle(self, joint_name: str, angle: float) -> Tuple[bool, str]:
        """
        Directly set a joint angle.
        
        Args:
            joint_name: Name of the joint (e.g., 'robot0:wrist_flex_joint')
            angle: Target angle in radians
        
        Returns:
            (success, message)
        """
        if not joint_name:
            return False, "No joint name provided"
        
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
    
    # Note: Wave functionality moved to bin/wave tool using low-level protocol
    
    def _reset_environment(self) -> Tuple[bool, str]:
        """Reset the environment."""
        try:
            self.obs, self.info = self.env.reset()
            self.current_grip_action = 0.0  # Reset grip state
            return True, "Environment reset successfully"
        except Exception as e:
            return False, f"Environment reset failed: {str(e)}"
    
    def run_session(self):
        """Run the session main loop."""
        try:
            while self.running:
                # Check for commands
                self._process_commands()
                
                # Keep environment alive by stepping with current grip state maintained
                if time.time() - self.last_action_time > 1.0:
                    action = np.array([0.0, 0.0, 0.0, self.current_grip_action], dtype=np.float32)
                    self.obs, _, _, _, self.info = self.env.step(action)
                    self._render_frame()  # Keep display updated
                    self.last_action_time = time.time()
                
                # Update state periodically
                self._update_state()
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.cleanup()
    
    def _process_commands(self):
        """Process incoming commands from CLI tools."""
        command_file = self.session_dir / f"{self.session_id}.cmd"
        response_file = self.session_dir / f"{self.session_id}.response"
        
        if command_file.exists():
            try:
                with open(command_file) as f:
                    command = json.load(f)
                
                # Remove command file
                command_file.unlink()
                
                # Execute the command
                success, message = self.execute_action(
                    command.get('action', ''),
                    **{k: v for k, v in command.items() if k != 'action'}
                )
                
                # Write response
                response = {
                    'success': success,
                    'message': message,
                    'timestamp': time.time()
                }
                
                with open(response_file, 'w') as f:
                    json.dump(response, f)
                    
            except Exception as e:
                # Write error response
                response = {
                    'success': False,
                    'message': f"Command processing failed: {str(e)}",
                    'timestamp': time.time()
                }
                
                try:
                    with open(response_file, 'w') as f:
                        json.dump(response, f)
                except:
                    pass


class SessionManager:
    """Manages multiple Fetch robot sessions."""
    
    @staticmethod
    def get_session_dir():
        """Get the session directory."""
        return Path(tempfile.gettempdir()) / "fetch-sessions"
    
    @staticmethod
    def create_session(show_ui=True) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())[:8]
        return session_id
    
    @staticmethod
    def list_sessions():
        """List all active sessions."""
        session_dir = SessionManager.get_session_dir()
        if not session_dir.exists():
            return []
        
        sessions = []
        for pid_file in session_dir.glob("*.pid"):
            session_id = pid_file.stem
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                
                # Check if process is still running
                try:
                    os.kill(pid, 0)  # Signal 0 just checks if process exists
                    sessions.append({
                        'id': session_id,
                        'pid': pid,
                        'status': 'running'
                    })
                except ProcessLookupError:
                    # Process is dead, clean up
                    SessionManager.cleanup_session(session_id)
                    
            except Exception:
                pass
        
        return sessions
    
    @staticmethod
    def get_session_state(session_id: str) -> Optional[Dict[str, Any]]:
        """Get the current state of a session."""
        session_dir = SessionManager.get_session_dir()
        state_file = session_dir / f"{session_id}.json"
        
        if not state_file.exists():
            return None
        
        try:
            with open(state_file) as f:
                return json.load(f)
        except Exception:
            return None
    
    @staticmethod
    def send_command(session_id: str, command: Dict[str, Any]) -> Tuple[bool, str]:
        """Send a command to a running session."""
        # For now, we'll use a simple file-based approach
        # In a production system, you'd use sockets or shared memory
        
        session_dir = SessionManager.get_session_dir()
        command_file = session_dir / f"{session_id}.cmd"
        response_file = session_dir / f"{session_id}.response"
        
        # Remove old response file
        if response_file.exists():
            response_file.unlink()
        
        # Write command
        with open(command_file, 'w') as f:
            json.dump(command, f)
        
        # Wait for response (timeout after 10 seconds)
        start_time = time.time()
        while time.time() - start_time < 10:
            if response_file.exists():
                try:
                    with open(response_file) as f:
                        response = json.load(f)
                    response_file.unlink()  # Clean up
                    return response.get('success', False), response.get('message', 'No message')
                except Exception:
                    pass
            time.sleep(0.1)
        
        return False, "Command timeout"
    
    @staticmethod
    def cleanup_session(session_id: str):
        """Clean up a dead session."""
        session_dir = SessionManager.get_session_dir()
        for suffix in ['json', 'lock', 'pid', 'cmd', 'response']:
            file_path = session_dir / f"{session_id}.{suffix}"
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass