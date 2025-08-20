#!/usr/bin/env python3
"""
Fetch Robot Session Management for Unix-style CLI tools.

This module provides shared session management allowing multiple CLI tools
to interact with a single persistent robot environment.
"""

import os
import sys
import json
import time
import uuid
import signal
import threading
import tempfile
import socket
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import mujoco
import cv2
from .fetch_protocol import CommandType, StateQuery, validate_action_vector, clamp_action_vector
from .session_registry import get_registry

class FetchSession:
    """Manages a persistent Fetch robot session with shared state."""
    
    def __init__(self, session_id: str, show_ui: bool = True):
        self.session_id = session_id
        self.show_ui = show_ui
        self.session_dir = Path(tempfile.gettempdir()) / "fetch-sessions"
        self.session_dir.mkdir(exist_ok=True)
        
        # Socket path
        self.socket_path = self.session_dir / f"{session_id}.sock"
        self.pid_file = self.session_dir / f"{session_id}.pid"
        
        # Environment
        self.env = None
        self.obs = None
        self.info = None
        
        # State
        self.running = True
        self.last_action_time = time.time()
        self.current_grip_action = 0.0
        
        # Thread synchronization for env.step()
        self.env_lock = threading.Lock()
        
        # DirectExecutor now handles rendering directly
        
        # Command queue for thread-safe execution
        import queue
        import uuid
        self.uuid = uuid
        self.command_queue = queue.Queue()
        self.command_results = {}  # Store results keyed by command ID
        self.result_events = {}    # Events to signal when results are ready
        
        # Socket server
        self.socket_server = None
        self.socket_thread = None
        
        # UI setup for clean display
        self.cv_window_name = f'Fetch Robot - Session {session_id}'
        
        # Initialize environment
        self._init_environment()
        
        # Initialize socket server
        self._init_socket_server()
        
        # Write PID file
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        
        # Register session in registry for direct API access
        registry = get_registry()
        registry.register_session(session_id, pid=os.getpid())
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._cleanup_handler)
        signal.signal(signal.SIGINT, self._cleanup_handler)
        
        print(f"✅ Session {session_id} started")
        print(f"🔌 Socket: {self.socket_path}")
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
        
        # Register environment in registry for direct API access
        registry = get_registry()
        registry.set_environment(self.session_id, self.env, self.model, self.data)
    
    def _init_socket_server(self):
        """Initialize Unix domain socket server for IPC."""
        # Remove existing socket file if it exists
        if self.socket_path.exists():
            self.socket_path.unlink()
        
        # Create Unix domain socket
        self.socket_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket_server.bind(str(self.socket_path))
        self.socket_server.listen(5)  # Allow up to 5 concurrent connections
        
        # Start socket server thread
        self.socket_thread = threading.Thread(target=self._socket_server_loop, daemon=True)
        self.socket_thread.start()
        
        print(f"🔌 Socket server listening on {self.socket_path}")
    
    def _socket_server_loop(self):
        """Main socket server loop to handle incoming connections."""
        print(f"🔄 Socket server loop starting for {self.session_id}")
        while self.running:
            try:
                print(f"🔍 Waiting for socket connection...")
                conn, addr = self.socket_server.accept()
                print(f"🔌 New socket connection accepted")
                
                # Handle each connection in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_socket_client, 
                    args=(conn,), 
                    daemon=True
                )
                client_thread.start()
                print(f"🧵 Client thread started")
                
            except OSError as e:
                # Socket was closed
                print(f"🛑 Socket server OSError (normal shutdown): {e}")
                break
            except Exception as e:
                if self.running:
                    print(f"❌ Socket server error: {e}")
                    import traceback
                    traceback.print_exc()
        print(f"🏁 Socket server loop ended for {self.session_id}")
    
    def _handle_socket_client(self, conn):
        """Handle a single socket client connection."""
        print(f"🔗 Socket client handler starting")
        try:
            # Handle one request per connection (request-response pattern)
            # Read message length (4 bytes)
            print(f"📖 Reading message length...")
            length_bytes = self._recv_all(conn, 4)
            if not length_bytes:
                print(f"❌ No length bytes received")
                return
            
            message_length = int.from_bytes(length_bytes, byteorder='big')
            print(f"📏 Message length: {message_length}")
            
            # Read the actual message
            print(f"📖 Reading message content...")
            message_bytes = self._recv_all(conn, message_length)
            if not message_bytes:
                print(f"❌ No message bytes received")
                return
            
            # Decode and process command
            print(f"🔍 Decoding JSON message...")
            command = json.loads(message_bytes.decode('utf-8'))
            print(f"🔍 Processing command: {command.get('action', 'unknown')}")
            print(f"🔍 Full command: {command}")
            
            try:
                print(f"🚀 Executing action...")
                success, response = self.execute_action(
                    command.get('action', ''),
                    **{k: v for k, v in command.items() if k != 'action'}
                )
                print(f"✅ Command completed: success={success}")
                print(f"📝 Response: {response[:200]}..." if len(str(response)) > 200 else f"📝 Response: {response}")
                
            except Exception as e:
                print(f"❌ Command failed with exception: {e}")
                import traceback
                traceback.print_exc()
                success, response = False, f"Command processing failed: {str(e)}"
            
            # Send response
            print(f"📤 Sending response...")
            response_data = {
                'success': success,
                'message': response
            }
            response_json = json.dumps(response_data).encode('utf-8')
            response_length = len(response_json)
            
            # Send length followed by message
            conn.sendall(response_length.to_bytes(4, byteorder='big'))
            conn.sendall(response_json)
            print(f"✅ Response sent successfully")
            
        except (ConnectionResetError, BrokenPipeError):
            # Client disconnected - normal
            print(f"🔌 Client disconnected normally")
            pass
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON from client: {e}")
        except Exception as e:
            print(f"❌ Socket client error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"🧹 Closing client connection")
            try:
                conn.close()
            except:
                pass
    
    def _recv_all(self, sock, n):
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
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
    
    # Render callback no longer needed - DirectExecutor renders directly
    
    def _process_queued_command(self, command: dict, executor):
        """Process a command queued from socket threads on main thread."""
        command_id = command['id']
        command_type = command['type']
        
        print(f"🎯 Processing queued command {command_id}: {command_type}")
        
        try:
            if command_type == 'direct_api_call':
                method = command['method']
                args = command['args']
                kwargs = command['kwargs']
                
                print(f"🔧 Executing {method} on main thread")
                
                # Get the method and call it on main thread
                if hasattr(executor, method):
                    method_func = getattr(executor, method)
                    result = method_func(*args, **kwargs)
                    
                    # Convert result to JSON string
                    import json
                    json_result = json.dumps(result)
                    self.command_results[command_id] = (True, json_result)
                    print(f"✅ Command {command_id} completed successfully")
                else:
                    self.command_results[command_id] = (False, f"Method '{method}' not found on DirectExecutor")
                    print(f"❌ Method '{method}' not found")
            else:
                self.command_results[command_id] = (False, f"Unknown command type: {command_type}")
                print(f"❌ Unknown command type: {command_type}")
        
        except Exception as e:
            print(f"❌ Command {command_id} failed: {e}")
            import traceback
            traceback.print_exc()
            self.command_results[command_id] = (False, f"Command failed: {str(e)}")
        
        # DirectExecutor already rendered during execution
        # Signal that result is ready
        if command_id in self.result_events:
            self.result_events[command_id].set()
            print(f"🚨 Signaled result ready for command {command_id}")
    
    def _cleanup_handler(self, signum, frame):
        """Handle cleanup on signal."""
        print(f"\n🛑 Session {self.session_id} received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up session resources."""
        self.running = False
        
        # Close socket server
        if self.socket_server:
            try:
                self.socket_server.close()
                print("🔌 Socket server closed")
            except Exception as e:
                print(f"Warning: Could not close socket server: {e}")
        
        # Remove socket file
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
                print("🗑️  Socket file removed")
            except Exception as e:
                print(f"Warning: Could not remove socket file: {e}")
        
        # Unregister from registry
        try:
            registry = get_registry()
            registry.unregister_session(self.session_id)
        except Exception as e:
            print(f"Warning: Could not unregister session: {e}")
        
        # Close OpenCV windows first (most important for preventing dialog)
        if self.show_ui:
            try:
                cv2.destroyAllWindows()
                # Give time for windows to actually close
                time.sleep(0.1)
                # Force process all window events
                for _ in range(10):
                    cv2.waitKey(1)
            except Exception as e:
                print(f"Warning: Could not close OpenCV windows: {e}")
        
        # Close environment
        if self.env:
            try:
                self.env.close()
                # Give MuJoCo time to clean up
                time.sleep(0.1)
            except Exception as e:
                print(f"Warning: Could not close environment: {e}")
        
        # Remove PID file
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except Exception as e:
                print(f"Warning: Could not remove PID file: {e}")
        
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
        """Update the internal state (state is now accessed via socket queries)."""
        # State is now queried on-demand via socket commands
        # No need to write to files anymore
        pass
    
    def execute_action(self, action_type: str, **kwargs) -> Tuple[bool, str]:
        """Execute a low-level robot action using the new protocol."""
        print(f"⚡ execute_action called: {action_type}, kwargs: {kwargs}")
        self.last_action_time = time.time()
        
        try:
            if action_type == CommandType.RAW_STEP.value:
                print(f"🎯 Executing RAW_STEP")
                return self._execute_raw_step(kwargs.get('action_vector'), kwargs.get('steps', 1))
            elif action_type == CommandType.BATCH_STEPS.value:
                print(f"🎯 Executing BATCH_STEPS")
                return self._execute_batch_steps(kwargs.get('action_sequence'), kwargs.get('step_delay', 0.02))
            elif action_type == CommandType.GET_STATE.value:
                print(f"🎯 Executing GET_STATE")
                return self._get_detailed_state(kwargs.get('query_type', StateQuery.FULL.value))
            elif action_type == CommandType.SET_JOINT.value:
                print(f"🎯 Executing SET_JOINT")
                return self._set_joint_angle(kwargs.get('joint_name'), kwargs.get('angle'))
            elif action_type == CommandType.RESET.value:
                print(f"🎯 Executing RESET")
                return self._reset_environment()
            elif action_type == CommandType.DIRECT_API_CALL.value:
                print(f"🎯 Executing DIRECT_API_CALL: method={kwargs.get('method')}")
                result = self._execute_direct_api_call(kwargs.get('method'), kwargs.get('args', []), kwargs.get('kwargs', {}))
                print(f"🎯 DIRECT_API_CALL result: {result}")
                return result
            elif action_type == CommandType.EMBEDDED_SCRIPT.value:
                print(f"🎯 Executing EMBEDDED_SCRIPT")
                return self._execute_embedded_script(kwargs.get('script_code'), kwargs.get('script_globals', {}))
            # Backward compatibility for legacy commands
            elif action_type in ['move', 'move_with_grip', 'gripper', 'lift', 'wave']:
                print(f"❌ Legacy action not supported: {action_type}")
                return False, f"Legacy action '{action_type}' no longer supported. Use low-level protocol commands."
            else:
                print(f"❌ Unknown action type: {action_type}")
                return False, f"Unknown action type: {action_type}"
        
        except Exception as e:
            print(f"❌ Action failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Action failed: {str(e)}"
        
        finally:
            print(f"🔄 Updating state after action")
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
    
    def _execute_direct_api_call(self, method: str, args: list, kwargs: dict) -> Tuple[bool, str]:
        """
        Queue a Direct API method for execution on the main thread.
        
        Args:
            method: Name of the DirectExecutor method to call
            args: Positional arguments
            kwargs: Keyword arguments
            
        Returns:
            (success, result_json)
        """
        print(f"🔧 _execute_direct_api_call: method={method}, args={args}, kwargs={kwargs}")
        
        # Generate unique command ID
        command_id = str(self.uuid.uuid4())
        
        # Create command for main thread execution
        command = {
            'id': command_id,
            'type': 'direct_api_call',
            'method': method,
            'args': args,
            'kwargs': kwargs
        }
        
        # Create event to wait for result
        result_event = threading.Event()
        self.result_events[command_id] = result_event
        
        print(f"📝 Queuing command {command_id} for main thread execution")
        self.command_queue.put(command)
        
        # Wait for result (with timeout)
        print(f"⏳ Waiting for result...")
        if result_event.wait(timeout=30):  # 30 second timeout
            # Get result
            result = self.command_results.pop(command_id)
            self.result_events.pop(command_id)
            print(f"✅ Got result: {result}")
            return result
        else:
            # Timeout
            self.result_events.pop(command_id, None)
            print(f"⏰ Command timed out")
            return False, "Command execution timed out"
    
    def _execute_embedded_script(self, script_code: str, script_globals: dict) -> Tuple[bool, str]:
        """
        Execute Python code within the session process with direct access to executor.
        
        Args:
            script_code: Python code to execute
            script_globals: Global variables for the script
            
        Returns:
            (success, result_json)
        """
        try:
            from .direct_executor import DirectExecutor
            import json
            
            # Create DirectExecutor with our environment
            executor = DirectExecutor(self.env, self.model, self.data)
            
            # Set up execution environment
            exec_globals = {
                'executor': executor,
                'env': self.env,
                'model': self.model,
                'data': self.data,
                'np': np,
                'time': time,
                'json': json,
                **script_globals  # User-provided globals
            }
            
            exec_locals = {}
            
            # Execute the script
            exec(script_code, exec_globals, exec_locals)
            
            # Return result (check if script set a 'result' variable)
            if 'result' in exec_locals:
                return True, json.dumps(exec_locals['result'])
            else:
                return True, json.dumps({"success": True, "message": "Script executed successfully"})
                
        except Exception as e:
            return False, f"Embedded script failed: {str(e)}"
    
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
        print(f"🚀 Starting session run loop for {self.session_id}")
        try:
            # Create DirectExecutor for main thread execution
            from .direct_executor import DirectExecutor
            executor = DirectExecutor(self.env, self.model, self.data, env_lock=self.env_lock, 
                                    cv_window_name=self.cv_window_name, show_ui=self.show_ui)
            print(f"✅ DirectExecutor created for main thread")
            
            while self.running:
                # Process queued commands from socket threads
                while not self.command_queue.empty():
                    try:
                        command = self.command_queue.get_nowait()
                        self._process_queued_command(command, executor)
                    except:
                        break  # Queue is empty
                
                # DirectExecutor now handles real-time rendering
                
                # Keep environment alive by stepping with current grip state maintained
                if time.time() - self.last_action_time > 1.0:
                    with self.env_lock:
                        action = np.array([0.0, 0.0, 0.0, self.current_grip_action], dtype=np.float32)
                        self.obs, _, _, _, self.info = self.env.step(action)
                        self._render_frame()  # Keep display updated
                        self.last_action_time = time.time()
                
                # Update state periodically  
                self._update_state()
                time.sleep(0.01)  # 10ms sleep for up to 100fps potential
                
        except KeyboardInterrupt:
            print(f"\n💡 Session {self.session_id} interrupted by user")
        except Exception as e:
            print(f"\n❌ Session {self.session_id} error in run loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"🏁 Session {self.session_id} ending...")
            self.cleanup()
    


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
        """Get the current state of a session via socket query."""
        from .fetch_protocol import create_get_state_command, StateQuery
        
        # Send GET_STATE command via socket
        command = create_get_state_command(StateQuery.FULL)
        success, response = SessionManager.send_command(session_id, command)
        
        if success:
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return None
        return None
    
    @staticmethod
    def send_command(session_id: str, command: Dict[str, Any]) -> Tuple[bool, str]:
        """Send a command to a running session via Unix domain socket."""
        session_dir = SessionManager.get_session_dir()
        socket_path = session_dir / f"{session_id}.sock"
        
        if not socket_path.exists():
            return False, f"Session {session_id} socket not found"
        
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(30)  # 30 second timeout
            sock.connect(str(socket_path))
            
            # Send command
            command_json = json.dumps(command).encode('utf-8')
            command_length = len(command_json)
            
            # Send length followed by message
            sock.sendall(command_length.to_bytes(4, byteorder='big'))
            sock.sendall(command_json)
            
            # Receive response length
            length_bytes = SessionManager._recv_all(sock, 4)
            if not length_bytes:
                return False, "No response from session"
            
            response_length = int.from_bytes(length_bytes, byteorder='big')
            
            # Receive response message
            response_bytes = SessionManager._recv_all(sock, response_length)
            if not response_bytes:
                return False, "Incomplete response from session"
            
            # Parse response
            response = json.loads(response_bytes.decode('utf-8'))
            return response.get('success', False), response.get('message', 'No message')
            
        except socket.timeout:
            return False, "Command timeout"
        except (ConnectionRefusedError, FileNotFoundError):
            return False, f"Cannot connect to session {session_id}"
        except Exception as e:
            return False, f"Socket communication error: {str(e)}"
        finally:
            try:
                sock.close()
            except:
                pass
    
    @staticmethod
    def _recv_all(sock, n):
        """Receive exactly n bytes from socket."""
        data = b''
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data += packet
        return data
    
    @staticmethod
    def cleanup_session(session_id: str):
        """Clean up a dead session."""
        session_dir = SessionManager.get_session_dir()
        # Clean up socket and PID files
        for suffix in ['sock', 'pid']:
            file_path = session_dir / f"{session_id}.{suffix}"
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception:
                    pass