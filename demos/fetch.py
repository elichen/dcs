#!/usr/bin/env python3
"""
Fetch robot pick-and-place demonstration using Claude's Direct Control System.
Shows UI by default, with optional GIF recording capability.

Usage:
    python fetch.py              # Run with UI display
    python fetch.py --gif         # Run with UI and save as GIF
    python fetch.py --gif-only    # Run headless and save as GIF only
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import mujoco
import argparse
import os
from datetime import datetime

# Only import imageio if GIF recording is requested
imageio = None

class FetchDemo:
    """Complete pick-and-place task with Fetch robot using Direct Control System."""
    
    def __init__(self, show_ui=True, record_gif=False):
        """
        Initialize the Fetch demo.
        
        Args:
            show_ui: Whether to display the UI window
            record_gif: Whether to record frames for GIF output
        """
        print("🤖 Initializing Fetch Robot Demo...")
        
        self.show_ui = show_ui
        self.record_gif = record_gif
        
        # Set render mode based on options
        if show_ui:
            render_mode = "human"
        else:
            render_mode = "rgb_array" if record_gif else None
        
        # Create environment
        self.env = gym.make("FetchPickAndPlace-v4", render_mode=render_mode, max_episode_steps=100)
        self.obs, self.info = self.env.reset()
        
        # Get MuJoCo model and data
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        # Get control ranges
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Recording setup
        if self.record_gif:
            global imageio
            import imageio as imageio_import
            imageio = imageio_import
            self.frames = []
            self.frame_counter = 0
            self.record_every_n_frames = 2  # Record every other frame to reduce file size
            
            # Create output directory
            self.output_dir = "/Users/elichen/code/dcs/recordings"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"📹 GIF recording enabled (capturing every {self.record_every_n_frames} frames)")
        
        print("✅ Environment ready")
        print(f"📦 Block at: {self.get_object_position()}")
        print(f"🎯 Target at: {self.get_target_position()}")
        if self.show_ui:
            print("🖥️  UI display enabled")
    
    def capture_frame(self):
        """Capture current frame for GIF if recording is enabled."""
        if not self.record_gif:
            return
        
        self.frame_counter += 1
        if self.frame_counter % self.record_every_n_frames == 0:
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)
    
    def save_gif(self, filename=None, fps=15):
        """Save recorded frames as GIF."""
        if not self.record_gif or not self.frames:
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fetch_demo_{timestamp}.gif"
        
        filepath = os.path.join(self.output_dir, filename)
        
        print(f"\n💾 Saving GIF with {len(self.frames)} frames...")
        imageio.mimsave(filepath, self.frames, fps=fps, loop=0)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ GIF saved! Size: {file_size_mb:.2f} MB")
        print(f"📍 Location: {filepath}")
        
        return filepath
    
    def get_object_position(self):
        """Get current block position."""
        if isinstance(self.obs, dict) and 'achieved_goal' in self.obs:
            return self.obs['achieved_goal'][:3]
        return np.array([1.3, 0.7, 0.425])
    
    def get_target_position(self):
        """Get target position for placement."""
        if isinstance(self.obs, dict) and 'desired_goal' in self.obs:
            return self.obs['desired_goal'][:3]
        return np.array([1.3, 0.9, 0.425])
    
    def get_gripper_position(self):
        """Get current gripper position."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        return self.data.site_xpos[site_id].copy()
    
    def move_to_position(self, target_pos, grip_action=0.0, max_steps=50):
        """Move end-effector to target position using velocity control."""
        print(f"   Moving to [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        for step in range(max_steps):
            current_pos = self.get_gripper_position()
            error = target_pos - current_pos
            distance = np.linalg.norm(error)
            
            if distance < 0.005:
                print(f"   ✅ Reached target (error: {distance:.4f}m)")
                break
            
            # Velocity control with proportional gain
            velocity = np.clip(error * 10, -1, 1)
            action = np.array([velocity[0], velocity[1], velocity[2], grip_action], dtype=np.float32)
            
            # Step environment
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.capture_frame()
            time.sleep(0.05)  # Control speed
            
            if step % 10 == 0:
                print(f"      Step {step}: distance = {distance:.4f}m")
        
        return distance < 0.01
    
    def control_gripper(self, open_gripper, steps=15):
        """Control gripper open/close."""
        grip_action = 1.0 if open_gripper else -1.0
        action_name = "Opening" if open_gripper else "Closing"
        print(f"   {action_name} gripper...")
        
        for _ in range(steps):
            action = np.array([0.0, 0.0, 0.0, grip_action], dtype=np.float32)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            self.capture_frame()
            time.sleep(0.05)
        
        return True
    
    def check_grasp_success(self):
        """Check if object is grasped."""
        object_pos = self.get_object_position()
        gripper_pos = self.get_gripper_position()
        distance = np.linalg.norm(object_pos - gripper_pos)
        grasped = distance < 0.05 and object_pos[2] > 0.43
        return grasped, distance, object_pos[2]
    
    def check_task_success(self):
        """Check if task is completed."""
        object_pos = self.get_object_position()
        target_pos = self.get_target_position()
        distance = np.linalg.norm(object_pos - target_pos)
        success = distance < 0.05
        return success, distance
    
    def complete_task(self):
        """Complete the full pick and place task."""
        print("\n🎯 STARTING FETCH PICK AND PLACE TASK")
        print("=" * 50)
        
        # Capture initial frame
        self.capture_frame()
        
        object_pos = self.get_object_position()
        target_pos = self.get_target_position()
        gripper_pos = self.get_gripper_position()
        
        print(f"📦 Object at: [{object_pos[0]:.3f}, {object_pos[1]:.3f}, {object_pos[2]:.3f}]")
        print(f"🎯 Target at: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"🤖 Gripper at: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
        
        # Step 1: Open gripper
        print("\n1️⃣ PHASE 1: PREPARE GRIPPER")
        self.control_gripper(open_gripper=True)
        
        # Step 2: Move above object
        print("\n2️⃣ PHASE 2: APPROACH OBJECT")
        approach_pos = object_pos.copy()
        approach_pos[2] += 0.10
        self.move_to_position(approach_pos, grip_action=1.0)
        
        # Step 3: Descend to object
        print("\n3️⃣ PHASE 3: DESCEND TO OBJECT")
        grasp_pos = object_pos.copy()
        grasp_pos[2] += 0.005
        self.move_to_position(grasp_pos, grip_action=1.0, max_steps=30)
        
        # Step 4: Close gripper
        print("\n4️⃣ PHASE 4: GRASP OBJECT")
        self.control_gripper(open_gripper=False)
        time.sleep(0.5)
        
        grasped, distance, height = self.check_grasp_success()
        print(f"   Grasp check: distance={distance:.4f}m, height={height:.3f}m")
        
        # Step 5: Lift object
        print("\n5️⃣ PHASE 5: LIFT OBJECT")
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.15
        self.move_to_position(lift_pos, grip_action=-1.0, max_steps=40)
        
        grasped, distance, height = self.check_grasp_success()
        if grasped:
            print(f"   ✅ OBJECT PICKED UP! Height: {height:.3f}m")
        
        # Step 6: Move to target position
        print("\n6️⃣ PHASE 6: TRANSPORT TO TARGET")
        transport_pos = target_pos.copy()
        transport_pos[2] += 0.10
        self.move_to_position(transport_pos, grip_action=-1.0, max_steps=60)
        
        # Step 7: Lower to place position
        print("\n7️⃣ PHASE 7: LOWER TO PLACE")
        place_pos = target_pos.copy()
        place_pos[2] += 0.05
        self.move_to_position(place_pos, grip_action=-1.0, max_steps=30)
        
        # Step 8: Release object
        print("\n8️⃣ PHASE 8: RELEASE OBJECT")
        self.control_gripper(open_gripper=True)
        time.sleep(0.5)
        
        # Step 9: Retract gripper
        print("\n9️⃣ PHASE 9: RETRACT")
        retract_pos = place_pos.copy()
        retract_pos[2] += 0.10
        self.move_to_position(retract_pos, grip_action=1.0, max_steps=30)
        
        # Capture final frames for GIF
        if self.record_gif:
            for _ in range(10):
                self.capture_frame()
                action = np.array([0, 0, 0, 0], dtype=np.float32)
                self.obs, _, _, _, _ = self.env.step(action)
                time.sleep(0.05)
        
        # Check final task success
        task_success, final_distance = self.check_task_success()
        
        print("\n" + "=" * 50)
        print("📊 TASK COMPLETION RESULTS:")
        
        if task_success:
            print("🎉 ✅ TASK SUCCESSFULLY COMPLETED!")
            print(f"   • Object placed within {final_distance:.3f}m of target")
        else:
            print(f"❌ Task incomplete: {final_distance:.3f}m from target")
            print(f"   • Object at: {self.get_object_position()}")
            print(f"   • Target at: {self.get_target_position()}")
        
        return task_success
    
    def run(self):
        """Run the complete demonstration."""
        print("\n🤖 CLAUDE'S DIRECT CONTROL SYSTEM")
        print("Fetch Robot Pick-and-Place Demonstration")
        print("=" * 60)
        
        try:
            # Complete the task
            success = self.complete_task()
            
            print("\n🔬 TECHNICAL SUMMARY:")
            print("   • Control Method: Direct velocity control")
            print("   • Feedback: Real-time position sensing")
            print("   • Planning: Explicit phase-based approach")
            print("   • No Learning: Immediate task completion")
            
            # Save GIF if recording
            if self.record_gif and self.frames:
                gif_path = self.save_gif()
                print(f"\n📹 Recording saved to: {gif_path}")
            
            # Keep window open briefly if UI is shown
            if self.show_ui:
                print("\n⏸️  Demo complete! Window will close in 5 seconds...")
                for i in range(5):
                    time.sleep(1)
                    print(f"   {5-i}...")
                    action = np.array([0, 0, 0, 0], dtype=np.float32)
                    self.obs, _, _, _, _ = self.env.step(action)
            
        finally:
            self.env.close()

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Fetch robot pick-and-place demo with Direct Control System"
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="Record the demo as an animated GIF (shows UI too)"
    )
    parser.add_argument(
        "--gif-only",
        action="store_true",
        help="Record GIF without showing UI (headless mode)"
    )
    
    args = parser.parse_args()
    
    # Determine settings based on arguments
    if args.gif_only:
        show_ui = False
        record_gif = True
        print("🎬 Running in headless mode with GIF recording...")
    elif args.gif:
        show_ui = True
        record_gif = True
        print("🎬 Running with UI display and GIF recording...")
    else:
        show_ui = True
        record_gif = False
        print("🖥️  Running with UI display only...")
    
    # Create and run demo
    demo = FetchDemo(show_ui=show_ui, record_gif=record_gif)
    demo.run()

if __name__ == "__main__":
    main()