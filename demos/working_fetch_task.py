#!/usr/bin/env python3
"""
Working FetchPickAndPlace task completion using Claude's Direct Control System.
This actually picks up the block and places it at the target location.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time
import mujoco

class WorkingFetchTask:
    """Complete the FetchPickAndPlace task with actual block manipulation."""
    
    def __init__(self):
        print("🤖 Initializing Working Fetch Task Solver...")
        
        # Create environment
        self.env = gym.make("FetchPickAndPlace-v4", render_mode="human", max_episode_steps=100)
        self.obs, self.info = self.env.reset()
        
        # Get MuJoCo model and data
        self.model = self.env.unwrapped.model
        self.data = self.env.unwrapped.data
        
        # Get control ranges
        self.ctrl_range = self.model.actuator_ctrlrange.copy()
        
        print("✅ Environment ready")
        print(f"📦 Block at: {self.get_object_position()}")
        print(f"🎯 Target at: {self.get_target_position()}")
    
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
        # Site ID for gripper center
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "robot0:grip")
        return self.data.site_xpos[site_id].copy()
    
    def move_to_position(self, target_pos, grip_action=0.0, max_steps=50):
        """Move end-effector to target position using velocity control."""
        print(f"   Moving to [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        for step in range(max_steps):
            # Get current gripper position
            current_pos = self.get_gripper_position()
            
            # Calculate error
            error = target_pos - current_pos
            distance = np.linalg.norm(error)
            
            if distance < 0.005:  # Reached target
                print(f"   ✅ Reached target (error: {distance:.4f}m)")
                break
            
            # Velocity control with proportional gain
            velocity = np.clip(error * 10, -1, 1)  # P-controller with gain=10
            
            # Create action [x_vel, y_vel, z_vel, gripper]
            action = np.array([velocity[0], velocity[1], velocity[2], grip_action], dtype=np.float32)
            
            # Step environment
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            
            if step % 10 == 0:
                print(f"      Step {step}: distance = {distance:.4f}m")
        
        return distance < 0.01  # Success if within 1cm
    
    def control_gripper(self, open_gripper, steps=15):
        """Control gripper open/close."""
        grip_action = 1.0 if open_gripper else -1.0
        action_name = "Opening" if open_gripper else "Closing"
        print(f"   {action_name} gripper...")
        
        # Apply gripper action
        for _ in range(steps):
            action = np.array([0.0, 0.0, 0.0, grip_action], dtype=np.float32)
            self.obs, reward, terminated, truncated, info = self.env.step(action)
            time.sleep(0.02)
        
        return True
    
    def check_grasp_success(self):
        """Check if object is grasped."""
        object_pos = self.get_object_position()
        gripper_pos = self.get_gripper_position()
        distance = np.linalg.norm(object_pos - gripper_pos)
        
        # Check if object is close to gripper and lifted
        grasped = distance < 0.05 and object_pos[2] > 0.43
        return grasped, distance, object_pos[2]
    
    def check_task_success(self):
        """Check if task is completed (object at target)."""
        object_pos = self.get_object_position()
        target_pos = self.get_target_position()
        distance = np.linalg.norm(object_pos - target_pos)
        
        success = distance < 0.05  # Within 5cm of target
        return success, distance
    
    def complete_task(self):
        """Complete the full pick and place task."""
        print("\n🎯 STARTING FETCH PICK AND PLACE TASK")
        print("=" * 50)
        
        # Get positions
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
        approach_pos[2] += 0.10  # 10cm above object
        success = self.move_to_position(approach_pos, grip_action=1.0)
        
        # Step 3: Descend to object
        print("\n3️⃣ PHASE 3: DESCEND TO OBJECT")
        grasp_pos = object_pos.copy()
        grasp_pos[2] += 0.005  # Just above object surface
        success = self.move_to_position(grasp_pos, grip_action=1.0, max_steps=30)
        
        # Step 4: Close gripper
        print("\n4️⃣ PHASE 4: GRASP OBJECT")
        self.control_gripper(open_gripper=False)
        time.sleep(0.5)  # Let physics settle
        
        # Check grasp
        grasped, distance, height = self.check_grasp_success()
        print(f"   Grasp check: distance={distance:.4f}m, height={height:.3f}m")
        
        # Step 5: Lift object
        print("\n5️⃣ PHASE 5: LIFT OBJECT")
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.15  # Lift 15cm
        success = self.move_to_position(lift_pos, grip_action=-1.0, max_steps=40)
        
        # Verify pickup
        grasped, distance, height = self.check_grasp_success()
        if grasped:
            print(f"   ✅ OBJECT PICKED UP! Height: {height:.3f}m")
        else:
            print(f"   ⚠️ Pickup uncertain: distance={distance:.4f}m, height={height:.3f}m")
        
        # Step 6: Move to target position
        print("\n6️⃣ PHASE 6: TRANSPORT TO TARGET")
        transport_pos = target_pos.copy()
        transport_pos[2] += 0.10  # Above target position
        success = self.move_to_position(transport_pos, grip_action=-1.0, max_steps=60)
        
        # Step 7: Lower to place position
        print("\n7️⃣ PHASE 7: LOWER TO PLACE")
        place_pos = target_pos.copy()
        place_pos[2] += 0.05  # Slightly above target
        success = self.move_to_position(place_pos, grip_action=-1.0, max_steps=30)
        
        # Step 8: Release object
        print("\n8️⃣ PHASE 8: RELEASE OBJECT")
        self.control_gripper(open_gripper=True)
        time.sleep(0.5)  # Let object settle
        
        # Step 9: Retract gripper
        print("\n9️⃣ PHASE 9: RETRACT")
        retract_pos = place_pos.copy()
        retract_pos[2] += 0.10
        success = self.move_to_position(retract_pos, grip_action=1.0, max_steps=30)
        
        # Check final task success
        task_success, final_distance = self.check_task_success()
        
        print("\n" + "=" * 50)
        print("📊 TASK COMPLETION RESULTS:")
        
        if task_success:
            print("🎉 ✅ TASK SUCCESSFULLY COMPLETED!")
            print(f"   • Object placed within {final_distance:.3f}m of target")
            print(f"   • Final object position: {self.get_object_position()}")
            print(f"   • Target position: {self.get_target_position()}")
            print("   • Claude's Direct Control System achieved the goal!")
        else:
            print(f"❌ Task incomplete: {final_distance:.3f}m from target")
            print(f"   • Object at: {self.get_object_position()}")
            print(f"   • Target at: {self.get_target_position()}")
        
        # Get final reward from environment
        if isinstance(self.info, dict) and 'is_success' in self.info:
            env_success = self.info['is_success']
            print(f"\n🏆 Environment Success Flag: {'✅ SUCCESS' if env_success else '❌ FAILED'}")
        
        return task_success
    
    def run_demo(self):
        """Run the complete demonstration."""
        print("🤖 CLAUDE'S DIRECT CONTROL SYSTEM")
        print("Solving FetchPickAndPlace Task")
        print("=" * 60)
        
        try:
            # Complete the task
            success = self.complete_task()
            
            print("\n🔬 TECHNICAL SUMMARY:")
            print("   • Control Method: Direct velocity control")
            print("   • Feedback: Real-time position sensing")
            print("   • Planning: Explicit phase-based approach")
            print("   • No Learning: Immediate task completion")
            
            # Keep visualization open
            print("\n⏸️  Task complete! Window will close in 5 seconds...")
            for i in range(5):
                time.sleep(1)
                print(f"   {5-i}...")
                # Keep stepping to see final state
                action = np.array([0, 0, 0, 0], dtype=np.float32)
                self.obs, _, _, _, _ = self.env.step(action)
            
        finally:
            self.env.close()

def main():
    """Main entry point."""
    demo = WorkingFetchTask()
    demo.run_demo()

if __name__ == "__main__":
    main()