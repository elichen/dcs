#!/usr/bin/env python3
"""
Gentle precision sliding with single, carefully calculated push.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from lib.fetch_api import FetchAPI

def get_positions(api):
    """Get current object and target positions."""
    state = api.get_state()
    obj_pos = state.get('object_position', [1.0, 0.8, 0.41])
    target_pos = state.get('target_position', [1.6, 0.8, 0.41])
    gripper_pos = state.get('gripper_position', [1.0, 0.7, 0.45])
    return obj_pos, target_pos, gripper_pos

def calculate_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return math.sqrt(sum((a - b)**2 for a, b in zip(pos1, pos2)))

def gentle_precision_slide(session_id):
    """Execute gentle precision sliding with single calculated push."""
    
    print(f"🎯 Starting gentle precision slide with session {session_id}")
    
    # Connect to session
    api = FetchAPI.connect(session_id)
    
    # Step 1: Capture and analyze initial state
    print("\n📸 Step 1: Analyzing initial state")
    success, initial_path = api.capture_image("gentle_01_initial")
    obj_pos, target_pos, gripper_pos = get_positions(api)
    
    print(f"📍 Object: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"🎯 Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"🤖 Gripper: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
    
    # Calculate push vector and distance
    push_vector = [target_pos[0] - obj_pos[0], target_pos[1] - obj_pos[1], 0]
    push_distance = math.sqrt(push_vector[0]**2 + push_vector[1]**2)
    push_unit = [push_vector[0] / push_distance, push_vector[1] / push_distance, 0]
    
    print(f"📏 Distance to target: {push_distance:.3f}m = {push_distance*1000:.1f}mm")
    print(f"📐 Push direction: [{push_unit[0]:.3f}, {push_unit[1]:.3f}]")
    
    # Step 2: Close gripper for contact
    print("\n🤏 Step 2: Preparing gripper")
    api.grip(False)  # Close for solid contact
    print("✅ Gripper closed")
    
    # Step 3: Lift to safe height
    print("\n⬆️  Step 3: Lifting to safe height")
    safe_height = 0.55
    success, msg = api.move_to([gripper_pos[0], gripper_pos[1], safe_height], maintain_grip=True)
    print(f"✅ Lifted to {safe_height}m")
    
    # Step 4: Position behind object with proper distance
    print("\n🎯 Step 4: Positioning behind object")
    
    # Position further behind for gentler approach
    behind_distance = 0.10  # 10cm behind object
    behind_x = obj_pos[0] - push_unit[0] * behind_distance
    behind_y = obj_pos[1] - push_unit[1] * behind_distance
    behind_z = safe_height
    
    print(f"📍 Behind position: [{behind_x:.3f}, {behind_y:.3f}, {behind_z:.3f}]")
    success, msg = api.move_to([behind_x, behind_y, behind_z], maintain_grip=True, max_steps=40)
    print(f"✅ Positioned behind object")
    
    # Capture positioning
    success, pos_path = api.capture_image("gentle_02_positioned")
    print(f"📸 Position captured: {pos_path}")
    
    # Step 5: Lower to push height gradually
    print("\n⬇️  Step 5: Lowering to push height")
    push_height = obj_pos[2] + 0.001  # Slightly above object
    success, msg = api.move_to([behind_x, behind_y, push_height], maintain_grip=True, max_steps=30)
    print(f"✅ At push height: {push_height:.3f}m")
    
    # Step 6: Calculate precise push target
    print("\n➡️  Step 6: Calculating precise push")
    
    # Push just to the target, no overshoot for gentleness
    # Add small safety margin (2cm short) to avoid overpushing
    safety_margin = 0.02
    effective_push_distance = max(0.05, push_distance - safety_margin)
    
    push_end_x = behind_x + push_unit[0] * effective_push_distance
    push_end_y = behind_y + push_unit[1] * effective_push_distance
    push_end_z = push_height
    
    print(f"🎯 Push target: [{push_end_x:.3f}, {push_end_y:.3f}, {push_end_z:.3f}]")
    print(f"📏 Effective push distance: {effective_push_distance:.3f}m")
    
    # Execute gentle push with more steps for control
    success, msg = api.move_to([push_end_x, push_end_y, push_end_z], maintain_grip=True, max_steps=80)
    print(f"🔄 Push result: {msg}")
    
    # Step 7: Check immediate result
    print("\n📏 Step 7: Checking result")
    
    # No delay - immediate result check
    
    final_obj_pos, _, _ = get_positions(api)
    final_distance = calculate_distance(final_obj_pos[:2], target_pos[:2])
    
    print(f"📍 Object moved to: [{final_obj_pos[0]:.3f}, {final_obj_pos[1]:.3f}, {final_obj_pos[2]:.3f}]")
    print(f"📏 Distance to target: {final_distance:.3f}m = {final_distance*1000:.1f}mm")
    
    # Step 8: Fine adjustment if needed
    if final_distance > 0.05:  # If more than 5cm away, try small adjustment
        print("\n🔧 Step 8: Fine adjustment needed")
        
        # Calculate small adjustment
        adj_vector = [target_pos[0] - final_obj_pos[0], target_pos[1] - final_obj_pos[1]]
        adj_distance = math.sqrt(adj_vector[0]**2 + adj_vector[1]**2)
        
        if adj_distance > 0.01:  # Only if meaningful distance
            adj_unit = [adj_vector[0] / adj_distance, adj_vector[1] / adj_distance]
            
            # Position for small push (closer to object)
            adj_behind_dist = 0.06  # 6cm for fine adjustment
            adj_behind_x = final_obj_pos[0] - adj_unit[0] * adj_behind_dist
            adj_behind_y = final_obj_pos[1] - adj_unit[1] * adj_behind_dist
            
            # Lift, position, lower, push gently
            api.move_to([adj_behind_x, adj_behind_y, safe_height], maintain_grip=True)
            api.move_to([adj_behind_x, adj_behind_y, push_height], maintain_grip=True)
            
            # Small gentle push
            adj_push_x = adj_behind_x + adj_unit[0] * min(0.08, adj_distance + 0.02)
            adj_push_y = adj_behind_y + adj_unit[1] * min(0.08, adj_distance + 0.02)
            
            success, msg = api.move_to([adj_push_x, adj_push_y, push_height], maintain_grip=True, max_steps=60)
            print(f"🔧 Adjustment result: {msg}")
            
            # Check final result immediately
            final_obj_pos, _, _ = get_positions(api)
            final_distance = calculate_distance(final_obj_pos[:2], target_pos[:2])
    
    # Final capture and assessment
    success, final_path = api.capture_image("gentle_03_final")
    print(f"📸 Final result: {final_path}")
    
    # Retract gripper
    print("\n⬆️  Step 9: Retracting gripper")
    api.lift(0.12)
    
    # Success assessment
    success_threshold = 0.04  # 4cm tolerance
    task_success = final_distance < success_threshold
    
    print(f"\n{'🎉 TASK SUCCESSFUL!' if task_success else '⚠️  Close attempt'}")
    print(f"📏 Final accuracy: {final_distance*1000:.1f}mm")
    print(f"🎯 Target threshold: {success_threshold*1000:.0f}mm")
    
    return task_success

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python gentle_slide.py <session-id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    success = gentle_precision_slide(session_id)
    sys.exit(0 if success else 1)