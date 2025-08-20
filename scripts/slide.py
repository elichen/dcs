#!/usr/bin/env python3
"""
Physics-aware precision sliding with iterative refinement.
"""

import sys
import json
import math
import numpy as np
from pathlib import Path

# Add parent directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

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

def wait_for_sliding_to_stop(api, timeout=3.0, stability_threshold=0.001):
    """Wait for object to stop sliding by monitoring position changes."""
    import time
    
    print(f"⏳ Waiting for sliding to stop...")
    
    start_time = time.time()
    last_pos = None
    stable_count = 0
    required_stable_readings = 8  # Need 8 consecutive stable readings (0.4 seconds)
    
    while time.time() - start_time < timeout:
        current_obj_pos, _, _ = get_positions(api)
        
        if last_pos is not None:
            movement = calculate_distance(current_obj_pos[:2], last_pos[:2])
            if movement < stability_threshold:  # Less than 1mm movement
                stable_count += 1
                if stable_count >= required_stable_readings:
                    elapsed = time.time() - start_time
                    print(f"✅ Object stable after {elapsed:.1f}s")
                    return True
            else:
                stable_count = 0  # Reset if object is still moving
                print(f"📍 Still sliding: {movement*1000:.1f}mm movement")
        
        last_pos = current_obj_pos[:2].copy()
        time.sleep(0.05)  # Check every 50ms
    
    print(f"⏰ Timeout after {timeout}s - assuming stopped")
    return False

def calculate_single_push(required_distance):
    """Calculate a single, ultra-conservative push distance based on measured physics."""
    
    # Load calibrated physics data
    try:
        with open(Path(__file__).parent.parent / 'calibrated_physics.json', 'r') as f:
            physics_data = json.load(f)
        expected_efficiency = physics_data['avg_efficiency']
    except (FileNotFoundError, KeyError):
        # Fallback to measured average if file not found
        expected_efficiency = 2.370
    
    safety_factor = 0.5  # Only push 50% to account for momentum and get closer
    
    # Calculate required push distance
    base_push = required_distance / expected_efficiency
    conservative_push = base_push * safety_factor
    
    # Clamp to reasonable bounds (1-6cm) - smaller range
    return max(0.01, min(0.06, conservative_push))

def execute_push(api, behind_pos, push_target, max_steps=60):
    """Execute a single push operation."""
    # Position behind object
    api.move_to(behind_pos, maintain_grip=True, max_steps=30)
    
    # Execute push
    success, msg = api.move_to(push_target, maintain_grip=True, max_steps=max_steps)
    return success, msg

def single_precision_push(api, obj_pos, target_pos, safe_height):
    """Execute a single, carefully calculated precision push."""
    
    print(f"\n🎯 Executing single precision push")
    
    # Get current positions
    current_obj_pos, _, _ = get_positions(api)
    required_distance = calculate_distance(current_obj_pos[:2], target_pos[:2])
    
    print(f"📏 Distance to target: {required_distance*1000:.1f}mm")
    
    # Calculate push vector and parameters
    push_vector = [target_pos[0] - current_obj_pos[0], target_pos[1] - current_obj_pos[1]]
    push_unit = [push_vector[0] / required_distance, push_vector[1] / required_distance]
    
    # Calculate conservative push distance
    push_distance = calculate_single_push(required_distance)
    
    # Position parameters - use calibrated values that worked
    behind_distance = 0.06  # 6cm behind for reliable contact
    push_height = current_obj_pos[2] + 0.002  # Slightly above object surface
    
    # Calculate positions
    behind_x = current_obj_pos[0] - push_unit[0] * behind_distance
    behind_y = current_obj_pos[1] - push_unit[1] * behind_distance
    behind_pos = [behind_x, behind_y, push_height]
    
    push_end_x = behind_x + push_unit[0] * push_distance
    push_end_y = behind_y + push_unit[1] * push_distance
    push_target = [push_end_x, push_end_y, push_height]
    
    print(f"📐 Push direction: [{push_unit[0]:.3f}, {push_unit[1]:.3f}]")
    print(f"📏 Conservative push distance: {push_distance*1000:.1f}mm")
    print(f"📍 Behind position: [{behind_x:.3f}, {behind_y:.3f}, {push_height:.3f}]")
    print(f"🎯 Push target: [{push_end_x:.3f}, {push_end_y:.3f}, {push_height:.3f}]")
    
    # Lift to safe height first
    api.move_to([current_obj_pos[0], current_obj_pos[1], safe_height], maintain_grip=True)
    
    # Execute the push
    success, msg = execute_push(api, behind_pos, push_target, max_steps=60)
    print(f"🔄 Push result: {msg}")
    
    # Wait for sliding to completely stop
    wait_for_sliding_to_stop(api)
    
    # Measure final result after sliding has stopped
    final_obj_pos, _, _ = get_positions(api)
    actual_movement = calculate_distance(current_obj_pos[:2], final_obj_pos[:2])
    final_distance = calculate_distance(final_obj_pos[:2], target_pos[:2])
    
    print(f"📊 Actual movement: {actual_movement*1000:.1f}mm")
    print(f"📏 Final distance to target: {final_distance*1000:.1f}mm")
    
    # Success thresholds
    success_threshold = 0.05  # 5cm acceptable
    precision_threshold = 0.03  # 3cm precision
    
    if final_distance <= precision_threshold:
        print(f"🎉 Precision achieved!")
        return True, final_distance
    elif final_distance <= success_threshold:
        print(f"✅ Acceptable accuracy!")
        return True, final_distance
    else:
        print(f"⚠️ Needs improvement")
        return False, final_distance

def precision_sliding_task(session_id):
    """Execute physics-aware precision sliding with iterative refinement."""
    
    print(f"🎯 Starting physics-aware precision slide with session {session_id}")
    
    # Connect to session
    api = FetchAPI.connect(session_id)
    
    # Step 1: Capture and analyze initial state
    print("\n📸 Step 1: Analyzing initial state")
    success, initial_path = api.capture_image("slide_01_initial")
    obj_pos, target_pos, gripper_pos = get_positions(api)
    
    print(f"📍 Object: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"🎯 Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    print(f"🤖 Gripper: [{gripper_pos[0]:.3f}, {gripper_pos[1]:.3f}, {gripper_pos[2]:.3f}]")
    
    # Calculate initial distance
    initial_distance = calculate_distance(obj_pos[:2], target_pos[:2])
    print(f"📏 Initial distance to target: {initial_distance:.3f}m = {initial_distance*1000:.1f}mm")
    
    # Step 2: Prepare gripper for optimal contact
    print("\n🤏 Step 2: Preparing gripper for contact")
    api.grip(False)  # Close for maximum contact surface
    print("✅ Gripper closed for solid contact")
    
    # Step 3: Setup for iterative pushing
    safe_height = 0.55
    print(f"\n⚡ Step 3: Moving to safe height ({safe_height}m)")
    api.move_to([gripper_pos[0], gripper_pos[1], safe_height], maintain_grip=True)
    
    # Capture setup state
    success, setup_path = api.capture_image("slide_02_setup")
    print(f"📸 Setup captured: {setup_path}")
    
    # Step 4: Execute single precision push
    print("\n🚀 Step 4: Executing single precision push")
    task_success, final_distance = single_precision_push(api, obj_pos, target_pos, safe_height)
    
    # Step 5: Final documentation
    print("\n📸 Step 5: Final documentation")
    success, final_path = api.capture_image("slide_03_final")
    print(f"📸 Final result: {final_path}")
    
    # Step 6: Retract gripper
    print("\n⬆️  Step 6: Retracting gripper")
    api.lift(0.15)
    
    # Final assessment
    improvement = initial_distance - final_distance
    improvement_percent = (improvement / initial_distance) * 100 if initial_distance > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS:")
    print(f"📏 Initial distance: {initial_distance*1000:.1f}mm")
    print(f"📏 Final accuracy: {final_distance*1000:.1f}mm")
    print(f"📈 Improvement: {improvement*1000:.1f}mm ({improvement_percent:.1f}%)")
    
    if task_success:
        if final_distance <= 0.03:
            print(f"🎉 PRECISION ACHIEVED! ({final_distance*1000:.1f}mm ≤ 30mm)")
        else:
            print(f"✅ TASK SUCCESSFUL! ({final_distance*1000:.1f}mm ≤ 50mm)")
    else:
        print(f"⚠️  Close attempt ({final_distance*1000:.1f}mm > 50mm)")
    
    print(f"{'='*60}")
    
    return task_success

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/slide.py <session-id>")
        sys.exit(1)
    
    session_id = sys.argv[1]
    success = precision_sliding_task(session_id)
    sys.exit(0 if success else 1)