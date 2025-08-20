#!/usr/bin/env python3
"""
Simple physics calibration - one push per fresh environment to get clean data.
"""

import sys
import json
import math
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

def single_push_test(session_id, push_distance):
    """Test a single push distance with proper contact verification."""
    
    print(f"🧪 Testing {push_distance*1000:.0f}mm push with session {session_id}")
    
    api = FetchAPI.connect(session_id)
    
    # Get initial positions
    obj_pos, target_pos, gripper_pos = get_positions(api)
    print(f"📍 Object: [{obj_pos[0]:.3f}, {obj_pos[1]:.3f}, {obj_pos[2]:.3f}]")
    print(f"🎯 Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Calculate push direction (toward target)
    push_vector = [target_pos[0] - obj_pos[0], target_pos[1] - obj_pos[1]]
    distance_to_target = math.sqrt(push_vector[0]**2 + push_vector[1]**2)
    push_direction = [push_vector[0] / distance_to_target, push_vector[1] / distance_to_target]
    
    print(f"📐 Push direction: [{push_direction[0]:.3f}, {push_direction[1]:.3f}]")
    print(f"📏 Distance to target: {distance_to_target*1000:.1f}mm")
    
    # Prepare gripper and position
    api.grip(False)  # Close for contact
    safe_height = 0.55
    api.move_to([obj_pos[0], obj_pos[1], safe_height], maintain_grip=True)
    
    # Position behind object for push - closer to ensure contact
    behind_distance = 0.06  # 6cm behind for better contact
    push_height = obj_pos[2] + 0.002  # Slightly above object surface
    
    behind_x = obj_pos[0] - push_direction[0] * behind_distance
    behind_y = obj_pos[1] - push_direction[1] * behind_distance
    
    # Calculate push target
    push_end_x = behind_x + push_direction[0] * push_distance
    push_end_y = behind_y + push_direction[1] * push_distance
    
    print(f"📍 Behind position: [{behind_x:.3f}, {behind_y:.3f}, {push_height:.3f}]")
    print(f"🎯 Push target: [{push_end_x:.3f}, {push_end_y:.3f}, {push_height:.3f}]")
    
    # Capture initial state
    api.capture_image(f"calib_initial_{push_distance*1000:.0f}mm")
    initial_obj_pos = obj_pos.copy()
    
    # Execute push sequence
    # 1. Position behind object
    api.move_to([behind_x, behind_y, push_height], maintain_grip=True, max_steps=40)
    
    # 2. Push through object
    success, msg = api.move_to([push_end_x, push_end_y, push_height], maintain_grip=True, max_steps=60)
    print(f"🔄 Push result: {msg}")
    
    # 3. Wait for sliding to stop
    print("⏳ Waiting for sliding to stop...")
    import time
    time.sleep(1.0)  # Wait for sliding to complete
    
    # 4. Capture final state
    api.capture_image(f"calib_final_{push_distance*1000:.0f}mm")
    
    # Measure results after sliding has stopped
    final_obj_pos, _, _ = get_positions(api)
    actual_movement = calculate_distance(initial_obj_pos[:2], final_obj_pos[:2])
    efficiency = actual_movement / push_distance if push_distance > 0 else 0
    
    print(f"📊 Results:")
    print(f"   Initial object: [{initial_obj_pos[0]:.3f}, {initial_obj_pos[1]:.3f}]")
    print(f"   Final object: [{final_obj_pos[0]:.3f}, {final_obj_pos[1]:.3f}]")
    print(f"   Movement: {actual_movement*1000:.1f}mm")
    print(f"   Efficiency: {efficiency:.3f}")
    
    # Check if push was toward target
    movement_vector = [final_obj_pos[0] - initial_obj_pos[0], final_obj_pos[1] - initial_obj_pos[1]]
    movement_distance = math.sqrt(movement_vector[0]**2 + movement_vector[1]**2)
    
    if movement_distance > 0.005:  # 5mm minimum movement
        movement_direction = [movement_vector[0] / movement_distance, movement_vector[1] / movement_distance]
        # Dot product to check alignment with intended push direction
        alignment = push_direction[0] * movement_direction[0] + push_direction[1] * movement_direction[1]
        print(f"   Direction alignment: {alignment:.3f} (1.0 = perfect)")
        
        valid_push = alignment > 0.7 and actual_movement > 0.01  # Good alignment and meaningful movement
    else:
        valid_push = False
        alignment = 0.0
    
    print(f"   Valid push: {'✅' if valid_push else '❌'}")
    
    return {
        'push_distance': push_distance,
        'actual_movement': actual_movement,
        'efficiency': efficiency,
        'alignment': alignment,
        'valid': valid_push,
        'initial_pos': initial_obj_pos,
        'final_pos': final_obj_pos
    }

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/calibrate_simple.py <session-id> <push-distance-mm>")
        print("Example: python scripts/calibrate_simple.py abc12345 80")
        sys.exit(1)
    
    session_id = sys.argv[1]
    push_distance_mm = float(sys.argv[2])
    push_distance = push_distance_mm / 1000.0  # Convert to meters
    
    result = single_push_test(session_id, push_distance)
    
    print(f"\n{'='*50}")
    if result['valid']:
        print(f"✅ Valid calibration data collected!")
        print(f"📈 {push_distance_mm:.0f}mm push → {result['actual_movement']*1000:.1f}mm movement")
        print(f"🎯 Efficiency: {result['efficiency']:.3f}")
    else:
        print(f"❌ Invalid push - poor contact or wrong direction")
    print(f"{'='*50}")