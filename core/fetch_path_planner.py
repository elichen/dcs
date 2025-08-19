#!/usr/bin/env python3
"""
Path planning algorithms for the Fetch robot.

This module provides collision-free path planning where Claude can
calculate waypoints to move the end-effector from one point to another.
"""

import numpy as np
from typing import List, Tuple, Optional
import mujoco


class FetchPathPlanner:
    """
    Path planner for Fetch robot end-effector movements.
    
    Implements RRT-like algorithms and simple geometric path planning
    with collision avoidance.
    """
    
    def __init__(self, model, data, ik_solver):
        """
        Initialize path planner.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data  
            ik_solver: FetchIKSolver instance
        """
        self.model = model
        self.data = data
        self.ik_solver = ik_solver
        
        # Workspace bounds (conservative estimates for Fetch)
        self.workspace_bounds = {
            'x': [0.2, 1.2],  # 0.2m to 1.2m from base
            'y': [-0.6, 0.6], # -0.6m to 0.6m left/right
            'z': [0.0, 1.5]   # 0.0m to 1.5m height
        }
        
        # Get obstacle information
        self.table_body_id = self._find_body_id('table')
        self.object_body_id = self._find_body_id('object')
        
        print(f"✓ FetchPathPlanner initialized")
        print(f"  - Workspace: x[{self.workspace_bounds['x'][0]}, {self.workspace_bounds['x'][1]}]")
        print(f"  - Workspace: y[{self.workspace_bounds['y'][0]}, {self.workspace_bounds['y'][1]}]") 
        print(f"  - Workspace: z[{self.workspace_bounds['z'][0]}, {self.workspace_bounds['z'][1]}]")
    
    def _find_body_id(self, body_name):
        """Find body ID by name, return -1 if not found."""
        try:
            return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        except:
            return -1
    
    def is_position_valid(self, position):
        """Check if a position is within the valid workspace."""
        x, y, z = position
        
        # Check workspace bounds
        if not (self.workspace_bounds['x'][0] <= x <= self.workspace_bounds['x'][1]):
            return False
        if not (self.workspace_bounds['y'][0] <= y <= self.workspace_bounds['y'][1]):
            return False
        if not (self.workspace_bounds['z'][0] <= z <= self.workspace_bounds['z'][1]):
            return False
        
        return True
    
    def check_collision(self, position, safety_margin=0.05):
        """
        Check if a position would cause collision.
        
        Args:
            position: [x, y, z] position to check
            safety_margin: Safety margin in meters
            
        Returns:
            bool: True if collision-free, False if collision
        """
        # Simple collision checking with known obstacles
        
        # Table collision (if table exists)
        if self.table_body_id != -1:
            # Get table position and size
            table_pos = self.data.xpos[self.table_body_id]
            # Assume table is 1.0m x 0.6m x 0.7m (Fetch default)
            table_size = np.array([0.5, 0.3, 0.35])  # Half-sizes
            
            # Check if position is too close to table
            rel_pos = position - table_pos
            if (abs(rel_pos[0]) < table_size[0] + safety_margin and 
                abs(rel_pos[1]) < table_size[1] + safety_margin and 
                abs(rel_pos[2]) < table_size[2] + safety_margin):
                return False  # Collision with table
        
        # Ground collision
        if position[2] < safety_margin:
            return False
        
        # Base collision (robot can't reach behind itself)
        if position[0] < 0.0:
            return False
        
        return True
    
    def interpolate_path(self, start_pos, end_pos, num_waypoints=10):
        """
        Create a straight-line interpolated path between two positions.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Ending position [x, y, z]
            num_waypoints: Number of intermediate waypoints
            
        Returns:
            List of waypoints including start and end
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        # Linear interpolation
        waypoints = []
        for i in range(num_waypoints + 2):  # +2 for start and end
            t = i / (num_waypoints + 1)
            waypoint = start_pos + t * (end_pos - start_pos)
            waypoints.append(waypoint.tolist())
        
        return waypoints
    
    def plan_straight_line(self, start_pos, end_pos, check_collisions=True):
        """
        Plan a straight-line path with collision checking.
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Target position [x, y, z]
            check_collisions: Whether to check for collisions
            
        Returns:
            tuple: (success, waypoints, message)
        """
        print(f"📐 Planning straight-line path:")
        print(f"  Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        print(f"  End:   [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
        
        # Check if start and end positions are valid
        if not self.is_position_valid(start_pos):
            return False, [], "Start position outside workspace"
        
        if not self.is_position_valid(end_pos):
            return False, [], "End position outside workspace"
        
        # Generate waypoints
        waypoints = self.interpolate_path(start_pos, end_pos, num_waypoints=5)
        
        # Check collisions if requested
        if check_collisions:
            for i, waypoint in enumerate(waypoints):
                if not self.check_collision(waypoint):
                    print(f"⚠ Collision detected at waypoint {i}: {waypoint}")
                    return False, [], f"Collision at waypoint {i}"
        
        print(f"✓ Straight-line path planned with {len(waypoints)} waypoints")
        return True, waypoints, "Success"
    
    def plan_arc_path(self, start_pos, end_pos, arc_height=0.1):
        """
        Plan an arc-shaped path (useful for pick-and-place).
        
        Args:
            start_pos: Starting position [x, y, z]
            end_pos: Target position [x, y, z] 
            arc_height: Additional height for the arc peak
            
        Returns:
            tuple: (success, waypoints, message)
        """
        start_pos = np.array(start_pos)
        end_pos = np.array(end_pos)
        
        print(f"🌉 Planning arc path with {arc_height:.3f}m height:")
        print(f"  Start: [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
        print(f"  End:   [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
        
        # Check workspace bounds
        if not (self.is_position_valid(start_pos) and self.is_position_valid(end_pos)):
            return False, [], "Start or end position outside workspace"
        
        # Calculate arc waypoints
        waypoints = []
        num_points = 8
        
        for i in range(num_points):
            t = i / (num_points - 1)  # 0 to 1
            
            # Linear interpolation for x and y
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            
            # Parabolic arc for z
            base_z = start_pos[2] + t * (end_pos[2] - start_pos[2])
            arc_offset = arc_height * 4 * t * (1 - t)  # Parabolic curve
            z = base_z + arc_offset
            
            waypoint = [x, y, z]
            
            # Check validity and collisions
            if not self.is_position_valid(waypoint):
                print(f"⚠ Arc waypoint {i} outside workspace: {waypoint}")
                # Try with lower arc
                z = base_z + arc_offset * 0.5
                waypoint = [x, y, z]
                
                if not self.is_position_valid(waypoint):
                    return False, [], f"Cannot create valid arc path"
            
            if not self.check_collision(waypoint):
                print(f"⚠ Collision detected at arc waypoint {i}")
                # Try with higher arc
                z = base_z + arc_offset + 0.05
                waypoint = [x, y, z]
                
                if not self.check_collision(waypoint):
                    print(f"  Adjusted waypoint {i} to avoid collision")
            
            waypoints.append(waypoint)
        
        print(f"✓ Arc path planned with {len(waypoints)} waypoints")
        return True, waypoints, "Success"
    
    def plan_approach_path(self, target_pos, approach_direction=[0, 0, 1], approach_distance=0.15):
        """
        Plan an approach path to a target (useful for grasping).
        
        Args:
            target_pos: Target position [x, y, z]
            approach_direction: Direction to approach from (normalized)
            approach_distance: Distance to approach from
            
        Returns:
            tuple: (success, waypoints, message)
        """
        target_pos = np.array(target_pos)
        approach_dir = np.array(approach_direction)
        approach_dir = approach_dir / np.linalg.norm(approach_dir)  # Normalize
        
        # Calculate approach position
        approach_pos = target_pos + approach_dir * approach_distance
        
        print(f"🎯 Planning approach path:")
        print(f"  Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"  Approach: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}, {approach_pos[2]:.3f}]")
        
        # Get current position
        current_pos, _ = self.ik_solver.get_current_ee_pose()
        
        # Plan path: current -> approach -> target
        waypoints = []
        
        # First, move to approach position
        success1, path1, msg1 = self.plan_arc_path(current_pos, approach_pos, arc_height=0.1)
        if not success1:
            return False, [], f"Failed to plan path to approach position: {msg1}"
        
        waypoints.extend(path1[:-1])  # Exclude last point to avoid duplication
        
        # Then, move from approach to target
        success2, path2, msg2 = self.plan_straight_line(approach_pos, target_pos)
        if not success2:
            return False, [], f"Failed to plan approach to target: {msg2}"
        
        waypoints.extend(path2)
        
        print(f"✓ Approach path planned with {len(waypoints)} total waypoints")
        return True, waypoints, "Success"
    
    def visualize_path(self, waypoints):
        """Print path information for debugging."""
        print(f"\n📍 Path visualization ({len(waypoints)} waypoints):")
        for i, wp in enumerate(waypoints):
            print(f"  {i:2d}: [{wp[0]:6.3f}, {wp[1]:6.3f}, {wp[2]:6.3f}]")
        print()


def test_path_planner():
    """Test the path planner."""
    import gymnasium as gym
    import gymnasium_robotics
    from dcs.core.fetch_ik_solver import FetchIKSolver
    
    print("Testing FetchPathPlanner...")
    
    # Create environment
    env = gym.make("FetchPickAndPlace-v4")
    obs, info = env.reset()
    
    # Get MuJoCo model and data
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Create IK solver and path planner
    ik_solver = FetchIKSolver(model, data)
    path_planner = FetchPathPlanner(model, data, ik_solver)
    
    # Get current position
    current_pos, _ = ik_solver.get_current_ee_pose()
    
    # Test straight-line planning
    target_pos = current_pos + np.array([0.2, 0.1, 0.1])
    success, waypoints, msg = path_planner.plan_straight_line(current_pos, target_pos)
    
    if success:
        print(f"✓ Straight path: {msg}")
        path_planner.visualize_path(waypoints)
    else:
        print(f"✗ Straight path failed: {msg}")
    
    # Test arc planning
    success, waypoints, msg = path_planner.plan_arc_path(current_pos, target_pos)
    
    if success:
        print(f"✓ Arc path: {msg}")
        path_planner.visualize_path(waypoints)
    else:
        print(f"✗ Arc path failed: {msg}")
    
    env.close()


if __name__ == "__main__":
    test_path_planner()