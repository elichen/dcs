#!/usr/bin/env python3
"""
Inverse Kinematics solver for the Fetch robot using MuJoCo.

This module provides direct IK solving capabilities where Claude can
calculate joint angles given desired end-effector poses.
"""

import numpy as np
import mujoco


class FetchIKSolver:
    """
    Direct inverse kinematics solver for Fetch robot.
    
    Uses Jacobian pseudoinverse method with numerical optimization
    to solve for joint angles given end-effector targets.
    """
    
    def __init__(self, model, data):
        """
        Initialize the IK solver with MuJoCo model and data.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data
        
        # Find important body and joint indices
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0:gripper_link')
        self.base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'robot0:base_link')
        
        # Fetch arm joint names (7-DOF arm)
        self.arm_joint_names = [
            'robot0:shoulder_pan_joint',
            'robot0:shoulder_lift_joint', 
            'robot0:upperarm_roll_joint',
            'robot0:elbow_flex_joint',
            'robot0:forearm_roll_joint',
            'robot0:wrist_flex_joint',
            'robot0:wrist_roll_joint'
        ]
        
        # Get joint indices
        self.arm_joint_ids = []
        for joint_name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in model")
            self.arm_joint_ids.append(joint_id)
        
        # Joint limits for the arm
        self.joint_limits_low = []
        self.joint_limits_high = []
        for joint_id in self.arm_joint_ids:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.joint_limits_low.append(self.model.jnt_range[joint_id, 0])
            self.joint_limits_high.append(self.model.jnt_range[joint_id, 1])
        
        self.joint_limits_low = np.array(self.joint_limits_low)
        self.joint_limits_high = np.array(self.joint_limits_high)
        
        print(f"✓ FetchIKSolver initialized")
        print(f"  - End-effector body ID: {self.ee_body_id}")
        print(f"  - Arm joints: {len(self.arm_joint_ids)}")
        print(f"  - Joint limits: [{self.joint_limits_low.min():.2f}, {self.joint_limits_high.max():.2f}]")
    
    def forward_kinematics(self, joint_angles=None):
        """
        Calculate end-effector position and orientation using forward kinematics.
        
        Args:
            joint_angles: Array of joint angles. If None, uses current joint positions.
            
        Returns:
            tuple: (position, orientation) where position is [x,y,z] and orientation is quaternion [w,x,y,z]
        """
        if joint_angles is not None:
            # Set joint angles
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = joint_angles[i]
        
        # Update kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Get end-effector pose
        ee_pos = self.data.xpos[self.ee_body_id].copy()
        ee_quat = self.data.xquat[self.ee_body_id].copy()  # [w, x, y, z]
        
        return ee_pos, ee_quat
    
    def calculate_jacobian(self, joint_angles=None):
        """
        Calculate the Jacobian matrix for the end-effector.
        
        Args:
            joint_angles: Joint angles to calculate Jacobian at
            
        Returns:
            np.array: 6x7 Jacobian matrix (3 position + 3 orientation)
        """
        if joint_angles is not None:
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                self.data.qpos[qpos_addr] = joint_angles[i]
            mujoco.mj_forward(self.model, self.data)
        
        # Calculate Jacobians
        jacp = np.zeros((3, self.model.nv))  # Position Jacobian
        jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
        
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.xpos[self.ee_body_id], self.ee_body_id)
        
        # Extract only the columns for our arm joints
        arm_jacp = np.zeros((3, len(self.arm_joint_ids)))
        arm_jacr = np.zeros((3, len(self.arm_joint_ids))) 
        
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_addr = self.model.jnt_dofadr[joint_id]
            arm_jacp[:, i] = jacp[:, dof_addr]
            arm_jacr[:, i] = jacr[:, dof_addr]
        
        # Combine position and rotation Jacobians
        jacobian = np.vstack([arm_jacp, arm_jacr])
        
        return jacobian
    
    def solve_ik(self, target_position, target_orientation=None, initial_guess=None, 
                 max_iterations=100, tolerance=1e-3, step_size=0.1):
        """
        Solve inverse kinematics using Jacobian pseudoinverse method.
        
        Args:
            target_position: [x, y, z] target position for end-effector
            target_orientation: [w, x, y, z] target quaternion (optional)
            initial_guess: Initial joint angles (optional)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            step_size: Step size for updates
            
        Returns:
            tuple: (success, joint_angles, final_error)
        """
        # Use current joint positions as initial guess if not provided
        if initial_guess is None:
            initial_guess = np.zeros(len(self.arm_joint_ids))
            for i, joint_id in enumerate(self.arm_joint_ids):
                qpos_addr = self.model.jnt_qposadr[joint_id]
                initial_guess[i] = self.data.qpos[qpos_addr]
        
        current_joints = initial_guess.copy()
        target_position = np.array(target_position)
        
        print(f"🎯 Solving IK for target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
        
        for iteration in range(max_iterations):
            # Calculate current end-effector position
            current_pos, current_quat = self.forward_kinematics(current_joints)
            
            # Calculate position error
            pos_error = target_position - current_pos
            pos_error_norm = np.linalg.norm(pos_error)
            
            # Check convergence
            if pos_error_norm < tolerance:
                print(f"✓ IK converged in {iteration} iterations, error: {pos_error_norm:.6f}")
                return True, current_joints, pos_error_norm
            
            # Calculate error vector (position only for now)
            if target_orientation is not None:
                # TODO: Add orientation error calculation
                error_vector = pos_error  # For now, just position
            else:
                error_vector = pos_error
            
            # Calculate Jacobian
            jacobian = self.calculate_jacobian(current_joints)
            
            # Use only position part of Jacobian if no orientation target
            if target_orientation is None:
                jacobian = jacobian[:3, :]  # Only position rows
            
            # Compute pseudoinverse
            try:
                jacobian_pinv = np.linalg.pinv(jacobian)
            except np.linalg.LinAlgError:
                print(f"⚠ Jacobian pseudoinverse failed at iteration {iteration}")
                return False, current_joints, pos_error_norm
            
            # Calculate joint update
            delta_joints = jacobian_pinv @ error_vector
            
            # Apply step size
            delta_joints *= step_size
            
            # Update joint angles
            current_joints += delta_joints
            
            # Clamp to joint limits
            current_joints = np.clip(current_joints, self.joint_limits_low, self.joint_limits_high)
            
            # Debug output every 20 iterations
            if iteration % 20 == 0:
                print(f"  Iteration {iteration}: error = {pos_error_norm:.6f}, joints = {current_joints[:3]}")
        
        print(f"⚠ IK failed to converge after {max_iterations} iterations, final error: {pos_error_norm:.6f}")
        return False, current_joints, pos_error_norm
    
    def get_current_ee_pose(self):
        """Get current end-effector position and orientation."""
        return self.forward_kinematics()
    
    def get_current_joint_angles(self):
        """Get current arm joint angles."""
        joint_angles = np.zeros(len(self.arm_joint_ids))
        for i, joint_id in enumerate(self.arm_joint_ids):
            qpos_addr = self.model.jnt_qposadr[joint_id]
            joint_angles[i] = self.data.qpos[qpos_addr]
        return joint_angles
    
    def check_joint_limits(self, joint_angles):
        """Check if joint angles are within limits."""
        return np.all(joint_angles >= self.joint_limits_low) and np.all(joint_angles <= self.joint_limits_high)


def test_ik_solver():
    """Test the IK solver with a simple example."""
    import gymnasium as gym
    import gymnasium_robotics
    
    print("Testing FetchIKSolver...")
    
    # Create environment
    env = gym.make("FetchPickAndPlace-v4")
    obs, info = env.reset()
    
    # Get MuJoCo model and data
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    # Create IK solver
    ik_solver = FetchIKSolver(model, data)
    
    # Get current end-effector position
    current_pos, current_quat = ik_solver.get_current_ee_pose()
    print(f"Current EE position: {current_pos}")
    
    # Test IK by moving slightly forward
    target_pos = current_pos + np.array([0.1, 0, 0])  # 10cm forward
    success, joint_solution, error = ik_solver.solve_ik(target_pos)
    
    if success:
        print(f"✓ IK solution found: {joint_solution}")
        
        # Verify solution
        new_pos, _ = ik_solver.forward_kinematics(joint_solution)
        verification_error = np.linalg.norm(target_pos - new_pos)
        print(f"Verification error: {verification_error:.6f}")
    else:
        print("✗ IK failed")
    
    env.close()


if __name__ == "__main__":
    test_ik_solver()