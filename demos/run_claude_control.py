#!/usr/bin/env python3
"""
Main entry point for Claude's Direct Control System.

This script provides the main interface for Claude to demonstrate
direct control over the Fetch robot using explicit mathematics.
"""

import argparse
import sys
import os
import time

# Add the mujoco directory to Python path
sys.path.append(os.path.dirname(__file__))

def run_interactive_demo():
    """Run the interactive demonstration."""
    from dcs.core.fetch_interactive_demo import main as demo_main
    demo_main()

def run_tests():
    """Run all system tests."""
    print("🧪 Running Claude Control System Tests")
    print("=" * 50)
    
    # Test 1: IK Solver
    print("\n1️⃣ Testing IK Solver...")
    try:
        from dcs.core.fetch_ik_solver import test_ik_solver
        test_ik_solver()
        print("✅ IK Solver test passed")
    except Exception as e:
        print(f"❌ IK Solver test failed: {e}")
        return False
    
    # Test 2: Path Planner
    print("\n2️⃣ Testing Path Planner...")
    try:
        from dcs.core.fetch_path_planner import test_path_planner
        test_path_planner()
        print("✅ Path Planner test passed")
    except Exception as e:
        print(f"❌ Path Planner test failed: {e}")
        return False
    
    # Test 3: Claude Controller
    print("\n3️⃣ Testing Claude Controller...")
    try:
        from dcs.core.fetch_claude_controller import test_claude_controller
        test_claude_controller()
        print("✅ Claude Controller test passed")
    except Exception as e:
        print(f"❌ Claude Controller test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! System ready for demonstration.")
    return True

def create_example_command(action_type="demonstrate"):
    """Create an example command file for Claude to use."""
    import json
    
    command_templates = {
        "demonstrate": {
            "timestamp": time.time(),
            "action": "demonstrate",
            "params": {},
            "message": "Run a complete pick-and-place demonstration"
        },
        
        "move": {
            "timestamp": time.time(),
            "action": "move_to",
            "params": {
                "position": [0.5, 0.2, 0.4],
                "path_type": "arc"
            },
            "message": "Move end-effector to [0.5, 0.2, 0.4] using arc path"
        },
        
        "pick": {
            "timestamp": time.time(),
            "action": "pick_object",
            "params": {
                "position": [0.6, 0.1, 0.05]
            },
            "message": "Pick up object at [0.6, 0.1, 0.05]"
        },
        
        "place": {
            "timestamp": time.time(),
            "action": "place_object",
            "params": {
                "position": [0.4, -0.1, 0.05]
            },
            "message": "Place object at [0.4, -0.1, 0.05]"
        },
        
        "gripper_open": {
            "timestamp": time.time(),
            "action": "control_gripper",
            "params": {
                "open": True
            },
            "message": "Open the gripper"
        },
        
        "gripper_close": {
            "timestamp": time.time(),
            "action": "control_gripper",
            "params": {
                "open": False
            },
            "message": "Close the gripper"
        },
        
        "reset": {
            "timestamp": time.time(),
            "action": "reset",
            "params": {},
            "message": "Reset the environment and controller"
        }
    }
    
    if action_type not in command_templates:
        print(f"❌ Unknown action type: {action_type}")
        print(f"Available actions: {list(command_templates.keys())}")
        return False
    
    command = command_templates[action_type]
    filename = "claude_commands.json"
    
    with open(filename, 'w') as f:
        json.dump(command, f, indent=2)
    
    print(f"✅ Created command file: {filename}")
    print(f"   Action: {command['action']}")
    print(f"   Message: {command['message']}")
    
    return True

def show_system_status():
    """Show the current system status."""
    print("🔍 Claude Control System Status")
    print("=" * 40)
    
    # Check file existence
    files_to_check = [
        "fetch_ik_solver.py",
        "fetch_path_planner.py", 
        "fetch_claude_controller.py",
        "fetch_interactive_demo.py"
    ]
    
    all_files_exist = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            all_files_exist = False
    
    if all_files_exist:
        print("\n🎯 System Status: Ready for demonstration")
        
        # Check if communication files exist
        comm_files = ["claude_commands.json", "robot_state.json"]
        for file in comm_files:
            if os.path.exists(file):
                print(f"📄 {file} exists")
            else:
                print(f"📄 {file} not found (will be created)")
    else:
        print("\n❌ System Status: Missing files - run setup first")
    
    print(f"\n📍 Working directory: {os.getcwd()}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Claude's Direct Control System for Fetch Robot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_claude_control.py demo              # Run interactive demo
    python run_claude_control.py test              # Run system tests
    python run_claude_control.py command move      # Create move command
    python run_claude_control.py command pick      # Create pick command
    python run_claude_control.py status            # Show system status

Available command types:
    demonstrate, move, pick, place, gripper_open, gripper_close, reset
        """
    )
    
    parser.add_argument('mode', choices=['demo', 'test', 'command', 'status'],
                       help='Operation mode')
    parser.add_argument('action', nargs='?',
                       help='Action type (for command mode)')
    
    args = parser.parse_args()
    
    # Header
    print("🤖 Claude's Direct Fetch Robot Control System")
    print("=" * 50)
    
    if args.mode == 'demo':
        print("🎭 Starting interactive demonstration...")
        run_interactive_demo()
    
    elif args.mode == 'test':
        print("🧪 Running system tests...")
        success = run_tests()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'command':
        if not args.action:
            print("❌ Command mode requires an action type")
            parser.print_help()
            sys.exit(1)
        
        print(f"📝 Creating command: {args.action}")
        success = create_example_command(args.action)
        sys.exit(0 if success else 1)
    
    elif args.mode == 'status':
        show_system_status()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()