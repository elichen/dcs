#!/usr/bin/env python3
"""Test script to verify DCS installation and setup."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing DCS imports...")
    
    try:
        from dcs.core.fetch_ik_solver import FetchIKSolver
        print("✅ IK Solver imported")
    except ImportError as e:
        print(f"❌ Failed to import IK Solver: {e}")
        return False
    
    try:
        from dcs.core.fetch_path_planner import FetchPathPlanner
        print("✅ Path Planner imported")
    except ImportError as e:
        print(f"❌ Failed to import Path Planner: {e}")
        return False
    
    try:
        from dcs.core.fetch_claude_controller import FetchClaudeController
        print("✅ Claude Controller imported")
    except ImportError as e:
        print(f"❌ Failed to import Claude Controller: {e}")
        return False
    
    print("\n✅ All core modules imported successfully!")
    return True

def test_demos():
    """Check that demo files exist."""
    print("\nChecking demo files...")
    
    demos = [
        "working_fetch_task.py",
        "enhanced_pickup_demo.py",
        "auto_visual_demo.py"
    ]
    
    demos_dir = os.path.join(os.path.dirname(__file__), "demos")
    
    for demo in demos:
        demo_path = os.path.join(demos_dir, demo)
        if os.path.exists(demo_path):
            print(f"✅ {demo} found")
        else:
            print(f"❌ {demo} not found")
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("Direct Control System (DCS) Setup Test")
    print("=" * 50)
    
    if test_imports():
        test_demos()
        
        print("\n" + "=" * 50)
        print("🎉 DCS is properly set up!")
        print("\nTo run the main demo:")
        print("  cd dcs/demos")
        print("  python working_fetch_task.py")
        print("\nFor documentation:")
        print("  cat dcs/README.md")
        print("  cat dcs/CLAUDE.md  # For Claude Code usage")
    else:
        print("\n❌ Setup incomplete. Please check imports.")

if __name__ == "__main__":
    main()