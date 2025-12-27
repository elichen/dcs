# Claude Code Guide for Direct Control System (DCS)

## Overview

The Direct Control System (DCS) provides Unix-style CLI tools for controlling a Fetch robot arm in MuJoCo simulation. It uses socket-based IPC for low-latency communication and supports multiple Gymnasium Robotics environments.

## Task Performance

Tested precision across Fetch environments:

| Task | Precision | Strategy |
|------|-----------|----------|
| Pick and Place | 2mm | Lift-based grasping |
| Push | 9mm | Vector-calculated positioning |
| Reach | 4mm | Direct movement |
| Slide | ~100mm on 530mm | Momentum-based push |

## Quick Start

```bash
# Start session in background (blocking call)
bin/env start --gif --gif-file demo.gif &
sleep 2

# Get session ID
bin/env list

# Execute task
bin/pick $SESSION $(bin/object $SESSION | jq -r '.x,.y,.z')
bin/place $SESSION $(bin/target $SESSION | jq -r '.x,.y,.z')

# Stop and save GIF
bin/env stop $SESSION
```

## Critical Lessons Learned

### Push and Slide Tasks Require Approach From Above

When pushing objects, moving directly to a "behind" position will collide with the object and push it the wrong direction. The correct approach:

```python
# 1. Calculate push vector from object toward target
dx = target[0] - obj[0]
dy = target[1] - obj[1]
dist = math.sqrt(dx**2 + dy**2)
ux, uy = dx/dist, dy/dist  # Unit vector

# 2. Calculate positions
behind_dist = 0.08  # 8cm behind object
behind_x = obj[0] - ux * behind_dist
behind_y = obj[1] - uy * behind_dist

push_end_x = behind_x + ux * (dist + 0.05)  # Push through target
push_end_y = behind_y + uy * (dist + 0.05)

# 3. Execute: lift -> position behind at height -> lower -> push
safe_height = 0.55
push_height = obj[2] + 0.002

api.grip(False)  # Close for contact
api.move_to([behind_x, behind_y, safe_height], maintain_grip=True)
api.move_to([behind_x, behind_y, push_height], maintain_grip=True)
api.move_to([push_end_x, push_end_y, push_height], maintain_grip=True, velocity_scale=1.0)
```

### Slide Task Physics

For slide tasks where the target is beyond robot reach:
- Use `velocity_scale=1.5` for hard push
- Wait for sliding to stop before measuring result
- The `scripts/slide.py` has calibrated physics calculations

### Gripper State Detection

The gripper joint threshold for detecting open/closed state is 0.035:
- Open gripper: ~0.05 joint position
- Closed with object: ~0.024 (can't fully close)
- Closed empty: ~0.0

Threshold must be >0.024 to correctly detect "closed with object" as closed.

## CLI Tools Reference

### Session Management
```bash
bin/env start [--env ENV] [--gif] [--gif-file FILE]  # Start (blocking)
bin/env list                                          # List sessions
bin/env stop <id>                                     # Stop session
```

### Robot Control
```bash
bin/move <id> <x> <y> <z>      # Move gripper to position
bin/grip <id> open|close       # Control gripper
bin/lift <id> <height>         # Lift vertically
bin/approach <id> <x> <y> <z>  # Move above position
bin/push <id> <x> <y> <z> [--power=soft|medium|hard]
```

### State Queries
```bash
bin/status <id>   # Full state JSON
bin/object <id>   # Object position
bin/target <id>   # Target position
```

### Composite Actions
```bash
bin/pick <id> <x> <y> <z>   # Pick sequence (open, approach, descend, close, lift)
bin/place <id> <x> <y> <z>  # Place sequence (transport, lower, open, retract)
```

## Python API

```python
from lib.fetch_api import FetchAPI

api = FetchAPI.connect(session_id)

# Query state
state = api.get_state()
obj = state['object_position']
target = state['target_position']

# Movement
api.move_to([x, y, z], maintain_grip=True, velocity_scale=1.0)
api.grip(True)   # Open
api.grip(False)  # Close
api.lift(0.15)

# Physics stepping (for waiting on sliding)
api.step_physics()
```

## File Structure

```
dcs/
├── bin/                    # CLI tools
│   ├── env                 # Session management
│   ├── move, grip, lift    # Basic control
│   ├── pick, place, push   # Task sequences
│   └── object, target      # State queries
├── lib/
│   ├── fetch_api.py        # Python API
│   ├── fetch_session.py    # Session management
│   └── direct_executor.py  # Command execution
├── scripts/
│   ├── slide.py            # Physics-aware slide task
│   └── calibrate.py        # Physics calibration
├── recordings/             # GIF outputs
└── CLAUDE.md               # This file
```

## Architecture

- **Socket IPC**: ~50us latency via Unix domain sockets
- **Thread safety**: MuJoCo/OpenGL on main thread, commands queued
- **Real-time rendering**: 50fps visualization during movement
- **GIF recording**: `--gif` flag captures all frames

## Usage Notes

- `bin/env start` is blocking - run in background with `&`
- Use `sleep 2` after starting to let session initialize
- All positions are in meters, table height is ~0.425
- Object z-coordinate indicates if on table (~0.41-0.43) or fallen (~0.02)
- For GIF recording, stop session cleanly to save the file
