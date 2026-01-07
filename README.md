# GestureForge

<p align="center">
  <img src="logo.png" width="200" alt="GestureForge Logo">
</p>

**Forge 3D creations with your hands.** Hand-tracking voxel painter with AI 3D Genesis — draw in AR, generate solid 3D shapes with gestures.

## Features

- 🎨 **Hand Tracking**: Draw voxels with pinch gestures
- 🌐 **Infinite Canvas**: Pan/zoom with left hand
- 🧊 **AI 3D Genesis**: Transform drawings into solid 3D shapes
- 🔄 **Grab & Spin**: Realistic physics-based rotation
- 📹 **Recording Mode**: Export demos as MP4

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

## Controls

| Hand | Gesture | Action |
|------|---------|--------|
| Right | Index Pinch | Draw |
| Right | Middle Pinch | Erase |
| Left | Index Pinch + Drag | Pan / Rotate 3D |
| Left | Pinky Pinch + Drag | Zoom |
| Left | Ring Pinch | Cycle colors |
| Left | Wave | Clear canvas |
| Both | Squeeze together | Generate 3D! |

| Key | Action |
|-----|--------|
| G | Generate 3D |
| C | Clear Genesis / Reset |
| R | Toggle Recording |
| S / L | Save / Load |

## AI 3D Genesis

1. Draw any shape
2. **Squeeze both hands** (pinch with both)
3. Watch particles morph into solid 3D
4. **Grab & spin** with left hand pinch
5. Release to see momentum/inertia

## Tech Stack

- Python 3.10+
- PyOpenGL (Hardware rendering)
- MediaPipe (Hand tracking)
- Pygame (Window/input)
- NumPy (Math)

## License

MIT
