# Quick Start Guide

## First Time Setup (5 minutes)

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv .venv

# Activate it (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### 2. Generate Example Point Cloud

If you don't have a point cloud file yet, generate an example:

```powershell
python generate_example_pointcloud.py
```

This creates `./pointclouds/rainforest_0.npy` with 100k procedural points.

### 3. (Optional) Add Audio Samples

Place any `.wav`, `.ogg`, or `.mp3` files in the `./audio/` folder. The system works without them (pure synthesis), but samples add texture.

### 4. Run!

```powershell
python main.py
```

## First Run Checklist

✓ **Window opens** - You should see the point cloud  
✓ **Camera works** - Click window, then use WASD + mouse  
✓ **Audio plays** - You should hear generative soundscape  
✓ **Effects work** - Press G to toggle glitch effect  

## Quick Troubleshooting

**"Point cloud not found"**  
→ Run `python generate_example_pointcloud.py` first

**No audio**  
→ Check Windows audio settings, or run visual-only mode (it continues anyway)

**Low FPS**  
→ Edit `main.py`, set `'max_points': 50000` (lower number)

**Window too small/large**  
→ Edit `main.py`, change `'window_width'` and `'window_height'`

## Essential Controls

- **WASD** - Move camera
- **Mouse** - Look (click window first)
- **Scroll** - Speed
- **G** - Toggle glitch
- **Space** - Pause
- **ESC** - Exit

## Using Your Own Point Cloud

Replace the generated file with your own:

1. Name it `rainforest_0.ply` (or `.xyz` or `.npy`)
2. Place in `./pointclouds/` folder
3. Run `python main.py`

The loader auto-detects the format!

## Next Steps

- Read `README.md` for full documentation
- Adjust `CONFIG` in `main.py` for customization
- Explore shader files in `./shaders/` for visual tweaks
- Check audio mappings in `audio_engine.py`

Enjoy exploring your rainforest organism! 🌿
