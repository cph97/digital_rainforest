# Rainforest Organism - GPU Audiovisual System

A real-time GPU-first audiovisual system that transforms point cloud data into an explorable "rainforest organism" environment with dynamic generative soundscape and audio-visual feedback loops.

## Features

- **GPU-Driven Rendering**: ModernGL-based point cloud visualization targeting 60fps with 500k+ points
- **Temporal Databending Effects**: Vertical smear, scanline drift, chromatic aberration, and glitch tears
- **Generative Audio Engine**: 
  - Granular synthesis from audio samples
  - FM synthesis for electronic drones
  - Percussive synthesis for rhythmic elements
  - Schroeder reverb processing
- **Point Cloud Sonification**: Maps spatial statistics to synthesis parameters
- **Audio↔Visual Feedback Loop**: Bidirectional reactive system
- **Real-time Audio Analysis**: RMS, onset detection, spectral centroid, frequency bands

## Requirements

- Python 3.8+
- Windows (tested), Linux/macOS should work
- OpenGL 3.3+ capable GPU
- Audio output device

## Installation

### Step 1: Create Virtual Environment

```bash
python -m venv .venv
```

### Step 2: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Setup

### Point Cloud Data

Place your point cloud file in the `./pointclouds/` folder with the name `rainforest_0`:

- **Supported formats**: `.ply`, `.xyz`, `.npy`
- **Auto-detection**: The loader will automatically detect which format exists
- **Format specifications**:
  - `.ply`: ASCII PLY with positions (x,y,z) and optional RGB colors
  - `.xyz`: Space-separated x y z [r g b] per line
  - `.npy`: NumPy array Nx3 (positions) or Nx6 (positions + colors)

**If colors are not present**, the system will generate procedural colors based on height and noise.

**Example folder structure:**
```
./pointclouds/
  └── rainforest_0.ply  (or .xyz or .npy)
```

### Audio Samples (Optional)

Place audio samples in the `./audio/` folder:

- **Supported formats**: `.wav`, `.ogg`, `.mp3`
- **Usage**: Samples are used for granular synthesis and feature extraction
- **If no samples**: System runs with pure generative synthesis

**Example folder structure:**
```
./audio/
  ├── ambient1.wav
  ├── texture2.ogg
  └── drone3.mp3
```

## Running

```bash
python main.py
```

The system will:
1. Auto-detect and load the point cloud
2. Load audio samples (if available)
3. Initialize GPU rendering pipeline
4. Start real-time audio engine
5. Open visualization window

## Controls

### Camera Navigation
- **W/A/S/D**: Move forward/left/backward/right
- **Q/E**: Move down/up
- **Mouse**: Look around (click window to capture mouse)
- **Scroll**: Adjust movement speed

### Effects & Settings
- **Space**: Toggle pause/resume simulation
- **G**: Toggle glitch/smear effect on/off
- **1/2**: Decrease/increase glitch intensity
- **R**: Reset camera to initial position
- **P**: Save screenshot (PNG)
- **H**: Toggle HUD display
- **ESC**: Exit application

## Configuration

Edit the `CONFIG` dictionary in `main.py` to customize:

```python
CONFIG = {
    'point_size': 3.0,              # Base point size
    'max_points': 500000,           # Maximum points to load
    'temporal_feedback': 0.85,      # Temporal accumulation (0-1)
    'smear_strength': 0.3,          # Vertical smear intensity
    'audio_gain': 0.5,              # Master audio volume
    'glitch_intensity': 1.0,        # Initial glitch strength
    # ... more settings
}
```

## Audio-Visual Feedback Mappings

### Audio → Visual
- **RMS** → Point size, brightness
- **Onset** → Point pulsation, smear intensity
- **Spectral Centroid** → Fog density, chroma offset
- **Band Energies** → Scanline drift, tear probability

### Visual → Audio
- **Camera Height** → FM drone pitch region
- **Camera Speed** → Reverb amount
- **Point Density** → Granular grain density
- **Glitch Energy** → Percussion probability, distortion

## Architecture

```
main.py                 # Entry point, window management, event handling
renderer.py             # ModernGL renderer, camera, post-processing
pointcloud_loader.py    # Auto-detect and load .ply/.xyz/.npy
audio_engine.py         # Real-time audio synthesis and sonification
analysis.py             # Audio feature extraction (RMS, onset, bands)
shaders/
  ├── points.vert       # Point cloud vertex shader
  ├── points.frag       # Point rendering with fog
  ├── post.vert         # Fullscreen quad vertex shader
  └── post.frag         # Temporal smear & databending effects
```

## Performance Tips

- **Large point clouds**: Adjust `max_points` in config to downsample
- **Low FPS**: Reduce `point_size` or enable resolution scaling
- **Audio latency**: Adjust `audio_block_size` (smaller = lower latency, higher CPU)
- **Glitch performance**: Lower `temporal_feedback` for less GPU load

## Troubleshooting

### "Point cloud not found"
- Ensure file is in `./pointclouds/` folder
- Check filename is exactly `rainforest_0` (case-sensitive on Linux/macOS)
- Verify file extension is `.ply`, `.xyz`, or `.npy`

### "Failed to start audio engine"
- Check audio device is available and not in use
- System will continue in visual-only mode
- Try different `audio_block_size` values

### Low FPS
- Reduce `max_points` to downsample point cloud
- Lower `point_size` value
- Disable glitch effect temporarily with 'G' key

### No audio output
- Verify audio samples are in `./audio/` folder
- Check system audio settings
- Audio engine will still generate synthesis without samples

## Technical Details

### GPU Pipeline
1. **Point Rendering**: Single draw call, GPU-driven vertex animation
2. **Ping-Pong FBO**: Temporal accumulation for feedback effects
3. **Post-Processing**: Databending-style vertical smear with audio reactivity

### Audio Pipeline
1. **Synthesis**: Granular + FM + Percussive layers
2. **Processing**: Reverb → Soft clipping → Normalization
3. **Analysis**: Real-time feature extraction with smoothing
4. **Feedback**: Visual parameters modulate synthesis

### Shaders
- **Vertex**: Per-point wind animation, audio-driven pulsation
- **Fragment**: Circular point splats, depth-based fog
- **Post**: Column-based vertical smear, chromatic aberration, scanline drift

## Credits

Built with:
- ModernGL - Modern OpenGL wrapper
- librosa - Audio analysis
- sounddevice - Real-time audio I/O
- NumPy, SciPy - Numerical processing

## License

This project is provided as-is for creative coding and audiovisual exploration.
