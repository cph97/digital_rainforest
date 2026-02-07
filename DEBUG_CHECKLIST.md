# Point Cloud Renderer Debug Checklist

## Implemented Fixes

### 1. Data Validation ✓
- Added comprehensive logging of raw PLY data (dtype, ranges, means)
- Validates position data is float32 in reasonable world units
- Validates color data is in 0-1 range
- Logs data before and after axis rotation

### 2. Coordinate Normalization ✓
- **Centering**: Optional subtraction of centroid (config: `center_pointcloud`)
- **Scaling**: Optional normalization to unit sphere (config: `normalize_pointcloud`)
- Preserves original centroid and scale factor for reference
- Recomputes bounding radius after normalization

### 3. Camera Setup ✓
- **Auto near/far planes**: Based on bounding radius
  - `near = max(0.01, radius * 0.01)`
  - `far = radius * 10.0`
- **Auto camera fit**: Positions camera at `radius * 2.5` distance
- **Configurable FOV**: Set via `config['fov']` (default: 60°)
- Diagonal 3/4 view angle for natural perspective

### 4. Depth Testing ✓
- Depth test enabled in OpenGL context
- Depth buffer cleared each frame
- Proper near/far planes prevent z-fighting

### 5. Vertex Shader ✓
- Correct transformation: `gl_Position = mvp * vec4(pos, 1.0)`
- Point size in pixels (not distance-scaled)
- Wind offset is small (max 0.02) and clamped
- Audio modulation is additive, not multiplicative on positions

### 6. Post-Processing Bypass ✓
- **B key**: Toggle post-processing bypass for debugging
- Config flag: `bypass_postprocess` (default: True)
- Allows isolation of rendering vs post-process issues

### 7. Diagnostic Tools ✓
- **R key**: Reset camera to initial position
- **F key**: Fit camera to view entire point cloud
- Comprehensive logging of:
  - Point count
  - Bounding box min/max
  - Centroid (original and transformed)
  - Bounding radius
  - Scale factor
  - Camera position
  - Near/far planes

## Configuration Flags

```python
CONFIG = {
    'normalize_pointcloud': True,   # Normalize to unit sphere
    'center_pointcloud': True,       # Center at origin
    'auto_camera_fit': True,         # Auto-position camera
    'fov': 60.0,                     # Field of view in degrees
    'debug_overlay': True,           # Show debug HUD
    'bypass_postprocess': True,      # Skip temporal effects
}
```

## Controls

- **WASD/QE**: Move camera (QE = up/down)
- **Mouse**: Look around (click to capture)
- **Scroll**: Adjust movement speed
- **R**: Reset camera to initial position
- **F**: Fit camera to view entire point cloud
- **B**: Toggle post-processing bypass (debug)
- **H**: Toggle HUD
- **ESC**: Exit

## Expected Behavior

With normalization enabled:
1. Point cloud is centered at origin
2. Scaled to unit sphere (radius = 1.0)
3. Camera positioned at distance 2.5 from center
4. Near plane: 0.01, Far plane: 10.0
5. All points visible in stable perspective view

Without normalization:
1. Point cloud at original world coordinates
2. Camera auto-positioned based on bounding radius
3. Near/far planes scaled to bounding radius
4. Should still show stable view (not starburst)

## Troubleshooting

If starburst still appears:
1. Check terminal output for data validation logs
2. Verify position ranges are reasonable (not extreme values)
3. Press **F** to fit camera to view
4. Press **B** to toggle post-processing
5. Check that colors aren't being read as positions
6. Verify bounding radius calculation is correct
