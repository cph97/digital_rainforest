# Bird Spawn System Fix - Summary

## Problem
Birds were not consistently visible and felt like HUD elements instead of spatial 3D objects:
- Sometimes spawned behind camera or off-screen
- Too close to camera (blocking environment)
- Used complex view matrix inversions
- Spawn distances based on near plane (too close)
- Drifted to invalid positions when camera moved

## Root Cause (CRITICAL)
**The `camera.front` vector from the Camera class did NOT match the forward vector derived from the view matrix used for rendering!**

Debug logs showed:
```
Camera.front: [ 0.580287   0.7482673 -0.3215013]
Render forward: [-0.87472004 -0.3626317  -0.32150128]
Vectors match: False
```

This meant birds spawned using one forward direction but were rendered using a completely different one, causing them to appear off-screen.

## Solution
Use the SAME camera basis vectors for spawning, updating, AND rendering:

```python
# Get vectors from view matrix (same as rendering)
cam_forward, cam_right, cam_up = self._camera_basis_for_render()
bird_position = camera.position + cam_forward * depth
```

## Key Changes

### 1. Spawn Logic (`_spawn_birds`)
**Before:**
- Used `_camera_center_world_pos(forward_d)` with view matrix inversions
- Spawn distances: `near_plane * 20.0` or `bounding_radius * 0.60` (too close)
- Stored `forward_d` parameter

**After:**
- Direct formula: `pos = cam_pos + cam_forward * depth`
- Spawn distances: `bounding_radius * 0.8` to `bounding_radius * 1.8` (proper scene depth)
- Stores `depth` parameter
- Validation: `assert dot(bird_pos - cam_pos, cam_front) > 0`

### 2. Update Logic (in `update()` method)
**Before:**
- Used `_camera_center_world_pos(forward_d_anim)` 
- Complex view-space validation with matrix multiplication
- Clamped to near/far based on near_plane

**After:**
- Direct formula: `pos = cam_pos + cam_forward * depth_anim`
- Simple dot product validation
- Repositions birds that drift behind camera
- Depth ranges match spawn ranges

### 3. Configuration Parameters

**New parameters (all optional):**
```python
'bird_depth_near': bounding_radius * 0.8  # Minimum spawn depth
'bird_depth_far': bounding_radius * 1.8   # Maximum spawn depth
'bird_spawn_forward_stack_step': bounding_radius * 0.05  # Depth increment for multiple birds
'bird_depth_wobble_amp': bounding_radius * 0.02  # Subtle depth animation amplitude
'bird_depth_wobble_freq': 0.6  # Depth animation frequency
```

**Removed parameters:**
- `bird_spawn_near_mul_nearplane`
- `bird_spawn_near_mul_bounding_radius`
- `bird_spawn_far_mul_bounding_radius`
- `bird_spawn_far_min_gap`
- `bird_fixed_forward_distance`

### 4. Instance Data Changes
**Before:**
```python
'forward_d': float  # Distance along view axis
```

**After:**
```python
'depth': float  # Distance along camera.front vector
```

## Validation
Every spawn and update includes validation:
```python
v = bird_position - camera.position
dot_product = np.dot(v, camera.front)
assert dot_product > 0  # Bird must be in front
```

If validation fails, bird is repositioned to `camera.position + camera.front * depth_near`

## Expected Behavior
✅ Birds always spawn in front of camera (never behind/sides)  
✅ Birds always visible in frustum (centered in view)  
✅ Birds at comfortable depth (0.8-1.8 × bounding_radius)  
✅ Background environment remains visible  
✅ Camera movement doesn't break placement  
✅ Birds feel like spatial 3D entities, not UI overlays  

## Technical Details

### Why This Works
1. **Simplicity**: Single vector operation, no matrix inversions
2. **Camera-relative**: Birds move with camera automatically
3. **Depth-scaled**: Uses scene scale (bounding_radius) not arbitrary constants
4. **Validated**: Explicit check ensures birds stay in front
5. **Predictable**: Same formula for spawn and update = consistent behavior

### Performance
- Removed expensive view matrix inversions per bird per frame
- Simple vector math: `O(1)` per bird
- No coordinate space transformations needed

## Testing
Run the application and verify:
1. Birds appear when bird audio plays
2. Birds are always visible in center of view
3. Birds don't block entire environment
4. Camera rotation keeps birds in front
5. No "bird behind camera" warnings (unless validation catches drift)

## Cleanup Note
The old `_camera_center_world_pos()` method is now unused and can be removed if desired.
The `_camera_basis_for_render()` method is still used for billboard rotation in `_render_birds()`.
