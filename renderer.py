import moderngl
import numpy as np
import pyrr
import logging
import time
from pathlib import Path

from PIL import Image
import io

try:
    from pygltflib import GLTF2
except Exception:
    GLTF2 = None

logger = logging.getLogger(__name__)


class BirdInstance:
    """Represents a navigating bird with position, velocity, and waypoint system."""
    
    def __init__(self, instance_id, model_idx, position, bounds_min, bounds_max, centroid, bounding_radius, seed=None, nav_center=None):
        self.instance_id = instance_id
        self.model_idx = model_idx
        self.seed = seed if seed is not None else instance_id * 101
        self.rng = np.random.RandomState(self.seed)
        
        # Position and movement
        self.position = np.array(position, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.target_velocity = np.zeros(3, dtype=np.float32)
        
        # Navigation waypoints
        self.waypoints = []
        self.current_waypoint_idx = 0
        self.waypoint_threshold = bounding_radius * 0.15  # Larger threshold for smoother transitions
        
        # Environment bounds for waypoint generation
        self.bounds_min = np.array(bounds_min, dtype=np.float32)
        self.bounds_max = np.array(bounds_max, dtype=np.float32)
        self.centroid = np.array(centroid, dtype=np.float32)
        self.nav_center = np.array(nav_center, dtype=np.float32) if nav_center is not None else self.centroid.copy()
        self.bounding_radius = bounding_radius
        
        # Movement parameters - clearly visible flight
        self.base_speed = bounding_radius * 0.3  # Fast enough to see movement
        self.current_speed = self.base_speed
        self.max_speed = bounding_radius * 0.8  # Higher max speed
        self.turn_speed = 1.2  # Moderate turning
        
        # Visual parameters
        self.ttl = self.rng.uniform(30.0, 60.0)  # Long time to live for extended flight
        self.base_scale_mul = self.rng.uniform(0.85, 1.15)
        self.pulse_amp = self.rng.uniform(0.05, 0.22)
        self.pulse_phase = self.rng.uniform(0.0, 2.0 * np.pi)
        self.pulse_freq = self.rng.uniform(0.8, 2.0)
        
        # Facing direction (for rotation)
        self.facing = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Banking angle for cinematic turns
        self.bank_angle = 0.0
        self.target_bank = 0.0
        
        # Audio reactivity state
        self.last_onset_time = -999.0
        self.energy_smoothed = 0.0
        
        # Generate initial waypoints around this bird's navigation center (usually camera at spawn)
        self._generate_waypoints(5, reference_pos=self.nav_center)
        
        # Start with initial velocity toward first waypoint
        if len(self.waypoints) > 0:
            to_first = self.waypoints[0] - self.position
            dist = np.linalg.norm(to_first)
            if dist > 0.01:
                direction = to_first / dist
                self.velocity = direction * self.base_speed
                self.facing = direction.copy()
    
    def _generate_waypoints(self, count, reference_pos=None):
        """Generate waypoints in a circular/spiral path around a reference position."""
        self.waypoints = []
        
        ref = np.array(reference_pos, dtype=np.float32) if reference_pos is not None else self.nav_center

        # Roaming behavior: pick varied targets across the environment bounds.
        # Slightly bias toward the current reference position so birds stay in view,
        # but still explore the whole space.
        bounds_center = (self.bounds_min + self.bounds_max) * 0.5
        bounds_size = (self.bounds_max - self.bounds_min)
        roam_span = np.maximum(bounds_size * 0.45, self.bounding_radius * 0.4)

        for _i in range(count):
            # Mix of global roaming and local roaming near the reference.
            if float(self.rng.uniform(0.0, 1.0)) < 0.65:
                base = ref
                jitter = (self.rng.uniform(-1.0, 1.0, size=3).astype(np.float32)) * roam_span
                waypoint = base + jitter
            else:
                waypoint = self.rng.uniform(self.bounds_min, self.bounds_max).astype(np.float32)

            # Keep Y within a nicer flight band (avoid hugging floor/ceiling too much)
            y_min = float(self.bounds_min[1] + bounds_size[1] * 0.15)
            y_max = float(self.bounds_max[1] - bounds_size[1] * 0.15)
            waypoint[1] = float(np.clip(waypoint[1], y_min, y_max))

            waypoint = np.clip(waypoint, self.bounds_min, self.bounds_max)
            self.waypoints.append(waypoint)
    
    def update(self, dt, audio_energy=0.0, audio_onset=0.0, current_time=0.0):
        """Update bird navigation based on audio features."""
        if dt <= 0:
            return True  # Still alive
        
        # Smooth audio energy
        energy_alpha = 0.1
        self.energy_smoothed = self.energy_smoothed * (1.0 - energy_alpha) + audio_energy * energy_alpha
        
        # Audio-driven speed modulation
        # Higher energy = faster movement
        energy_factor = 1.0 + self.energy_smoothed * 3.0  # 1x to 4x speed
        self.current_speed = np.clip(
            self.base_speed * energy_factor,
            self.base_speed * 0.5,
            self.max_speed
        )
        
        # Audio onset triggers direction change / new waypoint
        if audio_onset > 0.4 and (current_time - self.last_onset_time) > 0.5:
            self.last_onset_time = current_time
            # Generate new waypoint on strong onset
            self._generate_waypoints(3, reference_pos=self.nav_center)
            self.current_waypoint_idx = 0
            # Boost speed temporarily
            self.current_speed = self.max_speed
        
        # Navigate toward current waypoint
        if len(self.waypoints) > 0 and self.current_waypoint_idx < len(self.waypoints):
            target = self.waypoints[self.current_waypoint_idx]
            to_target = target - self.position
            distance = np.linalg.norm(to_target)
            
            if distance > 0.01:
                # Direction to target
                direction = to_target / distance
                
                # Calculate target velocity - direct approach
                self.target_velocity = direction * self.current_speed
                
                # Blend velocity toward target (simple lerp, responsive)
                blend = min(1.0, self.turn_speed * dt)
                self.velocity = self.velocity * (1.0 - blend) + self.target_velocity * blend
                
                # Update facing direction
                vel_mag = np.linalg.norm(self.velocity)
                if vel_mag > 0.001:
                    self.facing = self.velocity / vel_mag
                
                # Calculate banking for turns
                up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                turn_axis = np.cross(self.facing, direction)
                turn_amount = np.dot(turn_axis, up)
                self.target_bank = np.clip(turn_amount * 30.0, -25.0, 25.0)  # Gentle banking
                
            # Check if reached waypoint
            if distance < self.waypoint_threshold:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    # Generate new waypoints
                    self._generate_waypoints(5)
                    self.current_waypoint_idx = 0
        else:
            # No waypoints, hover in place with slight drift
            self.velocity *= 0.95
        
        # Exponential smoothing for banking
        bank_smoothing = 2.0
        bank_factor = 1.0 - np.exp(-bank_smoothing * dt)
        self.bank_angle = self.bank_angle + (self.target_bank - self.bank_angle) * bank_factor
        
        # Update position
        self.position += self.velocity * dt
        
        # Keep within bounds (soft clamp)
        margin = self.bounding_radius * 0.05
        for i in range(3):
            if self.position[i] < self.bounds_min[i] + margin:
                self.position[i] = self.bounds_min[i] + margin
                self.velocity[i] = abs(self.velocity[i]) * 0.5
            elif self.position[i] > self.bounds_max[i] - margin:
                self.position[i] = self.bounds_max[i] - margin
                self.velocity[i] = -abs(self.velocity[i]) * 0.5
        
        # TTL is managed externally by the renderer
        return True  # Return True - TTL check done externally
    
    def get_forward_vector(self):
        """Get the bird's forward facing direction."""
        return self.facing.copy()
    
    def to_dict(self):
        """Convert to dictionary for compatibility with existing render code."""
        return {
            'instance_id': self.instance_id,
            'model_idx': self.model_idx,
            'pos': self.position.copy(),
            'vel': self.velocity.copy(),
            'ttl': self.ttl,
            'base_scale_mul': self.base_scale_mul,
            'pulse_amp': self.pulse_amp,
            'pulse_phase': self.pulse_phase,
            'pulse_freq': self.pulse_freq,
            'facing': self.facing.copy(),
            'bank_angle': self.bank_angle,
        }


def _quat_slerp(q0, q1, t):
    q0 = np.array(q0, dtype=np.float32)
    q1 = np.array(q1, dtype=np.float32)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        out /= (np.linalg.norm(out) + 1e-8)
        return out
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.sin(theta_0 - theta) / (sin_0 + 1e-8)
    s1 = np.sin(theta) / (sin_0 + 1e-8)
    return (s0 * q0 + s1 * q1).astype(np.float32)


def _compose_trs(translation, rotation_xyzw, scale):
    t = np.array(translation if translation is not None else [0.0, 0.0, 0.0], dtype=np.float32)
    s = np.array(scale if scale is not None else [1.0, 1.0, 1.0], dtype=np.float32)
    q = np.array(rotation_xyzw if rotation_xyzw is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    # Quaternion is [x, y, z, w]
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r = np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy),       0.0],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),       0.0],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy), 0.0],
        [0.0,                   0.0,                   0.0,                   1.0],
    ], dtype=np.float32)

    tr = np.eye(4, dtype=np.float32)
    tr[0, 3] = t[0]
    tr[1, 3] = t[1]
    tr[2, 3] = t[2]

    sc = np.eye(4, dtype=np.float32)
    sc[0, 0] = s[0]
    sc[1, 1] = s[1]
    sc[2, 2] = s[2]

    # glTF local matrix is T * R * S
    return (tr @ r @ sc).astype(np.float32)


class BirdModel:
    def __init__(self, ctx, glb_path: Path):
        self.ctx = ctx
        self.glb_path = glb_path
        self.loaded = False

        self.pivot = np.zeros(3, dtype=np.float32)

        self.program = None
        self.vao = None
        self.vbo = None
        self.ibo = None

        self.texture = None
        self._use_texture = False
        self._base_color_factor = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self._gltf = None
        self._blob = None

        self._mesh_node = None
        self._mesh_index = None
        self._skin_index = None
        self._skin_joints = None
        self._inv_bind = None

        self._node_base_trs = None
        self._parents = None
        self._children = None
        self._root_nodes = None

        self._anim = None
        self._anim_duration = 0.0

        self._max_joints = 128
        self._joint_mats = np.zeros((128, 4, 4), dtype=np.float32)
        self._joint_count = 0

        self._load()

    def _load(self):
        if GLTF2 is None:
            logger.warning("pygltflib not available; skipping bird model")
            return
        if not self.glb_path.exists():
            logger.warning(f"Bird model not found: {self.glb_path}")
            return

        self._gltf = GLTF2().load(str(self.glb_path))
        gltf = self._gltf

        if not gltf.buffers:
            logger.warning("No buffers in bird1.glb")
            return
        self._blob = gltf.binary_blob()
        if self._blob is None:
            logger.warning("Could not read GLB binary blob")
            return

        # Find a node that references the mesh (and its skin)
        mesh_nodes = [(i, n) for i, n in enumerate(gltf.nodes) if getattr(n, 'mesh', None) is not None]
        if not mesh_nodes:
            logger.warning("No mesh nodes found in bird1.glb")
            return
        self._mesh_node, mesh_node_obj = mesh_nodes[0][0], mesh_nodes[0][1]
        self._mesh_index = int(mesh_node_obj.mesh)
        self._skin_index = int(mesh_node_obj.skin) if getattr(mesh_node_obj, 'skin', None) is not None else None

        # Build node parent/child relationships
        n_nodes = len(gltf.nodes)
        self._parents = [-1] * n_nodes
        self._children = [[] for _ in range(n_nodes)]
        for pi, pn in enumerate(gltf.nodes):
            if getattr(pn, 'children', None):
                for c in pn.children:
                    self._parents[int(c)] = pi
                    self._children[pi].append(int(c))
        if gltf.scenes and gltf.scene is not None:
            scene = gltf.scenes[int(gltf.scene)]
            self._root_nodes = [int(x) for x in (scene.nodes or [])]
        elif gltf.scenes:
            self._root_nodes = [int(x) for x in (gltf.scenes[0].nodes or [])]
        else:
            self._root_nodes = [0]

        self._node_base_trs = []
        for n in gltf.nodes:
            self._node_base_trs.append({
                'matrix': getattr(n, 'matrix', None),
                'translation': getattr(n, 'translation', None),
                'rotation': getattr(n, 'rotation', None),
                'scale': getattr(n, 'scale', None),
            })

        def _component_dtype(component_type):
            if component_type == 5120:
                return np.int8
            if component_type == 5121:
                return np.uint8
            if component_type == 5122:
                return np.int16
            if component_type == 5123:
                return np.uint16
            if component_type == 5125:
                return np.uint32
            if component_type == 5126:
                return np.float32
            raise ValueError(f"Unsupported componentType: {component_type}")

        def _component_norm_scale(component_type):
            if component_type == 5120:
                return 127.0
            if component_type == 5121:
                return 255.0
            if component_type == 5122:
                return 32767.0
            if component_type == 5123:
                return 65535.0
            return 1.0

        def _num_components(type_str):
            return {
                'SCALAR': 1,
                'VEC2': 2,
                'VEC3': 3,
                'VEC4': 4,
                'MAT4': 16,
            }[type_str]

        def read_accessor(accessor_idx, force_float=False, force_int=False):
            acc = gltf.accessors[int(accessor_idx)]
            bv = gltf.bufferViews[int(acc.bufferView)]

            dtype = _component_dtype(int(acc.componentType))
            ncomp = _num_components(acc.type)

            byte_offset = int((bv.byteOffset or 0) + (acc.byteOffset or 0))
            stride = int(bv.byteStride or (np.dtype(dtype).itemsize * ncomp))
            count = int(acc.count)

            # Read strided data safely
            out = np.empty((count, ncomp), dtype=dtype)
            comp_size = np.dtype(dtype).itemsize
            row_size = comp_size * ncomp
            for i in range(count):
                start = byte_offset + i * stride
                raw = self._blob[start:start + row_size]
                out[i, :] = np.frombuffer(raw, dtype=dtype, count=ncomp)

            # Handle normalized attributes (never normalize integer index accessors)
            if (not force_int) and getattr(acc, 'normalized', False) and dtype != np.float32:
                out = out.astype(np.float32) / _component_norm_scale(int(acc.componentType))

            if force_float and out.dtype != np.float32:
                out = out.astype(np.float32)
            if force_int and not np.issubdtype(out.dtype, np.integer):
                out = out.astype(np.int32)
            return out

        # Build program with skinning and optional texture
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                const int MAX_JOINTS = 128;
                uniform mat4 projection;
                uniform mat4 view;
                uniform mat4 model;
                uniform mat4 joint_mats[MAX_JOINTS];
                uniform int joint_count;

                in vec3 in_pos;
                in vec3 in_norm;
                in vec2 in_uv;
                in vec4 in_joints;
                in vec4 in_weights;

                out vec3 v_n;
                out vec2 v_uv;

                mat4 skin_matrix() {
                    ivec4 j = ivec4(in_joints);
                    vec4 w = in_weights;
                    mat4 m = mat4(0.0);
                    // Safe clamp to joint_count
                    j = clamp(j, ivec4(0), ivec4(max(joint_count - 1, 0)));
                    m += joint_mats[j.x] * w.x;
                    m += joint_mats[j.y] * w.y;
                    m += joint_mats[j.z] * w.z;
                    m += joint_mats[j.w] * w.w;
                    return m;
                }

                void main() {
                    mat4 sm = skin_matrix();
                    vec4 p = sm * vec4(in_pos, 1.0);
                    vec3 n = mat3(sm) * in_norm;
                    v_n = mat3(model) * n;
                    v_uv = in_uv;
                    gl_Position = projection * view * model * p;
                }
            ''',
            fragment_shader='''
                #version 330
                uniform vec4 base_color_factor;
                uniform int use_base_tex;
                uniform sampler2D base_tex;

                in vec3 v_n;
                in vec2 v_uv;
                out vec4 f_color;

                void main() {
                    vec3 n = normalize(v_n);
                    float l = clamp(dot(n, normalize(vec3(0.4, 0.8, 0.2))) * 0.5 + 0.5, 0.0, 1.0);
                    vec4 base = base_color_factor;
                    if (use_base_tex != 0) {
                        base *= texture(base_tex, v_uv);
                    }
                    vec3 col = base.rgb * (0.35 + 0.65 * l);
                    f_color = vec4(col, 1.0);
                }
            '''
        )

        # Mesh primitive
        if not gltf.meshes or not gltf.meshes[self._mesh_index].primitives:
            logger.warning("No meshes/primitives in bird1.glb")
            return
        prim = gltf.meshes[self._mesh_index].primitives[0]

        # Required attributes for skinning
        pos = read_accessor(prim.attributes.POSITION, force_float=True)
        try:
            pmin = np.min(pos, axis=0)
            pmax = np.max(pos, axis=0)
            self.pivot = ((pmin + pmax) * 0.5).astype(np.float32)
        except Exception:
            self.pivot = np.zeros(3, dtype=np.float32)
        nrm = read_accessor(prim.attributes.NORMAL, force_float=True) if hasattr(prim.attributes, 'NORMAL') and prim.attributes.NORMAL is not None else None
        if nrm is None:
            nrm = np.zeros_like(pos, dtype=np.float32)
            nrm[:, 1] = 1.0

        uv = None
        if hasattr(prim.attributes, 'TEXCOORD_0') and prim.attributes.TEXCOORD_0 is not None:
            uv = read_accessor(prim.attributes.TEXCOORD_0, force_float=True)
            if uv.shape[1] >= 2:
                uv = uv[:, :2]
        if uv is None:
            uv = np.zeros((pos.shape[0], 2), dtype=np.float32)

        joints = None
        if hasattr(prim.attributes, 'JOINTS_0') and prim.attributes.JOINTS_0 is not None:
            # JOINTS_0 are indices, not normalized floats
            joints = read_accessor(prim.attributes.JOINTS_0, force_int=True).astype(np.float32)
            if joints.shape[1] >= 4:
                joints = joints[:, :4]
        if joints is None:
            joints = np.zeros((pos.shape[0], 4), dtype=np.float32)

        weights = None
        if hasattr(prim.attributes, 'WEIGHTS_0') and prim.attributes.WEIGHTS_0 is not None:
            weights = read_accessor(prim.attributes.WEIGHTS_0, force_float=True)
            if weights.shape[1] >= 4:
                weights = weights[:, :4]
        if weights is None:
            weights = np.zeros((pos.shape[0], 4), dtype=np.float32)
            weights[:, 0] = 1.0

        # Normalize weights defensively
        wsum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / (wsum + 1e-8)

        if prim.indices is not None:
            idx = read_accessor(prim.indices, force_int=True).reshape(-1).astype(np.int32)
        else:
            idx = np.arange(len(pos), dtype=np.int32)

        vtx = np.hstack([pos.astype(np.float32), nrm.astype(np.float32), uv.astype(np.float32), joints.astype(np.float32), weights.astype(np.float32)])
        self.vbo = self.ctx.buffer(vtx.tobytes())
        self.ibo = self.ctx.buffer(idx.tobytes())
        self.vao = self.ctx.vertex_array(
            self.program,
            [(self.vbo, '3f 3f 2f 4f 4f', 'in_pos', 'in_norm', 'in_uv', 'in_joints', 'in_weights')],
            self.ibo
        )

        # Skin data
        if self._skin_index is not None and gltf.skins and len(gltf.skins) > self._skin_index:
            skin = gltf.skins[self._skin_index]
            self._skin_joints = [int(j) for j in (skin.joints or [])]
            self._joint_count = min(len(self._skin_joints), self._max_joints)

            if getattr(skin, 'inverseBindMatrices', None) is not None:
                inv = read_accessor(skin.inverseBindMatrices, force_float=True)
                inv = inv.reshape((-1, 16)).astype(np.float32)
                # glTF stores MAT4 in column-major; convert into row-major math matrix
                inv = inv.reshape((-1, 4, 4)).transpose(0, 2, 1)
                self._inv_bind = inv
            else:
                self._inv_bind = np.array([np.eye(4, dtype=np.float32) for _ in range(self._joint_count)], dtype=np.float32)
        else:
            logger.warning("Mesh has no skin; bird may not animate")

        # Animation duration
        if gltf.animations:
            self._anim = gltf.animations[0]
            dur = 0.0
            for s in self._anim.samplers:
                times = read_accessor(s.input, force_float=True).reshape(-1)
                if len(times) > 0:
                    dur = max(dur, float(np.max(times)))
            self._anim_duration = max(dur, 0.0)

        # Material (baseColor)
        try:
            if prim.material is not None and gltf.materials and len(gltf.materials) > int(prim.material):
                mat = gltf.materials[int(prim.material)]
                pbr = getattr(mat, 'pbrMetallicRoughness', None)
                if pbr is not None:
                    if getattr(pbr, 'baseColorFactor', None) is not None:
                        self._base_color_factor = np.array(pbr.baseColorFactor, dtype=np.float32)

                    tex_info = getattr(pbr, 'baseColorTexture', None)
                    if tex_info is not None and getattr(tex_info, 'index', None) is not None and gltf.textures:
                        logger.info(f"Bird baseColorTexture index={int(tex_info.index)}")
                        tex = gltf.textures[int(tex_info.index)]
                        if getattr(tex, 'source', None) is not None and gltf.images:
                            logger.info(f"Bird baseColorTexture source(image)={int(tex.source)}")
                            img = gltf.images[int(tex.source)]
                            raw = None
                            if getattr(img, 'bufferView', None) is not None:
                                logger.info(f"Bird baseColor image bufferView={int(img.bufferView)}")
                                bv = gltf.bufferViews[int(img.bufferView)]
                                start = int(bv.byteOffset or 0)
                                length = int(bv.byteLength or 0)
                                raw = self._blob[start:start + length]
                            elif getattr(img, 'uri', None):
                                uri = str(img.uri)
                                if uri.startswith('data:') and 'base64,' in uri:
                                    import base64
                                    raw = base64.b64decode(uri.split('base64,', 1)[1])
                                else:
                                    # External image path relative to glb
                                    p = (self.glb_path.parent / uri).resolve()
                                    if p.exists():
                                        raw = p.read_bytes()

                            if raw is not None:
                                pil = Image.open(io.BytesIO(raw)).convert('RGBA')
                                pil = pil.transpose(Image.FLIP_TOP_BOTTOM)
                                w, h = pil.size
                                self.texture = self.ctx.texture((w, h), 4, pil.tobytes())
                                self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
                                self.texture.repeat_x = True
                                self.texture.repeat_y = True
                                self._use_texture = True
                                logger.info(f"Loaded bird baseColor texture {w}x{h}")
        except Exception as e:
            logger.warning(f"Failed to load bird material: {e}")

        self.loaded = True
        logger.info(f"Loaded bird model (skinned): {self.glb_path}")

    def _sample_channel(self, times, values, t, is_quat=False):
        if len(times) == 0:
            return values[0]
        if t <= times[0]:
            return values[0]
        if t >= times[-1]:
            return values[-1]
        i = int(np.searchsorted(times, t) - 1)
        i = max(0, min(i, len(times) - 2))
        t0 = float(times[i])
        t1 = float(times[i + 1])
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        v0 = values[i]
        v1 = values[i + 1]
        if is_quat:
            return _quat_slerp(v0, v1, a)
        return (v0 * (1.0 - a) + v1 * a).astype(np.float32)

    def _get_anim_time(self, t_seconds):
        if self._anim is None or self._anim_duration <= 0.0:
            return 0.0
        return float(t_seconds % self._anim_duration)

    def _compute_local_trs_overrides(self, t):
        gltf = self._gltf
        local_trs = [dict(x) for x in self._node_base_trs]
        if self._anim is None:
            return local_trs

        # Accessor reader for animation (float only)
        blob = self._blob

        def read_anim_accessor(accessor_idx):
            acc = gltf.accessors[int(accessor_idx)]
            bv = gltf.bufferViews[int(acc.bufferView)]
            offset = int((bv.byteOffset or 0) + (acc.byteOffset or 0))
            dtype = np.float32
            ncomp = {'SCALAR': 1, 'VEC3': 3, 'VEC4': 4}[acc.type]
            stride = int(bv.byteStride or (4 * ncomp))
            count = int(acc.count)
            out = np.empty((count, ncomp), dtype=np.float32)
            row_size = 4 * ncomp
            for i in range(count):
                start = offset + i * stride
                raw = blob[start:start + row_size]
                out[i, :] = np.frombuffer(raw, dtype=np.float32, count=ncomp)
            return out

        for ch in self._anim.channels:
            node_idx = int(ch.target.node)
            sampler = self._anim.samplers[int(ch.sampler)]
            times = read_anim_accessor(sampler.input).reshape(-1)
            vals = read_anim_accessor(sampler.output)
            path = ch.target.path
            if path == 'translation':
                local_trs[node_idx]['translation'] = self._sample_channel(times, vals, t, is_quat=False).tolist()
                local_trs[node_idx]['matrix'] = None
            elif path == 'scale':
                local_trs[node_idx]['scale'] = self._sample_channel(times, vals, t, is_quat=False).tolist()
                local_trs[node_idx]['matrix'] = None
            elif path == 'rotation':
                local_trs[node_idx]['rotation'] = self._sample_channel(times, vals, t, is_quat=True).tolist()
                local_trs[node_idx]['matrix'] = None

        return local_trs

    def _compute_global_mats(self, local_trs):
        n = len(local_trs)
        globals_m = [np.eye(4, dtype=np.float32) for _ in range(n)]

        def dfs(node_idx, parent_m):
            trs = local_trs[node_idx]
            # Support node.matrix (glTF column-major) when present and not overridden by animated TRS
            if trs.get('matrix') is not None:
                # Convert glTF column-major list into row-major math matrix
                m = np.array(trs.get('matrix'), dtype=np.float32).reshape((4, 4)).T
                local_m = m.astype(np.float32)
            else:
                local_m = _compose_trs(trs.get('translation'), trs.get('rotation'), trs.get('scale'))
            # Global = parent * local (glTF hierarchy)
            g = (parent_m @ local_m).astype(np.float32)
            globals_m[node_idx] = g
            for c in self._children[node_idx]:
                dfs(c, g)

        for r in self._root_nodes:
            dfs(r, pyrr.matrix44.create_identity(dtype=np.float32))

        return globals_m

    def update_skinning(self, t_seconds):
        if not self.loaded or self._skin_joints is None or self._inv_bind is None:
            return
        t = self._get_anim_time(t_seconds)
        local_trs = self._compute_local_trs_overrides(t)
        globals_m = self._compute_global_mats(local_trs)

        # Mesh node global inverse (glTF skinning is relative to the skinned mesh node)
        mesh_global = globals_m[self._mesh_node] if self._mesh_node is not None else np.eye(4, dtype=np.float32)
        try:
            inv_mesh_global = np.linalg.inv(mesh_global).astype(np.float32)
        except Exception:
            inv_mesh_global = np.eye(4, dtype=np.float32)

        # Fill joint matrices
        jc = self._joint_count
        if jc <= 0:
            return
        for i in range(jc):
            jnode = self._skin_joints[i]
            joint_global = globals_m[jnode]
            ib = self._inv_bind[i] if i < len(self._inv_bind) else np.eye(4, dtype=np.float32)
            # glTF: jointMatrix = inverse(meshGlobal) * jointGlobal * inverseBind
            self._joint_mats[i] = (inv_mesh_global @ joint_global @ ib).astype(np.float32)
        # Fill remaining with identity
        for i in range(jc, self._max_joints):
            self._joint_mats[i] = np.eye(4, dtype=np.float32)

    def render(self, projection, view, model):
        if not self.loaded or self.vao is None:
            return
        self.program['projection'].write(projection.tobytes())
        self.program['view'].write(view.tobytes())
        self.program['model'].write(model.tobytes())

        # Material uniforms
        self.program['base_color_factor'].value = tuple(float(x) for x in self._base_color_factor)
        # Set bool uniform via int for better driver compatibility
        self.program['use_base_tex'].value = 1 if self._use_texture else 0
        if self._use_texture and self.texture is not None:
            self.texture.use(location=0)
            self.program['base_tex'].value = 0

        # Skinning uniforms
        self.program['joint_count'].value = int(self._joint_count)
        # GLSL expects column-major matrices; transpose before upload
        self.program['joint_mats'].write(self._joint_mats.transpose(0, 2, 1).tobytes())
        self.vao.render(moderngl.TRIANGLES)

    def release(self):
        if self.vao is not None:
            self.vao.release()
        if self.vbo is not None:
            self.vbo.release()
        if self.ibo is not None:
            self.ibo.release()
        if self.program is not None:
            self.program.release()
        if self.texture is not None:
            self.texture.release()

class Camera:
    """First-person camera with WASD + mouse look controls and bird tracking."""
    
    def __init__(self, position=None, yaw=90.0, pitch=0.0):
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.yaw = yaw
        self.pitch = pitch
        self.speed = 15.0
        self.sensitivity = 0.1
        self.zoom = 45.0
        
        self.front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        self.right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        self.world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Bird tracking mode - DISABLED by default so user can see spawn distribution
        self.tracking_enabled = True
        self.tracked_bird = 3  # Reference to the bird being tracked
        self.tracking_distance = 1.0  # Distance behind bird (will be set based on bounding_radius)
        self.tracking_height_offset = 0.3  # Height offset multiplier
        self.tracking_smoothness = 3.0  # How smoothly camera follows (higher = smoother)
        
        # Cinematic camera state
        self.target_position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.target_yaw = yaw
        self.target_pitch = pitch
        self.transition_progress = 1.0
        self.last_switch_time = 0.0
        
        # Orbit mode for variety
        self.orbit_angle = 0.0
        self.orbit_speed = 0.1  # Radians per second for slow orbit
        self.orbit_enabled = True
        
        # Camera shake for audio reactivity
        self.shake_amount = 0.0
        self.shake_decay = 5.0
        
        self._update_vectors()
    
    def _update_vectors(self):
        """Update camera vectors based on yaw and pitch."""
        front = np.array([
            np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch)),
            np.sin(np.radians(self.pitch)),
            np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        ], dtype=np.float32)
        self.front = front / np.linalg.norm(front)
        
        self.right = np.cross(self.front, self.world_up)
        self.right = self.right / np.linalg.norm(self.right)
        
        self.up = np.cross(self.right, self.front)
        self.up = self.up / np.linalg.norm(self.up)
    
    def get_view_matrix(self):
        """Return view matrix."""
        # Manually construct view matrix for better control
        z_axis = -self.front  # Camera looks down -Z in OpenGL
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(self.world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        # Create view matrix
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = x_axis
        view[1, :3] = y_axis
        view[2, :3] = z_axis
        view[0, 3] = -np.dot(x_axis, self.position)
        view[1, 3] = -np.dot(y_axis, self.position)
        view[2, 3] = -np.dot(z_axis, self.position)
        
        return view
    
    def process_keyboard(self, direction, delta_time):
        """Process WASD movement."""
        velocity = self.speed * delta_time
        
        if direction == 'FORWARD':
            self.position += self.front * velocity
        elif direction == 'BACKWARD':
            self.position -= self.front * velocity
        elif direction == 'LEFT':
            self.position -= self.right * velocity
        elif direction == 'RIGHT':
            self.position += self.right * velocity
        elif direction == 'UP':
            self.position += self.world_up * velocity
        elif direction == 'DOWN':
            self.position -= self.world_up * velocity
    
    def apply_audio_movement(self, audio_features, delta_time, base_position, centroid, bounding_radius):
        """Apply audio-reactive camera movement with position switching."""
        # Extract audio features
        rms = audio_features.get('rms', 0.0)
        onset = audio_features.get('onset', 0.0)
        band_low = audio_features.get('band_low', 0.0)
        band_mid = audio_features.get('band_mid', 0.0)
        band_high = audio_features.get('band_high', 0.0)
        
        # Initialize target position if not set
        if not hasattr(self, 'target_position'):
            self.target_position = self.position.copy()
            self.transition_progress = 1.0
            self.last_switch_time = 0.0
        
        import time
        current_time = time.time()
        
        # Audio-based position switching with cooldown
        # Combine onset (transients) and RMS (loudness) for better triggering
        audio_trigger = onset * 0.7 + rms * 0.3  # Weighted combination
        trigger_threshold = 0.2  # Lowered from 0.3 for better sensitivity
        min_switch_interval = 0.5  # Minimum seconds between switches
        time_since_switch = current_time - self.last_switch_time
        
        # Switch to new random position on strong audio event
        if audio_trigger > trigger_threshold and time_since_switch > min_switch_interval:
            # Generate random position around the point cloud
            # Random spherical coordinates
            theta = np.random.rand() * 2 * np.pi  # Azimuth
            phi = (np.random.rand() * 0.6 + 0.2) * np.pi  # Elevation (avoid top/bottom)
            
            # Distance from centroid - varies with audio energy
            distance = bounding_radius * (1.5 + rms * 2.0)
            
            # Convert to Cartesian
            x = distance * np.sin(phi) * np.cos(theta)
            y = distance * np.cos(phi)
            z = distance * np.sin(phi) * np.sin(theta)
            
            self.target_position = centroid + np.array([x, y, z], dtype=np.float32)
            self.transition_progress = 0.0
            self.last_switch_time = current_time
            
            # Orient camera toward the centroid so forward direction stays stable/meaningful
            to_center = (centroid - self.target_position).astype(np.float32)
            n = float(np.linalg.norm(to_center))
            if n > 1e-6:
                d = to_center / n
                self.target_yaw = float(np.degrees(np.arctan2(d[2], d[0])))
                self.target_pitch = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))
            else:
                self.target_yaw = float(self.yaw)
                self.target_pitch = float(self.pitch)
        
        # Smooth transition to target position
        if self.transition_progress < 1.0:
            # Faster transition with higher mid frequencies
            transition_speed = (1.5 + band_mid * 2.0) * delta_time
            self.transition_progress = min(1.0, self.transition_progress + transition_speed)
            
            # Ease-in-out interpolation
            t = self.transition_progress
            ease_t = t * t * (3.0 - 2.0 * t)  # Smoothstep
            
            # Interpolate position
            self.position = self.position * (1.0 - ease_t) + self.target_position * ease_t
            
            # Interpolate camera orientation
            if hasattr(self, 'target_yaw'):
                self.yaw = self.yaw * (1.0 - ease_t) + self.target_yaw * ease_t
                self.pitch = self.pitch * (1.0 - ease_t) + self.target_pitch * ease_t
                self._update_vectors()
        
        # Add subtle drift even when not switching
        import time
        t = time.time()
        drift_amount = 0.5 + rms * 1.5
        drift_x = np.sin(t * 0.3) * drift_amount
        drift_y = np.sin(t * 0.2) * drift_amount * 0.5
        drift_z = np.cos(t * 0.3) * drift_amount
        
        self.position += np.array([drift_x, drift_y, drift_z], dtype=np.float32) * delta_time

    def apply_orbit_movement(self, delta_time, centroid, bounding_radius):
        """Fallback automated camera movement (slow orbit) when audio features are unavailable."""
        dt = float(max(0.0, min(delta_time, 0.1)))
        self.orbit_angle += float(self.orbit_speed) * dt

        distance = float(bounding_radius) * 2.2
        height = float(bounding_radius) * 0.35
        c = np.array(centroid, dtype=np.float32)

        self.position = c + np.array([
            np.cos(self.orbit_angle) * distance,
            height,
            np.sin(self.orbit_angle) * distance,
        ], dtype=np.float32)

        to_center = (c - self.position).astype(np.float32)
        n = float(np.linalg.norm(to_center))
        if n > 1e-6:
            d = to_center / n
            self.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
            self.pitch = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))
        self._update_vectors()
    
    def process_mouse_movement(self, xoffset, yoffset, constrain_pitch=True):
        """Process mouse look."""
        xoffset *= self.sensitivity
        yoffset *= self.sensitivity
        
        self.yaw += xoffset
        self.pitch += yoffset
        
        if constrain_pitch:
            self.pitch = np.clip(self.pitch, -89.0, 89.0)
        
        self._update_vectors()
    
    def process_scroll(self, yoffset):
        """Process scroll for zoom/speed."""
        self.speed += yoffset * 0.5
        self.speed = max(0.1, min(self.speed, 50.0))
    
    def reset(self, position=None, yaw=None, pitch=None):
        """Reset camera to initial position."""
        self.position = np.array(position if position is not None else [0.0, 200.0, 0.0], dtype=np.float32)
        self.yaw = yaw if yaw is not None else -90.0
        self.pitch = pitch if pitch is not None else 0.0
        self.speed = 15.0
        self._update_vectors()
    
    def update_bird_tracking(self, bird, delta_time, audio_features=None, bounding_radius=1.0):
        """Update camera to cinematically follow a tracked bird with smooth interpolation."""
        if bird is None or not self.tracking_enabled:
            return
        
        # Clamp delta_time to avoid huge jumps
        dt = min(delta_time, 0.1)
        
        # Get bird state
        bird_pos = bird.position.copy()
        bird_facing = bird.get_forward_vector()
        
        # Camera follows behind and above the bird
        # Use a fixed offset that rotates with bird facing direction
        follow_distance = bounding_radius * 0.8  # Further back for wider view
        height_offset = bounding_radius * 0.3  # Higher for better perspective
        
        # Calculate camera position behind the bird
        # Use bird's facing direction to determine "behind"
        behind_dir = -bird_facing
        behind_dir[1] = 0  # Keep horizontal
        behind_len = np.linalg.norm(behind_dir)
        if behind_len > 0.01:
            behind_dir = behind_dir / behind_len
        else:
            behind_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        
        # Target position: behind and above bird
        target_pos = bird_pos + behind_dir * follow_distance
        target_pos[1] = bird_pos[1] + height_offset
        
        # Exponential smoothing for position (frame-rate independent)
        # Lower value = smoother/slower, higher = snappier
        pos_smoothing = 0.3  # Slow, smooth position follow
        pos_factor = 1.0 - np.exp(-pos_smoothing * dt)
        
        self.position = self.position + (target_pos - self.position) * pos_factor
        
        # Calculate look direction to bird
        look_target = bird_pos
        to_bird = look_target - self.position
        dist = np.linalg.norm(to_bird)
        
        if dist > 0.01:
            direction = to_bird / dist
            target_yaw = float(np.degrees(np.arctan2(direction[2], direction[0])))
            target_pitch = float(np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0))))
            
            # Smooth yaw with angle wrapping
            yaw_diff = target_yaw - self.yaw
            while yaw_diff > 180:
                yaw_diff -= 360
            while yaw_diff < -180:
                yaw_diff += 360
            
            # Exponential smoothing for rotation - match position smoothing for consistency
            rot_smoothing = 0.4  # Slow, cinematic rotation
            rot_factor = 1.0 - np.exp(-rot_smoothing * dt)
            
            self.yaw += yaw_diff * rot_factor
            self.pitch += (target_pitch - self.pitch) * rot_factor
            self.pitch = np.clip(self.pitch, -30.0, 30.0)  # Limit pitch for cinematic feel
        
        self._update_vectors()
    
    def set_tracked_bird(self, bird):
        """Set the bird to track."""
        self.tracked_bird = bird
        if bird is not None:
            logger.info(f"Camera now tracking bird {bird.instance_id}")
    
    def toggle_tracking(self):
        """Toggle bird tracking mode."""
        self.tracking_enabled = not self.tracking_enabled
        if not self.tracking_enabled:
            self.tracked_bird = None
        logger.info(f"Bird tracking: {'enabled' if self.tracking_enabled else 'disabled'}")
        return self.tracking_enabled
    
    def toggle_orbit(self):
        """Toggle orbit mode for variety."""
        self.orbit_enabled = not self.orbit_enabled
        logger.info(f"Camera orbit: {'enabled' if self.orbit_enabled else 'disabled'}")
        return self.orbit_enabled


class PointCloudRenderer:
    """GPU-driven point cloud renderer with ModernGL."""
    
    def __init__(self, ctx, width, height, positions, colors, config, bounding_radius=None):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.config = config
        
        logger.info(f"Initializing renderer with {len(positions)} points")
        
        self.point_count = len(positions)
        
        bounds_min = positions.min(axis=0)
        bounds_max = positions.max(axis=0)
        self.centroid = positions.mean(axis=0)
        bounds_size = bounds_max - bounds_min
        
        # Use provided bounding radius or compute from bounds
        if bounding_radius is not None:
            self.bounding_radius = bounding_radius
        else:
            centered = positions - self.centroid
            distances = np.linalg.norm(centered, axis=1)
            self.bounding_radius = distances.max()
        
        logger.info(f"Point cloud bounds: min={bounds_min}, max={bounds_max}")
        logger.info(f"Centroid: {self.centroid}")
        logger.info(f"Bounding radius: {self.bounding_radius:.2f}")

        self.bounds_min = bounds_min.astype(np.float32)
        self.bounds_max = bounds_max.astype(np.float32)
        
        # Auto-calculate near/far planes based on bounding radius
        self.near_plane = max(0.01, self.bounding_radius * 0.01)
        self.far_plane = self.bounding_radius * 10.0
        
        logger.info(f"Auto near/far: {self.near_plane:.3f} / {self.far_plane:.1f}")
        
        # Position camera based on config
        manual_cam_pos = None
        try:
            v = config.get('camera_initial_position', None)
            if isinstance(v, (list, tuple, np.ndarray)) and len(v) == 3:
                manual_cam_pos = np.array(v, dtype=np.float32).reshape(3)
        except Exception:
            manual_cam_pos = None

        if manual_cam_pos is not None:
            camera_pos = manual_cam_pos
        elif config.get('auto_camera_fit', True):
            camera_distance = self.bounding_radius * 2.5

            # Position camera at a diagonal angle: front-right and slightly above
            offset = np.array([camera_distance * 0.7, camera_distance * 0.3, camera_distance * 0.7], dtype=np.float32)
            camera_pos = self.centroid + offset
            logger.info(f"Auto-fit camera at distance: {camera_distance:.2f}")
        else:
            camera_pos = self.centroid + np.array([0, 0, self.bounding_radius * 2.5], dtype=np.float32)
        
        logger.info(f"Camera position: {camera_pos}")
        logger.info(f"Point cloud centroid: {self.centroid}")
        
        yaw = None
        pitch = None
        try:
            if 'camera_initial_yaw' in config:
                yaw = float(config.get('camera_initial_yaw'))
            if 'camera_initial_pitch' in config:
                pitch = float(config.get('camera_initial_pitch'))
        except Exception:
            yaw = None
            pitch = None

        if yaw is None or pitch is None:
            # Calculate yaw and pitch to look at centroid
            to_centroid = self.centroid - camera_pos
            distance_xz = np.sqrt(to_centroid[0]**2 + to_centroid[2]**2)

            # Yaw: angle in XZ plane (0 = +X, 90 = +Z, -90 = -Z, 180/-180 = -X)
            if yaw is None:
                yaw = float(np.degrees(np.arctan2(to_centroid[2], to_centroid[0])))

            # Pitch: angle from horizontal (positive = looking up, negative = looking down)
            if pitch is None:
                pitch = float(np.degrees(np.arctan2(-to_centroid[1], distance_xz)))
        
        logger.info(f"Calculated camera angles to look at centroid: yaw={yaw:.1f}, pitch={pitch:.1f}")
        
        # Camera looking toward the centroid from diagonal angle
        self.camera = Camera(position=camera_pos, yaw=yaw, pitch=pitch)
        self.initial_camera_pos = camera_pos.copy()
        self.initial_yaw = yaw
        self.initial_pitch = pitch
        
        logger.info(f"Camera forward after init: {self.camera.front}")
        
        # Automated camera settings
        # User preference: camera interacts with birds (not sound)
        self.camera_auto_orbit_enabled = bool(self.config.get('camera_auto_orbit_enabled', False))
        try:
            self.camera.orbit_speed = float(self.config.get('camera_orbit_speed', self.camera.orbit_speed))
        except Exception:
            pass

        self.camera_random_rotation_enabled = bool(self.config.get('camera_random_rotation_enabled', False))
        self.camera_random_rotation_interval = float(self.config.get('camera_random_rotation_interval', 3.5))
        self.camera_random_yaw_range = float(self.config.get('camera_random_yaw_range', 25.0))
        self.camera_random_pitch_range = float(self.config.get('camera_random_pitch_range', 10.0))
        self.camera_random_rotation_smoothness = float(self.config.get('camera_random_rotation_smoothness', 1.8))
        self._camera_rand_rot_rng = np.random.RandomState(int(self.config.get('camera_random_rotation_seed', 2026)))
        self._camera_rand_rot_next_time = 0.0
        self._camera_rand_rot_current = np.array([0.0, 0.0], dtype=np.float32)  # yaw_offset, pitch_offset
        self._camera_rand_rot_target = np.array([0.0, 0.0], dtype=np.float32)

        self.birds_idle_anim_enabled = bool(self.config.get('birds_idle_anim_enabled', False))
        self.birds_idle_anim_amp = float(self.config.get('birds_idle_anim_amp', 0.03))
        self.birds_idle_anim_speed = float(self.config.get('birds_idle_anim_speed', 1.2))
        self.audio_camera_enabled = bool(self.config.get('audio_camera_enabled', False))
        self.bird_camera_enabled = bool(self.config.get('bird_camera_enabled', False))
        self.bird_camera_center_enabled = bool(self.config.get('bird_camera_center_enabled', False))
        self.bird_camera_center_height = float(self.config.get('bird_camera_center_height', 0.0))
        self.bird_camera_center_smoothness = float(self.config.get('bird_camera_center_smoothness', 1.4))
        self.bird_camera_switch_interval = float(self.config.get('bird_camera_switch_interval', 6.0))
        self._last_bird_camera_switch_time = -999.0
        self.camera_base_position = camera_pos.copy()
        
        self._create_point_program()
        self._create_point_buffers(positions, colors)
        self._create_post_processing()

        self.birds = []
        self.bird_time = 0.0
        self.bird_instances = []
        self._last_bird_spawn_count = 0
        self._species_stack_counts = {}
        self._bird_energy = 0.0
        bird_count_min = int(self.config.get('bird_count_min', 1))
        bird_count_max = int(self.config.get('bird_count_max', 40))
        if 'bird_count' in self.config:
            self.bird_count = int(self.config.get('bird_count', 6))
        else:
            if bird_count_min > bird_count_max:
                bird_count_min, bird_count_max = bird_count_max, bird_count_min
            bird_count_min = max(0, bird_count_min)
            bird_count_max = max(0, bird_count_max)
            self.bird_count = int(np.random.randint(bird_count_min, bird_count_max + 1))
        self.bird_spread = float(self.config.get('bird_spread', 0.18))
        self.bird_seed = int(self.config.get('bird_seed', 1337))
        self.bird_max_instances = int(self.config.get('bird_max_instances', 60))
        
        # Glitch-bird burst control system
        self.glitch_bird_burst_size = int(self.config.get('glitch_bird_burst_size', 4))  # Birds per glitch burst
        self.glitch_bird_max_stack = int(self.config.get('glitch_bird_max_stack', 12))  # Max birds from glitch/stutter stacking
        self.glitch_cooldown_time = float(self.config.get('glitch_cooldown_time', 0.6))
        self.glitch_decay_rate = float(self.config.get('glitch_decay_rate', 5.0))  # Faster decay: 5 birds/sec
        self.glitch_decay_enabled = bool(self.config.get('glitch_decay_enabled', True))

        self.birds_frozen = bool(self.config.get('birds_frozen', False))
        self.birds_frozen_keepalive = bool(self.config.get('birds_frozen_keepalive', True))
        self.birds_follow_camera = bool(self.config.get('birds_follow_camera', False))
        self._last_birds_follow_cam_pos = None

        self.bird_nav_center_follow_camera = bool(self.config.get('bird_nav_center_follow_camera', True))
        try:
            self.bird_nav_center_forward_distance = float(self.config.get('bird_nav_center_forward_distance', self.bounding_radius * 2.2))
        except Exception:
            self.bird_nav_center_forward_distance = float(self.bounding_radius) * 2.2
        try:
            self.bird_nav_center_smoothness = float(self.config.get('bird_nav_center_smoothness', 2.5))
        except Exception:
            self.bird_nav_center_smoothness = 2.5
        try:
            self.bird_nav_center_waypoint_refresh_dist = float(self.config.get('bird_nav_center_waypoint_refresh_dist', self.bounding_radius * 0.8))
        except Exception:
            self.bird_nav_center_waypoint_refresh_dist = float(self.bounding_radius) * 0.8
        self._bird_nav_center_current = None
        
        self._last_glitch_time = -999.0  # Time of last glitch burst
        self._glitch_was_active = False  # Track glitch state transitions

        bird_dir = Path(__file__).parent / 'birdMesh'
        if bird_dir.exists():
            bird_paths = sorted(
                [p for p in bird_dir.glob('*.glb') if p.is_file()]
                + [p for p in bird_dir.glob('*.gbl') if p.is_file()]
            )
            if len(bird_paths) == 0:
                logger.warning(f"No bird meshes found in {bird_dir} (expected .glb/.gbl)")
            else:
                logger.info(f"Found {len(bird_paths)} bird mesh files")
            for p in bird_paths:
                try:
                    m = BirdModel(self.ctx, p)
                    if getattr(m, 'loaded', False):
                        self.birds.append(m)
                    else:
                        logger.warning(f"Bird model did not load (loaded=False): {p.name}")
                except Exception as e:
                    logger.warning(f"Failed to load bird model {p.name}: {e}")

        if len(self.birds) > 0:
            logger.info(f"Loaded {len(self.birds)} bird models: {[b.glb_path.name for b in self.birds]}")
        else:
            logger.warning("No bird models loaded")

        if len(self.birds) > 0:
            n = max(1, int(self.bird_count))
            self._spawn_birds(n)
            # Camera tracking is disabled by default - user can press T to enable
            logger.info(f"Spawned {len(self.bird_instances)} birds. Press T to enable camera tracking.")

            if bool(self.config.get('camera_initial_at_birds', False)) and len(self.bird_instances) > 0:
                try:
                    h = float(self.config.get('camera_initial_at_birds_height', 0.0))
                except Exception:
                    h = 0.0

                try:
                    positions = np.array([b.position for b in self.bird_instances], dtype=np.float32)
                    birds_center = positions.mean(axis=0).astype(np.float32)
                    birds_center[1] = float(birds_center[1] + h)
                    self.camera.position = birds_center

                    to_target = (self.centroid.astype(np.float32) - self.camera.position).astype(np.float32)
                    nrm = float(np.linalg.norm(to_target))
                    if nrm > 1e-6:
                        d = to_target / nrm
                        self.camera.yaw = float(np.degrees(np.arctan2(d[2], d[0])))
                        self.camera.pitch = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))
                        self.camera._update_vectors()
                except Exception:
                    pass
        
        self.time = 0.0
        self.glitch_enabled = False
        self.paused = False
        
        logger.info("Renderer initialized successfully")
    
    def fit_camera(self):
        """Fit camera to view entire point cloud."""
        camera_distance = self.bounding_radius * 2.5
        offset = np.array([camera_distance * 0.7, camera_distance * 0.3, camera_distance * 0.7], dtype=np.float32)
        camera_pos = self.centroid + offset
        
        # Calculate yaw and pitch to look at centroid
        to_centroid = self.centroid - camera_pos
        distance_xz = np.sqrt(to_centroid[0]**2 + to_centroid[2]**2)
        yaw = np.degrees(np.arctan2(to_centroid[2], to_centroid[0]))
        pitch = np.degrees(np.arctan2(-to_centroid[1], distance_xz))
        
        self.camera.reset(position=camera_pos, yaw=yaw, pitch=pitch)
        logger.info(f"Camera fitted to view at distance: {camera_distance:.2f}, yaw={yaw:.1f}, pitch={pitch:.1f}")
    
    def _orient_camera_to_target(self, target_pos):
        """Orient camera to look at a target position (for bird tracking)."""
        # Calculate direction from camera to target
        to_target = target_pos - self.camera.position
        distance = np.linalg.norm(to_target)
        
        if distance < 1e-6:
            return
        
        # Normalize direction
        direction = to_target / distance
        
        # Calculate yaw and pitch from direction
        # Yaw: angle in XZ plane
        yaw = np.degrees(np.arctan2(direction[2], direction[0]))
        
        # Pitch: angle from horizontal
        pitch = np.degrees(np.arcsin(np.clip(direction[1], -1.0, 1.0)))
        
        # Smoothly interpolate to new orientation for smooth camera movement
        blend = 0.3  # 30% blend per frame for smooth transition
        self.camera.yaw = self.camera.yaw * (1.0 - blend) + yaw * blend
        self.camera.pitch = self.camera.pitch * (1.0 - blend) + pitch * blend
        
        # Update camera vectors
        self.camera._update_vectors()
        
        logger.info(f"Camera oriented to target at {target_pos}, yaw={self.camera.yaw:.1f}, pitch={self.camera.pitch:.1f}")

    def _regen_bird_offsets(self, n: int):
        rng = np.random.RandomState(self.bird_seed)
        offs = np.zeros((n, 3), dtype=np.float32)

        # Even distribution in a disk (Fibonacci/sunflower) to avoid clustering/stacking.
        # offs[:,0:2] in [-1,1] roughly, offs[:,2] in [0,1] for depth variation.
        golden_angle = 2.399963229728653  # pi * (3 - sqrt(5))
        jitter = 0.12
        for i in range(n):
            # r in [0,1], more uniform in area
            r = np.sqrt((i + 0.5) / max(n, 1))
            theta = i * golden_angle + float(rng.uniform(-0.25, 0.25))
            x = r * np.cos(theta)
            y = r * np.sin(theta)

            # Small jitter to make it feel organic, but still spread out
            x = float(np.clip(x + rng.uniform(-jitter, jitter), -1.0, 1.0))
            y = float(np.clip(y + rng.uniform(-jitter, jitter), -1.0, 1.0))
            z = float(rng.uniform(0.0, 1.0))

            offs[i] = (x, y, z)

        return offs

    def _camera_basis_for_render(self):
        view = self.camera.get_view_matrix()
        # Numpy arrays are row-major, but GLSL expects column-major; uploading raw bytes
        # effectively transposes the matrix. Use view.T as the matrix the shader sees.
        view_used = view.T
        try:
            inv_view = np.linalg.inv(view_used).astype(np.float32)
        except Exception:
            inv_view = np.eye(4, dtype=np.float32)

        cam_right = inv_view[:3, 0].astype(np.float32)
        cam_up = inv_view[:3, 1].astype(np.float32)
        cam_forward = (-inv_view[:3, 2]).astype(np.float32)
        cam_forward = cam_forward / (np.linalg.norm(cam_forward) + 1e-8)
        cam_right = cam_right / (np.linalg.norm(cam_right) + 1e-8)
        cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)
        return cam_forward, cam_right, cam_up

    def _camera_center_world_pos(self, forward_d: float):
        view = self.camera.get_view_matrix()
        view_used = view.T
        try:
            inv_view = np.linalg.inv(view_used).astype(np.float32)
        except Exception:
            inv_view = np.eye(4, dtype=np.float32)

        # Center of view in camera space is along -Z.
        p_view = np.array([0.0, 0.0, -float(forward_d), 1.0], dtype=np.float32)
        p_world = inv_view @ p_view
        return p_world[:3].astype(np.float32)

    def _ray_box_t_range(self, origin, direction, bmin, bmax):
        tmin = -np.inf
        tmax = np.inf
        for k in range(3):
            o = float(origin[k])
            d = float(direction[k])
            mn = float(bmin[k])
            mx = float(bmax[k])
            if abs(d) < 1e-8:
                if o < mn or o > mx:
                    return None
                continue
            inv = 1.0 / d
            t1 = (mn - o) * inv
            t2 = (mx - o) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmax < tmin:
                return None
        return float(tmin), float(tmax)

    def _spawn_birds(self, count: int, burst=False):
        """Spawn navigating birds within the environment bounds."""
        if len(self.birds) == 0:
            return
        count = int(max(0, count))
        if count == 0:
            return

        capacity = int(max(0, self.bird_max_instances - len(self.bird_instances)))
        if capacity <= 0:
            return
        count = min(count, capacity)

        logger.info(f"Spawning {count} navigating birds")

        spawn_layout = str(self.config.get('bird_spawn_layout', 'default')).strip().lower()
        logger.info(f"Bird spawn_layout resolved to: {spawn_layout}")

        try:
            spawn_min_dist_abs = self.config.get('bird_spawn_min_dist', None)
            spawn_min_dist_abs = float(spawn_min_dist_abs) if spawn_min_dist_abs is not None else None
        except Exception:
            spawn_min_dist_abs = None
        try:
            spawn_max_dist_abs = self.config.get('bird_spawn_max_dist', None)
            spawn_max_dist_abs = float(spawn_max_dist_abs) if spawn_max_dist_abs is not None else None
        except Exception:
            spawn_max_dist_abs = None
        if spawn_min_dist_abs is not None and spawn_max_dist_abs is not None:
            if spawn_max_dist_abs < spawn_min_dist_abs:
                spawn_min_dist_abs, spawn_max_dist_abs = spawn_max_dist_abs, spawn_min_dist_abs
            spawn_min_dist_abs = float(max(0.0, spawn_min_dist_abs))
            spawn_max_dist_abs = float(max(spawn_min_dist_abs, spawn_max_dist_abs))

        try:
            u_clamp = float(self.config.get('bird_spawn_u_clamp', 1.0))
        except Exception:
            u_clamp = 1.0
        u_clamp = float(max(0.0, min(1.0, u_clamp)))

        try:
            ring_radius = float(self.config.get('bird_species_ring_radius', 0.0))
        except Exception:
            ring_radius = 0.0
        if ring_radius <= 0.0:
            if spawn_min_dist_abs is not None and spawn_max_dist_abs is not None:
                ring_radius = float(spawn_min_dist_abs + 0.5 * (spawn_max_dist_abs - spawn_min_dist_abs))
            else:
                ring_radius = float(self.bounding_radius) * 2.2

        try:
            ring_height = float(self.config.get('bird_species_ring_height', 0.0))
        except Exception:
            ring_height = 0.0

        try:
            stack_step = float(self.config.get('bird_species_stack_step', 0.04))
        except Exception:
            stack_step = 0.04
        stack_step = float(max(0.0, stack_step))

        try:
            jitter = float(self.config.get('bird_species_jitter', 0.01))
        except Exception:
            jitter = 0.01
        jitter = float(max(0.0, jitter))

        manual_offsets = self.config.get('bird_species_manual_offsets', None)
        manual_space = str(self.config.get('bird_species_manual_space', 'world')).strip().lower()
        if manual_space not in ('world', 'camera'):
            manual_space = 'world'

        try:
            sector_spread = float(self.config.get('bird_spawn_sector_spread', 0.35))
        except Exception:
            sector_spread = 0.35
        sector_spread = float(max(0.0, min(2.0, sector_spread)))

        try:
            spawn_min_sep_abs = self.config.get('bird_spawn_min_separation', None)
            spawn_min_sep_abs = float(spawn_min_sep_abs) if spawn_min_sep_abs is not None else None
        except Exception:
            spawn_min_sep_abs = None
        if spawn_min_sep_abs is None:
            spawn_min_sep_abs = float(self.bounding_radius) * 0.25
        spawn_min_sep_abs = float(max(0.0, spawn_min_sep_abs))
        spawn_min_sep2 = float(spawn_min_sep_abs) * float(spawn_min_sep_abs)

        by_model = self.config.get('bird_spawn_by_model', None)
        if by_model is None:
            by_model = (len(self.birds) >= 9)
        by_model = bool(by_model)

        for j in range(count):
            instance_id = len(self.bird_instances)
            
            # Use dedicated RNG for spawn position (not model selection)
            spawn_seed = int((time.time() * 1000000 + instance_id * 7919 + j * 137) % 2**31)
            spawn_rng = np.random.RandomState(spawn_seed)
            
            # Model selection uses original seed
            model_rng = np.random.RandomState(self.bird_seed + instance_id * 101)
            model_idx = int(model_rng.randint(0, len(self.birds)))

            if spawn_layout in ('world', 'bounds', 'global'):
                bounds_size = (self.bounds_max - self.bounds_min).astype(np.float32)
                y_min = float(self.bounds_min[1] + float(bounds_size[1]) * 0.15)
                y_max = float(self.bounds_max[1] - float(bounds_size[1]) * 0.15)

                pos = None
                best = None
                for _attempt in range(24):
                    candidate = spawn_rng.uniform(self.bounds_min, self.bounds_max).astype(np.float32)
                    candidate[1] = float(np.clip(candidate[1], y_min, y_max))

                    ok = True
                    best = None
                    for inst in self.bird_instances[-min(24, len(self.bird_instances)):]:
                        if not isinstance(inst, BirdInstance):
                            continue
                        dp = inst.position - candidate
                        d2 = float(np.dot(dp, dp))
                        if best is None or d2 < best:
                            best = d2
                        if d2 < spawn_min_sep2:
                            ok = False
                            break
                    if ok:
                        pos = candidate
                        break

                if pos is None:
                    pos = candidate
                nav_center = pos.copy()
                bird = BirdInstance(
                    instance_id=instance_id,
                    model_idx=model_idx,
                    position=pos,
                    bounds_min=self.bounds_min,
                    bounds_max=self.bounds_max,
                    centroid=self.centroid,
                    bounding_radius=self.bounding_radius,
                    seed=self.bird_seed + instance_id * 101,
                    nav_center=nav_center
                )
                self.bird_instances.append(bird)
                if instance_id < 5:
                    logger.info(f"World spawn sample {instance_id}: pos={pos}")
                logger.info(f"Bird {instance_id}: spawned at {pos}, speed={bird.base_speed:.3f}, waypoints={len(bird.waypoints)}")
                continue

            cam_pos = self.camera.position.copy()
            cam_front = self.camera.front.copy().astype(np.float32)
            cam_right = self.camera.right.copy().astype(np.float32)
            cam_up = self.camera.world_up.copy().astype(np.float32)

            # Species ring layout: 9 fixed camera-relative anchors, same-species stacking
            if spawn_layout == 'species_ring' and len(self.birds) >= 9:
                s = int(model_idx) % 9
                if isinstance(manual_offsets, (list, tuple)) and len(manual_offsets) >= 9:
                    try:
                        off = np.array(manual_offsets[s], dtype=np.float32).reshape(3)
                    except Exception:
                        off = np.zeros(3, dtype=np.float32)

                    angle = float(s) * (2.0 * np.pi / 9.0)
                    default_dir = (cam_right * float(np.cos(angle)) + cam_front * float(np.sin(angle))).astype(np.float32)
                    nd0 = float(np.linalg.norm(default_dir))
                    if nd0 > 1e-6:
                        default_dir = default_dir / nd0
                    else:
                        default_dir = cam_right.copy()

                    if manual_space == 'camera':
                        # Interpret offsets as camera-basis coordinates: x=right, y=up, z=forward
                        manual_dir = (cam_right * float(off[0]) + cam_front * float(off[2])).astype(np.float32)
                        nd = float(np.linalg.norm(manual_dir))
                        if nd > 1e-6:
                            ring_dir = manual_dir / nd
                            anchor = cam_pos + cam_right * float(off[0]) + cam_up * float(off[1]) + cam_front * float(off[2])
                        else:
                            ring_dir = default_dir
                            anchor = cam_pos + ring_dir * float(ring_radius) + cam_up * float(ring_height + float(off[1]))
                    else:
                        # Interpret offsets as world-space coordinates relative to camera position.
                        manual_dir = off.copy()
                        manual_dir[1] = 0.0
                        nd = float(np.linalg.norm(manual_dir))
                        if nd > 1e-6:
                            ring_dir = manual_dir / nd
                            anchor = cam_pos + off
                        else:
                            ring_dir = default_dir
                            anchor = cam_pos + ring_dir * float(ring_radius) + cam_up * float(ring_height + float(off[1]))
                else:
                    angle = float(s) * (2.0 * np.pi / 9.0)

                    ring_dir = (cam_right * float(np.cos(angle)) + cam_front * float(np.sin(angle))).astype(np.float32)
                    nd = float(np.linalg.norm(ring_dir))
                    if nd > 1e-6:
                        ring_dir = ring_dir / nd
                    else:
                        ring_dir = cam_right.copy()

                    anchor = cam_pos + ring_dir * float(ring_radius) + cam_up * float(ring_height)

                stack_idx = int(self._species_stack_counts.get(s, 0))
                self._species_stack_counts[s] = stack_idx + 1

                offset = ring_dir * 0.0
                if jitter > 0.0:
                    jv = spawn_rng.normal(0.0, 1.0, size=3).astype(np.float32)
                    jv = jv - ring_dir * float(np.dot(jv, ring_dir))
                    nj = float(np.linalg.norm(jv))
                    if nj > 1e-6:
                        jv = jv / nj
                        offset = offset + jv * float(spawn_rng.uniform(-jitter, jitter))

                offset = offset + cam_up * float(stack_idx) * float(stack_step)
                pos = anchor + offset

                nearest_d = None
                best = None
                for inst in self.bird_instances[-min(12, len(self.bird_instances)):]:
                    dp = inst.position - pos
                    d2 = float(np.dot(dp, dp))
                    if best is None or d2 < best:
                        best = d2
                if best is not None:
                    nearest_d = float(np.sqrt(max(0.0, best)))

                logger.info(
                    f"Spawn {instance_id}: layout=species_ring, species={s}, model={model_idx}, "
                    f"cam={cam_pos}, anchor={anchor}, offset={offset}, final={pos}, "
                    f"dist={float(np.linalg.norm(pos - cam_pos)):.3f}, nearest_recent={nearest_d}"
                )

                bird = BirdInstance(
                    instance_id=instance_id,
                    model_idx=model_idx,
                    position=pos,
                    bounds_min=self.bounds_min,
                    bounds_max=self.bounds_max,
                    centroid=self.centroid,
                    bounding_radius=self.bounding_radius,
                    seed=self.bird_seed + instance_id * 101,
                    nav_center=cam_pos
                )
                self.bird_instances.append(bird)
                logger.info(f"Bird {instance_id}: spawned at {pos}, speed={bird.base_speed:.3f}, waypoints={len(bird.waypoints)}")
                continue

            base_dir = None
            sector_name = None
            if by_model and len(self.birds) >= 9:
                sector_dirs = [
                    (cam_up, 'UP'),
                    (-cam_up, 'DOWN'),
                    (cam_left := (-cam_right), 'LEFT'),
                    (cam_right, 'RIGHT'),
                    (cam_front, 'FRONT'),
                    (-cam_front, 'BACK'),
                    (cam_front + cam_left, 'FRONT_LEFT'),
                    (cam_front + cam_right, 'FRONT_RIGHT'),
                    (-cam_front + cam_up, 'BACK_UP'),
                ]
                if 0 <= model_idx < len(sector_dirs):
                    base_dir = sector_dirs[model_idx][0]
                    sector_name = sector_dirs[model_idx][1]
                    nbd = float(np.linalg.norm(base_dir))
                    if nbd > 1e-6:
                        base_dir = base_dir / nbd
                    else:
                        base_dir = None

            # Spawn in a SPHERICAL SHELL around the CURRENT CAMERA POSITION.
            # Use uniform direction sampling (prevents polar clustering) and a volume-weighted radius.
            if spawn_min_dist_abs is not None and spawn_max_dist_abs is not None:
                min_dist = float(spawn_min_dist_abs)
                max_dist = float(spawn_max_dist_abs)
            else:
                min_dist = float(self.bounding_radius) * 1.5
                max_dist = float(self.bounding_radius) * 4.0
            r0, r1 = min_dist, max_dist

            # Basic minimum separation to prevent a visible clump when multiple birds spawn in one frame.
            min_sep = float(self.bounding_radius) * 0.25
            min_sep2 = min_sep * min_sep

            pos = None
            offset = None
            nearest_d = None
            for _attempt in range(12):
                if base_dir is None:
                    u = float(spawn_rng.uniform(-1.0, 1.0))
                    if u_clamp < 1.0:
                        u = float(np.clip(u, -u_clamp, u_clamp))
                    theta = float(spawn_rng.uniform(0.0, 2.0 * np.pi))
                    s = float(np.sqrt(max(0.0, 1.0 - u * u)))
                    dir_vec = np.array([s * np.cos(theta), u, s * np.sin(theta)], dtype=np.float32)
                else:
                    v = spawn_rng.normal(0.0, 1.0, size=3).astype(np.float32)
                    v = v - base_dir * float(np.dot(v, base_dir))
                    nv = float(np.linalg.norm(v))
                    if nv > 1e-6:
                        v = v / nv
                    else:
                        v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                        v = v - base_dir * float(np.dot(v, base_dir))
                        nv = float(np.linalg.norm(v))
                        if nv > 1e-6:
                            v = v / nv
                    a = float(spawn_rng.uniform(0.0, 1.0)) * sector_spread
                    dir_vec = base_dir + v * a
                    nd = float(np.linalg.norm(dir_vec))
                    if nd > 1e-6:
                        dir_vec = dir_vec / nd
                    else:
                        dir_vec = base_dir

                # Radius: uniform in volume between r0..r1
                t = float(spawn_rng.uniform(0.0, 1.0))
                spawn_dist = (r0 ** 3 + t * (r1 ** 3 - r0 ** 3)) ** (1.0 / 3.0)

                offset = dir_vec * float(spawn_dist)
                candidate = cam_pos + offset

                ok = True
                best = None
                for inst in self.bird_instances[-min(12, len(self.bird_instances)):]:
                    dp = inst.position - candidate
                    d2 = float(np.dot(dp, dp))
                    if best is None or d2 < best:
                        best = d2
                    if d2 < min_sep2:
                        ok = False
                        break
                if ok:
                    pos = candidate
                    if best is not None:
                        nearest_d = float(np.sqrt(max(0.0, best)))
                    break

            if pos is None:
                pos = cam_pos + offset
                best = None
                for inst in self.bird_instances[-min(12, len(self.bird_instances)):]:
                    dp = inst.position - pos
                    d2 = float(np.dot(dp, dp))
                    if best is None or d2 < best:
                        best = d2
                if best is not None:
                    nearest_d = float(np.sqrt(max(0.0, best)))
            
            # Don't clamp to bounds - let birds spawn freely around camera
            # They will be clamped during movement updates if needed
            
            logger.info(
                f"Spawn {instance_id}: model={model_idx}, sector={sector_name}, cam={cam_pos}, offset={offset}, final={pos}, "
                f"dist={float(np.linalg.norm(pos - cam_pos)):.3f}, nearest_recent={nearest_d}"
            )
            
            # Create BirdInstance with navigation capabilities
            bird = BirdInstance(
                instance_id=instance_id,
                model_idx=model_idx,
                position=pos,
                bounds_min=self.bounds_min,
                bounds_max=self.bounds_max,
                centroid=self.centroid,
                bounding_radius=self.bounding_radius,
                seed=self.bird_seed + instance_id * 101,
                nav_center=cam_pos
            )
            
            self.bird_instances.append(bird)
            logger.info(f"Bird {instance_id}: spawned at {pos}, speed={bird.base_speed:.3f}, waypoints={len(bird.waypoints)}")
        
        # If tracking is enabled and no bird is being tracked, track the first one
        if self.camera.tracking_enabled and self.camera.tracked_bird is None and len(self.bird_instances) > 0:
            self.camera.set_tracked_bird(self.bird_instances[0])
            self.camera.tracking_distance = self.bounding_radius * 0.15
    
    def _create_point_buffers(self, positions, colors):
        """Create VBO for point cloud data."""
        render_mode = self.config.get('render_mode', 'points')
        
        logger.info(f"VBO data - First 3 positions: {positions[:3]}")
        logger.info(f"VBO data - First 3 colors: {colors[:3]}")
        logger.info(f"VBO data - Color range: {colors.min()} to {colors.max()}")
        
        if render_mode == 'cubes':
            # Create cube geometry
            cube_verts, cube_norms, cube_indices = self._create_cube_geometry()
            
            # Create buffers for cube geometry
            self.cube_vbo = self.ctx.buffer(cube_verts.tobytes())
            self.cube_nbo = self.ctx.buffer(cube_norms.tobytes())
            self.cube_ibo = self.ctx.buffer(cube_indices.tobytes())
            
            # Create instance buffer for positions and colors
            instance_data = np.zeros(len(positions), dtype=[
                ('position', np.float32, 3),
                ('color', np.float32, 3)
            ])
            instance_data['position'] = positions
            instance_data['color'] = colors
            
            self.instance_vbo = self.ctx.buffer(instance_data.tobytes())
            
            # Create VAO with instanced rendering
            self.vao = self.ctx.vertex_array(
                self.point_program,
                [
                    (self.cube_vbo, '3f', 'in_vert'),
                    (self.cube_nbo, '3f', 'in_norm'),
                    (self.instance_vbo, '3f 3f/i', 'instance_position', 'instance_color'),
                ],
                self.cube_ibo
            )
            
            logger.info(f"Created instanced cube rendering with {len(positions)} instances")
        else:
            # Point rendering
            vertices = np.zeros(len(positions), dtype=[
                ('position', np.float32, 3),
                ('color', np.float32, 3)
            ])
            vertices['position'] = positions
            vertices['color'] = colors
            
            self.vbo = self.ctx.buffer(vertices.tobytes())
            self.vao = self.ctx.vertex_array(
                self.point_program,
                [
                    (self.vbo, '3f 3f', 'in_position', 'in_color')
                ]
            )
    
    def _create_cube_geometry(self):
        """Create cube geometry for instanced rendering."""
        # Cube vertices (unit cube centered at origin)
        vertices = np.array([
            # Front face
            [-0.5, -0.5,  0.5], [ 0.5, -0.5,  0.5], [ 0.5,  0.5,  0.5], [-0.5,  0.5,  0.5],
            # Back face
            [-0.5, -0.5, -0.5], [-0.5,  0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5, -0.5, -0.5],
            # Top face
            [-0.5,  0.5, -0.5], [-0.5,  0.5,  0.5], [ 0.5,  0.5,  0.5], [ 0.5,  0.5, -0.5],
            # Bottom face
            [-0.5, -0.5, -0.5], [ 0.5, -0.5, -0.5], [ 0.5, -0.5,  0.5], [-0.5, -0.5,  0.5],
            # Right face
            [ 0.5, -0.5, -0.5], [ 0.5,  0.5, -0.5], [ 0.5,  0.5,  0.5], [ 0.5, -0.5,  0.5],
            # Left face
            [-0.5, -0.5, -0.5], [-0.5, -0.5,  0.5], [-0.5,  0.5,  0.5], [-0.5,  0.5, -0.5],
        ], dtype=np.float32)
        
        # Normals for each face
        normals = np.array([
            # Front
            [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],
            # Back
            [0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1],
            # Top
            [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
            # Bottom
            [0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0],
            # Right
            [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
            # Left
            [-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0],
        ], dtype=np.float32)
        
        # Indices for triangles (2 triangles per face)
        indices = np.array([
            0,1,2, 0,2,3,       # Front
            4,5,6, 4,6,7,       # Back
            8,9,10, 8,10,11,    # Top
            12,13,14, 12,14,15, # Bottom
            16,17,18, 16,18,19, # Right
            20,21,22, 20,22,23, # Left
        ], dtype=np.int32)
        
        return vertices, normals, indices
    
    def _create_point_program(self):
        """Load and compile rendering shaders based on render mode."""
        shader_dir = Path(__file__).parent / 'shaders'
        
        render_mode = self.config.get('render_mode', 'points')
        
        if render_mode == 'cubes':
            with open(shader_dir / 'cubes.vert', 'r') as f:
                vertex_shader = f.read()
            with open(shader_dir / 'cubes.frag', 'r') as f:
                fragment_shader = f.read()
        else:
            with open(shader_dir / 'points.vert', 'r') as f:
                vertex_shader = f.read()
            with open(shader_dir / 'points.frag', 'r') as f:
                fragment_shader = f.read()
        
        self.point_program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
    
    def _create_post_processing(self):
        """Create framebuffers and post-processing pipeline."""
        depth_attachment = self.ctx.depth_renderbuffer((self.width, self.height))
        
        self.fbo_current = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4)],
            depth_attachment=depth_attachment
        )
        
        depth_attachment2 = self.ctx.depth_renderbuffer((self.width, self.height))
        self.fbo_previous = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((self.width, self.height), 4)],
            depth_attachment=depth_attachment2
        )
        
        self.fbo_current.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_previous.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        
        quad_vertices = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
             1.0,  1.0
        ], dtype=np.float32)
        
        self.quad_vbo = self.ctx.buffer(quad_vertices.tobytes())
        
        shader_dir = Path(__file__).parent / 'shaders'
        
        with open(shader_dir / 'post.vert', 'r') as f:
            post_vert = f.read()
        
        with open(shader_dir / 'post.frag', 'r') as f:
            post_frag = f.read()
        
        self.post_program = self.ctx.program(
            vertex_shader=post_vert,
            fragment_shader=post_frag
        )
        
        self.quad_vao = self.ctx.vertex_array(
            self.post_program,
            [(self.quad_vbo, '2f', 'in_position')]
        )

    def _render_birds(self, projection, view):
        if len(self.birds) == 0:
            return

        base_scale = self.bounding_radius * 0.02
        if len(self.bird_instances) == 0:
            return

        for inst in self.bird_instances:
            # Handle both BirdInstance class and legacy dict format
            if isinstance(inst, BirdInstance):
                model_idx = inst.model_idx % len(self.birds)
                bird_model = self.birds[model_idx]
                pos = inst.position
                facing = inst.facing
                bank_angle = inst.bank_angle
                base_scale_mul = inst.base_scale_mul
                pulse_amp = inst.pulse_amp
                pulse_phase = inst.pulse_phase
                pulse_freq = inst.pulse_freq
            else:
                model_idx = int(inst.get('model_idx', 0)) % len(self.birds)
                bird_model = self.birds[model_idx]
                pos = inst.get('pos', None)
                if pos is None:
                    continue
                facing = inst.get('facing', np.array([1.0, 0.0, 0.0], dtype=np.float32))
                bank_angle = inst.get('bank_angle', 0.0)
                base_scale_mul = float(inst.get('base_scale_mul', 1.0))
                pulse_amp = float(inst.get('pulse_amp', 0.12))
                pulse_phase = float(inst.get('pulse_phase', 0.0))
                pulse_freq = float(inst.get('pulse_freq', 1.2))

            # Audio reactive scale pulse using bird energy (0..1-ish), keep it subtle.
            e = float(np.clip((float(self._bird_energy) - 0.01) / 0.06, 0.0, 1.0))
            pulse = 1.0 + pulse_amp * e * float(np.sin(self.time * pulse_freq + pulse_phase))

            scl = base_scale * base_scale_mul * pulse
            scale = pyrr.matrix44.create_from_scale(
                np.array([scl, scl, scl], dtype=np.float32),
                dtype=np.float32
            )

            pivot = getattr(bird_model, 'pivot', None)
            if pivot is None:
                pivot = np.zeros(3, dtype=np.float32)
            pivot_t = pyrr.matrix44.create_from_translation((-np.array(pivot, dtype=np.float32)), dtype=np.float32)

            # Build rotation matrix from facing direction and bank angle
            # Forward = facing direction, Up = world up rotated by bank
            forward = np.array(facing, dtype=np.float32)
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            right = np.cross(forward, world_up)
            right_len = np.linalg.norm(right)
            if right_len < 0.01:
                right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            else:
                right = right / right_len
            
            up = np.cross(right, forward)
            up = up / (np.linalg.norm(up) + 1e-8)
            
            # Apply banking (rotation around forward axis)
            bank_rad = np.radians(bank_angle)
            cos_bank = np.cos(bank_rad)
            sin_bank = np.sin(bank_rad)
            banked_right = right * cos_bank + up * sin_bank
            banked_up = up * cos_bank - right * sin_bank
            
            # Build rotation matrix (bird faces along forward)
            rot = np.eye(4, dtype=np.float32)
            rot[:3, 0] = banked_right
            rot[:3, 1] = banked_up
            rot[:3, 2] = -forward  # Bird model faces -Z
            rot[3, 3] = 1.0

            # Compose in the standard order: T * R * S
            trans = pyrr.matrix44.create_from_translation(np.array(pos, dtype=np.float32), dtype=np.float32)
            model = pyrr.matrix44.multiply(scale, pivot_t)
            model = pyrr.matrix44.multiply(rot, model)
            model = pyrr.matrix44.multiply(trans, model)
            bird_model.render(projection, view, model)
    
    def render(self, audio_features):
        """Main render loop."""
        # Only log on first frame
        if not hasattr(self, '_first_render_logged'):
            logger.info(f"First render - Camera pos: {self.camera.position}, front: {self.camera.front}")
            self._first_render_logged = True
        
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Render to FBO for post-processing
        bypass_post = self.config.get('bypass_postprocess', False)
        
        if bypass_post:
            # Render directly to screen
            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.1, 0.2, 1.0)
        else:
            # Render to framebuffer for post-processing
            self.fbo_current.use()
            self.fbo_current.clear(0.0, 0.1, 0.2, 1.0)
        
        fov = self.config.get('fov', 60.0)
        projection = pyrr.matrix44.create_perspective_projection(
            fov, self.width / self.height, self.near_plane, self.far_plane, dtype=np.float32
        )
        view = self.camera.get_view_matrix()
        
        render_mode = self.config.get('render_mode', 'points')
        
        if render_mode == 'cubes':
            # Cube rendering with separate projection and view
            self.point_program['projection'].write(projection.tobytes())
            self.point_program['view'].write(view.tobytes())
            self.point_program['cube_size'].value = self.config.get('cube_size', 0.01)
        else:
            # Point rendering with combined MVP
            mvp = projection @ view
            self.point_program['mvp'].write(mvp.tobytes())
            self.point_program['point_size'].value = self.config['point_size']
        
        self.point_program['time'].value = self.time
        self.point_program['audio_rms'].value = audio_features.get('rms', 0.0)
        self.point_program['audio_onset'].value = audio_features.get('onset', 0.0)
        self.point_program['delay_energy'].value = audio_features.get('delay_energy', 0.0)
        self.point_program['wind_strength'].value = 1.0
        
        fog_density = 0.01 + audio_features.get('spectral_centroid', 0.0) * 0.02
        self.point_program['fog_density'].value = fog_density
        self.point_program['fog_start'].value = 100.0
        self.point_program['fog_color'].value = (0.05, 0.05, 0.08)
        
        if render_mode == 'cubes':
            # Render instanced cubes
            self.vao.render(moderngl.TRIANGLES, instances=self.point_count)
        else:
            # Render points
            self.vao.render(moderngl.POINTS, vertices=self.point_count)
        
        self.ctx.disable(moderngl.DEPTH_TEST)
        
        # Apply post-processing if enabled
        if not bypass_post:
            # Switch to screen for post-processing output
            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            
            # Bind textures
            self.post_program['current_frame'].value = 0
            self.post_program['previous_frame'].value = 1
            self.fbo_current.color_attachments[0].use(location=0)
            self.fbo_previous.color_attachments[0].use(location=1)
            
            # Set uniforms for databending effect
            self.post_program['time'].value = self.time
            self.post_program['temporal_feedback'].value = self.config['temporal_feedback']
            self.post_program['smear_strength'].value = self.config['smear_strength']
            self.post_program['tear_rate'].value = 0.08
            self.post_program['column_shift_scale'].value = 0.3  # Vertical displacement amount
            self.post_program['chroma_offset'].value = 2.0  # RGB separation
            self.post_program['glitch_intensity'].value = self.config.get('glitch_intensity', 0.5)
            self.post_program['audio_onset'].value = audio_features.get('onset', 0.0)
            self.post_program['audio_band_low'].value = audio_features.get('band_low', 0.0)
            self.post_program['audio_band_mid'].value = audio_features.get('band_mid', 0.0)
            self.post_program['audio_band_high'].value = audio_features.get('band_high', 0.0)
            self.post_program['resolution'].value = (self.width, self.height)
            self.post_program['glitch_enabled'].value = self.glitch_enabled
            
            # Render fullscreen quad with post-processing
            self.quad_vao.render(moderngl.TRIANGLE_STRIP)
            
            # Swap framebuffers for temporal feedback
            self.fbo_current, self.fbo_previous = self.fbo_previous, self.fbo_current

        # Render birds after post-processing so databending doesn't affect them
        self.ctx.screen.use()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self._render_birds(projection, view)
    
    def update(self, delta_time, audio_features=None):
        """Update time and animations."""
        if not self.paused:
            self.time += delta_time

            try:
                bird_sep_radius = float(self.config.get('bird_separation_radius', self.bounding_radius * 0.35))
            except Exception:
                bird_sep_radius = float(self.bounding_radius) * 0.35
            try:
                bird_sep_strength = float(self.config.get('bird_separation_strength', 1.6))
            except Exception:
                bird_sep_strength = 1.6
            bird_sep_radius = float(max(0.0, bird_sep_radius))
            bird_sep_strength = float(max(0.0, bird_sep_strength))
            bird_sep_r2 = bird_sep_radius * bird_sep_radius

            if audio_features is not None:
                try:
                    self._bird_energy = float(audio_features.get('bird_energy', 0.0))
                except Exception:
                    self._bird_energy = 0.0

            bird_playing = bool(audio_features.get('bird_playing', False)) if audio_features is not None else False

            if (not self.birds_frozen) and self.bird_nav_center_follow_camera and len(self.bird_instances) > 0:
                cam_pos = self.camera.position.copy().astype(np.float32)
                cam_front = self.camera.front.copy().astype(np.float32)
                nf = float(np.linalg.norm(cam_front))
                if nf > 1e-6:
                    cam_front = cam_front / nf
                desired_center = (cam_pos + cam_front * float(self.bird_nav_center_forward_distance)).astype(np.float32)

                if self._bird_nav_center_current is None:
                    self._bird_nav_center_current = desired_center
                else:
                    dt = float(max(0.0, min(delta_time, 0.1)))
                    smooth = float(max(0.01, self.bird_nav_center_smoothness))
                    k = 1.0 - np.exp(-smooth * dt)
                    self._bird_nav_center_current = (self._bird_nav_center_current + (desired_center - self._bird_nav_center_current) * k).astype(np.float32)

                for inst in self.bird_instances:
                    if not isinstance(inst, BirdInstance):
                        continue

                    prev = getattr(inst, '_last_nav_center', None)
                    inst.nav_center = self._bird_nav_center_current.copy()

                    if prev is None:
                        inst._last_nav_center = inst.nav_center.copy()
                    else:
                        dp = inst.nav_center - prev
                        if float(np.dot(dp, dp)) >= float(self.bird_nav_center_waypoint_refresh_dist) ** 2:
                            inst._generate_waypoints(5, reference_pos=inst.nav_center)
                            inst.current_waypoint_idx = 0
                            inst._last_nav_center = inst.nav_center.copy()

            # If birds are frozen but should remain around the camera, translate them by the camera delta.
            if self.birds_frozen and self.birds_follow_camera and len(self.bird_instances) > 0:
                cam_pos = self.camera.position.copy()
                if self._last_birds_follow_cam_pos is None:
                    self._last_birds_follow_cam_pos = cam_pos
                else:
                    delta = (cam_pos - self._last_birds_follow_cam_pos).astype(np.float32)
                    if float(np.dot(delta, delta)) > 1e-12:
                        for inst in self.bird_instances:
                            if isinstance(inst, BirdInstance):
                                inst.position = (inst.position + delta).astype(np.float32)
                                inst.nav_center = (inst.nav_center + delta).astype(np.float32)
                                if hasattr(inst, '_frozen_base_pos') and inst._frozen_base_pos is not None:
                                    inst._frozen_base_pos = (inst._frozen_base_pos + delta).astype(np.float32)
                            else:
                                p = inst.get('pos', None)
                                if p is not None:
                                    inst['pos'] = (np.array(p, dtype=np.float32) + delta).astype(np.float32)
                        self._last_birds_follow_cam_pos = cam_pos

                if int(self.time) % 2 == 0 and int(self.time * 10) % 10 == 0:
                    logger.info(f"Bird follow camera: cam_pos={cam_pos}, bird0={self.bird_instances[0].position if len(self.bird_instances) > 0 else None}")

            # Bird-driven cinematic camera (independent of audio)
            if self.camera_auto_orbit_enabled:
                self.camera.tracking_enabled = False
                self.camera.tracked_bird = None
                self.camera.apply_orbit_movement(delta_time, self.centroid, self.bounding_radius)

                if self.camera_random_rotation_enabled:
                    dt = float(max(0.0, min(delta_time, 0.1)))
                    if self._camera_rand_rot_next_time <= 0.0:
                        self._camera_rand_rot_next_time = float(self.time) + self.camera_random_rotation_interval

                    if float(self.time) >= float(self._camera_rand_rot_next_time):
                        self._camera_rand_rot_next_time = float(self.time) + self.camera_random_rotation_interval
                        yaw_off = float(self._camera_rand_rot_rng.uniform(-self.camera_random_yaw_range, self.camera_random_yaw_range))
                        pitch_off = float(self._camera_rand_rot_rng.uniform(-self.camera_random_pitch_range, self.camera_random_pitch_range))
                        self._camera_rand_rot_target = np.array([yaw_off, pitch_off], dtype=np.float32)

                    smooth = float(max(0.01, self.camera_random_rotation_smoothness))
                    k = 1.0 - np.exp(-smooth * dt)
                    self._camera_rand_rot_current = (self._camera_rand_rot_current + (self._camera_rand_rot_target - self._camera_rand_rot_current) * k).astype(np.float32)

                    self.camera.yaw = float(self.camera.yaw) + float(self._camera_rand_rot_current[0])
                    self.camera.pitch = float(self.camera.pitch) + float(self._camera_rand_rot_current[1])
                    self.camera.pitch = float(np.clip(self.camera.pitch, -89.0, 89.0))
                    self.camera._update_vectors()

            elif self.bird_camera_enabled and self.bird_camera_center_enabled and len(self.bird_instances) > 0:
                # Center the camera at the centroid of the current bird distribution.
                # This places the camera in the "center of the radius of the birds".
                self.camera.tracking_enabled = False
                self.camera.tracked_bird = None

                dt = float(max(0.0, min(delta_time, 0.1)))
                positions = np.array([b.position for b in self.bird_instances], dtype=np.float32)
                birds_center = positions.mean(axis=0)
                birds_center[1] = float(birds_center[1] + self.bird_camera_center_height)

                smooth = float(max(0.01, self.bird_camera_center_smoothness))
                pos_factor = 1.0 - np.exp(-smooth * dt)
                self.camera.position = self.camera.position + (birds_center - self.camera.position) * pos_factor

                # Smoothly orient the camera to look at the bird cluster.
                to_center = (birds_center - self.camera.position).astype(np.float32)
                n = float(np.linalg.norm(to_center))
                if n > 1e-3:
                    d = to_center / n
                    target_yaw = float(np.degrees(np.arctan2(d[2], d[0])))
                    target_pitch = float(np.degrees(np.arcsin(np.clip(d[1], -1.0, 1.0))))

                    yaw_diff = target_yaw - float(self.camera.yaw)
                    while yaw_diff > 180.0:
                        yaw_diff -= 360.0
                    while yaw_diff < -180.0:
                        yaw_diff += 360.0

                    rot_factor = 1.0 - np.exp(-(smooth * 1.1) * dt)
                    self.camera.yaw = float(self.camera.yaw) + yaw_diff * rot_factor
                    self.camera.pitch = float(self.camera.pitch) + (target_pitch - float(self.camera.pitch)) * rot_factor
                    self.camera._update_vectors()

            elif self.bird_camera_enabled and not self.camera.tracking_enabled:
                self.camera.tracking_enabled = True

            if (not self.camera_auto_orbit_enabled) and self.bird_camera_enabled and self.camera.tracking_enabled:
                if len(self.bird_instances) > 0:
                    # Pick a bird if none is tracked, or periodically switch for variety
                    need_switch = (self.camera.tracked_bird is None)
                    if not need_switch:
                        if (self.time - self._last_bird_camera_switch_time) >= self.bird_camera_switch_interval:
                            need_switch = True

                    if need_switch:
                        idx = int(np.random.randint(0, len(self.bird_instances)))
                        self.camera.set_tracked_bird(self.bird_instances[idx])
                        self._last_bird_camera_switch_time = self.time

                    # Smooth follow
                    if self.camera.tracked_bird is not None:
                        self.camera.update_bird_tracking(
                            self.camera.tracked_bird,
                            delta_time,
                            audio_features=None,
                            bounding_radius=self.bounding_radius
                        )
                else:
                    # No birds yet: keep camera moving so the scene is never static
                    self.camera.apply_orbit_movement(delta_time, self.centroid, self.bounding_radius)
            elif not self.camera_auto_orbit_enabled:
                # Audio-reactive camera (only if not tracking a bird)
                if self.audio_camera_enabled and not self.camera.tracking_enabled:
                    if audio_features:
                        self.camera.apply_audio_movement(
                            audio_features,
                            delta_time,
                            self.camera_base_position,
                            self.centroid,
                            self.bounding_radius
                        )
                    else:
                        # If there is no audio feature stream, still keep the camera moving.
                        self.camera.apply_orbit_movement(delta_time, self.centroid, self.bounding_radius)

            # Spawn per-frame events (non-cumulative) AFTER camera movement is final for this frame
            if audio_features is not None and len(self.birds) > 0:
                # Check for stutter-driven spawns (independent of bird audio)
                stutter_active = bool(audio_features.get('stutter_active', False))
                stutter_intensity = float(audio_features.get('stutter_intensity', 0.0))
                
                # Trigger spawn events when stutter is active
                if stutter_active and stutter_intensity > 0.0:
                    # Force spawn event when stutter detected
                    events = 1
                    logger.warning(f">>> STUTTER ACTIVE: intensity={stutter_intensity:.2f}, current birds={len(self.bird_instances)}")
                else:
                    try:
                        events = int(audio_features.get('bird_spawn_events', 0))
                    except Exception:
                        events = 0
                    if events > 0:
                        logger.info(f"Bird spawn events from audio: {events}")
                
                if events > 0:
                    burst = False
                    try:
                        burst = float(audio_features.get('onset', 0.0)) > 0.4
                    except Exception:
                        burst = False
                    
                    # Glitch-bird burst system with cooldown and cap
                    glitch_active = bool(audio_features.get('glitch_active', False))
                    
                    # Track glitch state transitions
                    if glitch_active and not self._glitch_was_active:
                        logger.info(f"GLITCH EVENT STARTED at t={self.time:.2f}")
                    elif not glitch_active and self._glitch_was_active:
                        logger.info(f"GLITCH EVENT ENDED at t={self.time:.2f}")
                    
                    self._glitch_was_active = glitch_active

                    for _ in range(events):
                        if glitch_active or stutter_active:
                            # Check cooldown
                            time_since_last_burst = self.time - self._last_glitch_time
                            if time_since_last_burst < self.glitch_cooldown_time:
                                logger.debug(f"Glitch burst on cooldown ({time_since_last_burst:.2f}s < {self.glitch_cooldown_time}s)")
                                continue
                            
                            # Check if we're at max stack
                            current_stack = len(self.bird_instances)
                            if current_stack >= self.glitch_bird_max_stack:
                                logger.debug(f"Max bird stack reached ({current_stack}/{self.glitch_bird_max_stack})")
                                continue
                            
                            # MAP STUTTER INTENSITY TO BIRD DENSITY
                            # stutter_intensity: 0.0 (no stutter) to 1.0 (all layers stuttering)
                            # More stutter = more birds (scaled for max 30 cap)
                            if stutter_active:
                                # Scale burst size based on stutter intensity
                                base_burst = self.glitch_bird_burst_size
                                intensity_multiplier = 0.3 + stutter_intensity * 1.0  # 0.3x to 1.3x (smaller range for 30 cap)
                                k = int(base_burst * intensity_multiplier)
                                k = max(1, k)  # At least 1 bird
                                logger.warning(f"STUTTER BURST: intensity={stutter_intensity:.2f}, spawning {k} birds (base={base_burst})")
                            else:
                                # Regular glitch burst
                                k = self.glitch_bird_burst_size
                                logger.info(f"GLITCH BURST: spawning {k} birds")
                            
                            # Cap to available slots
                            available_slots = self.glitch_bird_max_stack - current_stack
                            k = min(k, available_slots)
                            
                            logger.info(f"Spawning {k} birds (stack: {current_stack} -> {current_stack + k}/{self.glitch_bird_max_stack})")
                            self._spawn_birds(k, burst=True)
                            self._last_glitch_time = self.time
                        else:
                            # NO GLITCH/STUTTER: Spawn single bird
                            k = 1
                            self._spawn_birds(k, burst=burst)

            # Keepalive: if bird audio is active but no birds exist (e.g. after TTL expiry), spawn one.
            if bird_playing and len(self.birds) > 0 and len(self.bird_instances) == 0:
                if (not self.birds_frozen) or self.birds_frozen_keepalive:
                    self._spawn_birds(1, burst=False)

            # Update bird navigation with audio-driven movement
            if len(self.bird_instances) > 0:
                dt = float(delta_time)
                
                # Get audio features for bird movement
                audio_energy = float(audio_features.get('bird_energy', audio_features.get('rms', 0.0))) if audio_features else 0.0
                audio_onset = float(audio_features.get('onset', 0.0)) if audio_features else 0.0
                stutter_active = bool(audio_features.get('stutter_active', False)) if audio_features else False
                
                keep = []
                birds_to_decay = 0
                
                # Calculate how many birds to remove due to decay when stutter is NOT active
                if (not self.birds_frozen) and self.glitch_decay_enabled and not stutter_active and len(self.bird_instances) > 1:
                    birds_to_decay = int(self.glitch_decay_rate * dt)
                    if birds_to_decay > 0:
                        logger.debug(f"DECAY: Removing {birds_to_decay} birds")
                
                for idx, inst in enumerate(self.bird_instances):
                    # Apply decay by skipping oldest birds first
                    if birds_to_decay > 0 and idx < birds_to_decay:
                        # If we're removing the tracked bird, clear tracking
                        if self.camera.tracked_bird is inst:
                            self.camera.tracked_bird = None
                        continue
                    
                    # Handle BirdInstance class
                    if isinstance(inst, BirdInstance):
                        if self.birds_frozen:
                            # Freeze: keep spawned clusters fixed in place.
                            inst.velocity *= 0.0
                            inst.target_velocity *= 0.0

                            if self.birds_idle_anim_enabled:
                                if not hasattr(inst, '_frozen_base_pos') or inst._frozen_base_pos is None:
                                    inst._frozen_base_pos = inst.position.copy().astype(np.float32)

                                a = float(self.birds_idle_anim_amp)
                                w = float(self.birds_idle_anim_speed)
                                phase = float(inst.instance_id) * 0.77
                                t = float(self.time)
                                dx = np.sin(t * w + phase) * (a * 0.6)
                                dy = np.sin(t * (w * 1.3) + phase * 1.9) * a
                                dz = np.cos(t * (w * 0.9) + phase * 1.2) * (a * 0.6)
                                inst.position = (inst._frozen_base_pos + np.array([dx, dy, dz], dtype=np.float32)).astype(np.float32)
                        else:
                            # Normal update
                            old_pos = inst.position.copy()
                            inst.update(
                                dt=dt,
                                audio_energy=audio_energy,
                                audio_onset=audio_onset,
                                current_time=self.time
                            )

                            if bird_sep_strength > 0.0 and bird_sep_radius > 0.0 and len(self.bird_instances) > 1:
                                repulse = np.zeros(3, dtype=np.float32)
                                p = inst.position
                                for other in self.bird_instances:
                                    if other is inst or (not isinstance(other, BirdInstance)):
                                        continue
                                    d = p - other.position
                                    d2 = float(np.dot(d, d))
                                    if d2 < 1e-12 or d2 >= bird_sep_r2:
                                        continue
                                    inv = 1.0 / (np.sqrt(d2) + 1e-6)
                                    w = (1.0 - (np.sqrt(d2) / (bird_sep_radius + 1e-6)))
                                    repulse = repulse + (d * inv * float(w))

                                if float(np.dot(repulse, repulse)) > 1e-12:
                                    repulse = repulse / (float(np.linalg.norm(repulse)) + 1e-6)
                                    inst.velocity = (inst.velocity + repulse * float(bird_sep_strength) * float(dt) * float(inst.base_speed)).astype(np.float32)
                                    sp = float(np.linalg.norm(inst.velocity))
                                    if sp > float(inst.max_speed):
                                        inst.velocity = (inst.velocity / (sp + 1e-6) * float(inst.max_speed)).astype(np.float32)
                            
                            # Log position changes for multiple birds to show movement
                            pos_delta = np.linalg.norm(inst.position - old_pos)
                            if idx < 3 and int(self.time * 2) % 4 == 0:  # Log first 3 birds every 2s
                                logger.info(f"Bird {idx}: pos={inst.position}, vel={np.linalg.norm(inst.velocity):.3f}, delta={pos_delta:.4f}")
                        
                        # TTL only decreases when no bird audio is playing
                        if (not self.birds_frozen) and (not bird_playing):
                            inst.ttl -= dt
                        
                        if self.birds_frozen:
                            keep.append(inst)
                        elif inst.ttl > 0:
                            keep.append(inst)
                        else:
                            # If tracked bird dies, clear tracking
                            if self.camera.tracked_bird is inst:
                                self.camera.tracked_bird = None
                            logger.info(f"Bird {inst.instance_id} expired")
                    else:
                        # Legacy dict format fallback
                        ttl = float(inst.get('ttl', 0.0))
                        if not bird_playing:
                            ttl -= dt
                        if ttl <= 0.0:
                            continue
                        inst['ttl'] = ttl
                        keep.append(inst)

                self.bird_instances = keep
                
                # Update camera to track active bird
                if self.camera.tracking_enabled and len(self.bird_instances) > 0:
                    # If no bird is being tracked, select the first one
                    if self.camera.tracked_bird is None or self.camera.tracked_bird not in self.bird_instances:
                        self.camera.set_tracked_bird(self.bird_instances[0])
                    
                    # Update camera tracking
                    if self.camera.tracked_bird is not None:
                        self.camera.update_bird_tracking(
                            self.camera.tracked_bird,
                            dt,
                            audio_features,
                            self.bounding_radius
                        )

            if len(self.birds) > 0:
                # Audio-reactive flap speed. Use bird_energy from audio thread.
                # Add small per-model phase/speed offsets so different species don't flap in sync.
                e = float(np.clip((float(self._bird_energy) - 0.01) / 0.06, 0.0, 1.0))
                base_speed = 0.7 + 3.3 * e
                self.bird_time += float(delta_time) * float(base_speed)

                for mi, b in enumerate(self.birds):
                    if not getattr(b, 'loaded', False):
                        continue
                    rng = np.random.RandomState(self.bird_seed + mi * 997)
                    speed_jitter = float(rng.uniform(0.92, 1.08))
                    phase = float(rng.uniform(0.0, 10.0))
                    b.update_skinning(self.bird_time * speed_jitter + phase)
    
    def toggle_glitch(self):
        """Toggle glitch effect."""
        self.glitch_enabled = not self.glitch_enabled
        logger.info(f"Glitch effect: {'enabled' if self.glitch_enabled else 'disabled'}")
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.paused = not self.paused
        logger.info(f"Simulation: {'paused' if self.paused else 'running'}")
    
    def toggle_audio_camera(self):
        """Toggle audio-reactive camera movement."""
        self.audio_camera_enabled = not self.audio_camera_enabled
        if not self.audio_camera_enabled:
            # Reset to base position when disabled
            self.camera.position = self.camera_base_position.copy()
        logger.info(f"Audio-reactive camera: {'enabled' if self.audio_camera_enabled else 'disabled'}")
    
    def toggle_bird_tracking(self):
        """Toggle camera bird tracking mode."""
        enabled = self.camera.toggle_tracking()
        if enabled and len(self.bird_instances) > 0:
            self.camera.set_tracked_bird(self.bird_instances[0])
        return enabled
    
    def next_tracked_bird(self):
        """Switch to tracking the next bird."""
        if len(self.bird_instances) == 0:
            return
        
        current_idx = 0
        if self.camera.tracked_bird is not None:
            try:
                current_idx = self.bird_instances.index(self.camera.tracked_bird)
            except ValueError:
                current_idx = 0
        
        next_idx = (current_idx + 1) % len(self.bird_instances)
        self.camera.set_tracked_bird(self.bird_instances[next_idx])
        logger.info(f"Tracking bird {next_idx + 1}/{len(self.bird_instances)}")
    
    def toggle_camera_orbit(self):
        """Toggle camera orbit mode."""
        return self.camera.toggle_orbit()
    
    def adjust_glitch_intensity(self, delta):
        """Adjust glitch intensity."""
        self.config['glitch_intensity'] = max(0.0, min(2.0, self.config.get('glitch_intensity', 1.0) + delta))
        logger.info(f"Glitch intensity: {self.config['glitch_intensity']:.2f}")
    
    def screenshot(self, filename='screenshot.png'):
        """Save screenshot to file."""
        from PIL import Image
        
        data = self.fbo_current.read(components=3)
        img = Image.frombytes('RGB', (self.width, self.height), data)
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        img.save(filename)
        logger.info(f"Screenshot saved: {filename}")
    
    def resize(self, width, height):
        """Handle window resize."""
        self.width = width
        self.height = height
        
        self.fbo_current.release()
        self.fbo_previous.release()
        
        self.fbo_current = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        self.fbo_previous = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)]
        )
        
        self.fbo_current.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo_previous.color_attachments[0].filter = (moderngl.LINEAR, moderngl.LINEAR)
    
    def release(self):
        """Clean up GPU resources."""
        render_mode = self.config.get('render_mode', 'points')
        
        if render_mode == 'cubes':
            # Release cube-specific buffers
            if hasattr(self, 'cube_vbo'):
                self.cube_vbo.release()
            if hasattr(self, 'cube_nbo'):
                self.cube_nbo.release()
            if hasattr(self, 'cube_ibo'):
                self.cube_ibo.release()
            if hasattr(self, 'instance_vbo'):
                self.instance_vbo.release()
        else:
            # Release point-specific buffers
            if hasattr(self, 'vbo'):
                self.vbo.release()
        
        # Release common resources
        if hasattr(self, 'vao'):
            self.vao.release()
        if hasattr(self, 'quad_vbo'):
            self.quad_vbo.release()
        if hasattr(self, 'quad_vao'):
            self.quad_vao.release()
        if hasattr(self, 'point_program'):
            self.point_program.release()
        if hasattr(self, 'post_program'):
            self.post_program.release()
        if hasattr(self, 'fbo_current'):
            self.fbo_current.release()
        if hasattr(self, 'fbo_previous'):
            self.fbo_previous.release()

        if self.bird is not None:
            try:
                self.bird.release()
            except Exception:
                pass
