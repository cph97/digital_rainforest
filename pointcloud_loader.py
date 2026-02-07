import numpy as np
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class PointCloudLoader:
    """Auto-detect and load point cloud from .ply, .xyz, or .npy formats."""
    
    def __init__(self, folder, name, max_points=500000, normalize=False, center=False):
        self.folder = Path(folder)
        self.name = name
        self.max_points = max_points
        self.normalize = normalize
        self.center = center
        self.positions = None
        self.colors = None
        self.bounds = None
        self.centroid = None
        self.original_centroid = None
        self.bounding_radius = None
        self.scale_factor = 1.0
        
    def load(self):
        """Auto-detect format and load point cloud."""
        extensions = ['.ply', '.xyz', '.npy']
        found_file = None
        
        for ext in extensions:
            candidate = self.folder / f"{self.name}{ext}"
            if candidate.exists():
                found_file = candidate
                break
        
        if found_file is None:
            raise FileNotFoundError(
                f"Point cloud '{self.name}' not found in {self.folder} "
                f"with extensions {extensions}"
            )
        
        logger.info(f"Loading point cloud from: {found_file}")
        
        if found_file.suffix == '.ply':
            self._load_ply(found_file)
        elif found_file.suffix == '.xyz':
            self._load_xyz(found_file)
        elif found_file.suffix == '.npy':
            self._load_npy(found_file)
        
        self._downsample_if_needed()
        
        # DIAGNOSTIC: Validate raw data before any transforms
        logger.info("=== RAW DATA VALIDATION ===")
        logger.info(f"Position dtype: {self.positions.dtype}")
        logger.info(f"Position shape: {self.positions.shape}")
        logger.info(f"Position range: X[{self.positions[:,0].min():.2f}, {self.positions[:,0].max():.2f}] "
                   f"Y[{self.positions[:,1].min():.2f}, {self.positions[:,1].max():.2f}] "
                   f"Z[{self.positions[:,2].min():.2f}, {self.positions[:,2].max():.2f}]")
        logger.info(f"Position mean: [{self.positions[:,0].mean():.2f}, {self.positions[:,1].mean():.2f}, {self.positions[:,2].mean():.2f}]")
        
        if self.colors is not None:
            logger.info(f"Color dtype: {self.colors.dtype}")
            logger.info(f"Color range: R[{self.colors[:,0].min():.3f}, {self.colors[:,0].max():.3f}] "
                       f"G[{self.colors[:,1].min():.3f}, {self.colors[:,1].max():.3f}] "
                       f"B[{self.colors[:,2].min():.3f}, {self.colors[:,2].max():.3f}]")
        
        # Fix orientation: rotate point cloud so Y is up (trees stand upright)
        # Original appears to have Z as up, so swap Y and Z and negate
        temp_positions = self.positions.copy()
        self.positions[:, 0] = temp_positions[:, 0]  # X stays X
        self.positions[:, 1] = temp_positions[:, 2]  # Z becomes Y (up)
        self.positions[:, 2] = -temp_positions[:, 1]  # -Y becomes Z (forward)
        
        logger.info("=== AFTER AXIS ROTATION ===")
        logger.info(f"Position range: X[{self.positions[:,0].min():.2f}, {self.positions[:,0].max():.2f}] "
                   f"Y[{self.positions[:,1].min():.2f}, {self.positions[:,1].max():.2f}] "
                   f"Z[{self.positions[:,2].min():.2f}, {self.positions[:,2].max():.2f}]")
        
        # Compute statistics before normalization
        self._compute_statistics()
        self.original_centroid = self.centroid.copy()
        
        # Apply centering and normalization if requested
        if self.center or self.normalize:
            self._apply_normalization()
        
        logger.info("=== FINAL STATISTICS ===")
        logger.info(f"Loaded {len(self.positions)} points")
        logger.info(f"Bounds: {self.bounds}")
        logger.info(f"Centroid: {self.centroid}")
        logger.info(f"Bounding radius: {self.bounding_radius:.2f}")
        logger.info(f"Scale factor: {self.scale_factor:.4f}")
        
        return self.positions, self.colors
    
    def _load_ply(self, filepath):
        """Parse PLY format (ASCII or binary)."""
        # Read header first to determine format
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if 'end_header' in line:
                    break
        
        vertex_count = 0
        has_color = False
        is_binary = False
        
        for line in header_lines:
            if 'format binary' in line:
                is_binary = True
            if 'element vertex' in line:
                vertex_count = int(line.split()[-1])
            if 'property' in line and ('red' in line or 'diffuse_red' in line or 'uchar red' in line):
                has_color = True
        
        if is_binary:
            logger.info("Loading binary PLY file")
            self._load_ply_binary(filepath, vertex_count, has_color, len(header_lines))
        else:
            logger.info("Loading ASCII PLY file")
            self._load_ply_ascii(filepath, vertex_count, has_color, len(header_lines))
    
    def _load_ply_ascii(self, filepath, vertex_count, has_color, header_line_count):
        """Load ASCII PLY format."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        positions = []
        colors = []
        
        for line in lines[header_line_count:header_line_count + vertex_count]:
            parts = line.strip().split()
            if len(parts) >= 3:
                positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if has_color and len(parts) >= 6:
                    colors.append([float(parts[3])/255.0, float(parts[4])/255.0, float(parts[5])/255.0])
        
        self.positions = np.array(positions, dtype=np.float32)
        
        if colors and len(colors) == len(positions):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            logger.info("No colors in PLY, generating procedural colors")
            self._generate_procedural_colors()
    
    def _load_ply_binary(self, filepath, vertex_count, has_color, header_line_count):
        """Load binary PLY format."""
        import struct
        
        with open(filepath, 'rb') as f:
            # Skip header
            for _ in range(header_line_count):
                f.readline()
            
            positions = []
            colors = []
            
            # Binary format: typically float x, y, z, uchar r, g, b
            if has_color:
                for _ in range(vertex_count):
                    data = f.read(15)  # 3 floats (12 bytes) + 3 uchars (3 bytes)
                    if len(data) < 15:
                        break
                    x, y, z = struct.unpack('fff', data[:12])
                    r, g, b = struct.unpack('BBB', data[12:15])
                    positions.append([x, y, z])
                    colors.append([r/255.0, g/255.0, b/255.0])
            else:
                for _ in range(vertex_count):
                    data = f.read(12)  # 3 floats
                    if len(data) < 12:
                        break
                    x, y, z = struct.unpack('fff', data)
                    positions.append([x, y, z])
        
        self.positions = np.array(positions, dtype=np.float32)
        
        if colors and len(colors) == len(positions):
            self.colors = np.array(colors, dtype=np.float32)
        else:
            logger.info("No colors in binary PLY, generating procedural colors")
            self._generate_procedural_colors()
    
    def _load_xyz(self, filepath):
        """Load XYZ format (space-separated x y z per line)."""
        data = np.loadtxt(filepath, dtype=np.float32)
        
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        
        self.positions = data[:, :3]
        
        if data.shape[1] >= 6:
            self.colors = data[:, 3:6] / 255.0
        else:
            logger.info("No colors in XYZ, generating procedural colors")
            self._generate_procedural_colors()
    
    def _load_npy(self, filepath):
        """Load NPY format (Nx3 or Nx6 array)."""
        data = np.load(filepath).astype(np.float32)
        
        if data.ndim == 1:
            data = data.reshape(-1, 3)
        
        self.positions = data[:, :3]
        
        if data.shape[1] >= 6:
            self.colors = data[:, 3:6]
            if self.colors.max() > 1.5:
                self.colors /= 255.0
        else:
            logger.info("No colors in NPY, generating procedural colors")
            self._generate_procedural_colors()
    
    def _generate_procedural_colors(self):
        """Generate colors based on height (Y) and noise."""
        if self.positions is None or len(self.positions) == 0:
            return
        
        y_vals = self.positions[:, 1]
        y_min, y_max = y_vals.min(), y_vals.max()
        y_norm = (y_vals - y_min) / (y_max - y_min + 1e-6)
        
        noise = np.random.rand(len(self.positions)) * 0.2
        
        r = np.clip(0.2 + y_norm * 0.5 + noise, 0, 1)
        g = np.clip(0.4 + y_norm * 0.4 + noise, 0, 1)
        b = np.clip(0.1 + (1 - y_norm) * 0.3 + noise, 0, 1)
        
        self.colors = np.stack([r, g, b], axis=1).astype(np.float32)
    
    def _downsample_if_needed(self):
        """Downsample if point count exceeds max_points."""
        if self.max_points is None or len(self.positions) <= self.max_points:
            return
        
        logger.warning(f"Downsampling from {len(self.positions)} to {self.max_points} points")
        
        indices = np.random.choice(len(self.positions), self.max_points, replace=False)
        self.positions = self.positions[indices]
        if self.colors is not None:
            self.colors = self.colors[indices]
    
    def _compute_statistics(self):
        """Compute bounding box, centroid, and radius."""
        self.centroid = self.positions.mean(axis=0)
        self.bounds = {
            'min': self.positions.min(axis=0),
            'max': self.positions.max(axis=0),
            'size': self.positions.max(axis=0) - self.positions.min(axis=0)
        }
        
        # Compute bounding radius (max distance from centroid)
        centered = self.positions - self.centroid
        distances = np.linalg.norm(centered, axis=1)
        self.bounding_radius = distances.max()
    
    def _apply_normalization(self):
        """Apply centering and/or scaling normalization."""
        logger.info("=== APPLYING NORMALIZATION ===")
        
        if self.center:
            logger.info(f"Centering point cloud (subtracting centroid: {self.centroid})")
            self.positions -= self.centroid
            self.centroid = np.zeros(3, dtype=np.float32)
        
        if self.normalize:
            # Normalize to unit sphere based on bounding radius
            if self.bounding_radius > 0:
                self.scale_factor = 1.0 / self.bounding_radius
                logger.info(f"Normalizing to unit sphere (radius: {self.bounding_radius:.2f}, scale: {self.scale_factor:.4f})")
                self.positions *= self.scale_factor
                self.bounding_radius = 1.0
            else:
                logger.warning("Bounding radius is zero, skipping normalization")
        
        # Recompute statistics after normalization
        self._compute_statistics()
    
    def get_statistics(self):
        """Return computed statistics for sonification."""
        return {
            'bounds': self.bounds,
            'centroid': self.centroid,
            'count': len(self.positions),
            'height_range': (self.positions[:, 1].min(), self.positions[:, 1].max())
        }
