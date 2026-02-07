import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_rainforest_pointcloud(num_points=100000, output_format='npy'):
    """
    Generate a procedural rainforest-like point cloud for testing.
    Creates a layered organic structure with varying density.
    """
    logger.info(f"Generating {num_points} point rainforest cloud...")
    
    points = []
    colors = []
    
    np.random.seed(42)
    
    ground_points = int(num_points * 0.2)
    for _ in range(ground_points):
        x = np.random.uniform(-10, 10)
        z = np.random.uniform(-10, 10)
        y = np.random.uniform(-0.5, 0.5)
        
        points.append([x, y, z])
        colors.append([0.2 + np.random.rand() * 0.2, 0.3 + np.random.rand() * 0.3, 0.1])
    
    trunk_points = int(num_points * 0.15)
    num_trunks = 20
    for _ in range(trunk_points):
        trunk_id = np.random.randint(0, num_trunks)
        trunk_x = np.random.uniform(-8, 8)
        trunk_z = np.random.uniform(-8, 8)
        
        height = np.random.uniform(0, 15)
        radius = 0.3 + np.random.rand() * 0.2
        angle = np.random.uniform(0, 2 * np.pi)
        
        x = trunk_x + np.cos(angle) * radius
        z = trunk_z + np.sin(angle) * radius
        y = height
        
        points.append([x, y, z])
        brown = 0.3 + np.random.rand() * 0.2
        colors.append([brown, brown * 0.6, brown * 0.3])
    
    canopy_points = int(num_points * 0.65)
    for _ in range(canopy_points):
        center_x = np.random.uniform(-8, 8)
        center_z = np.random.uniform(-8, 8)
        center_y = np.random.uniform(8, 20)
        
        radius = np.random.uniform(2, 5)
        
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, radius)
        
        x = center_x + r * np.sin(phi) * np.cos(theta)
        y = center_y + r * np.cos(phi)
        z = center_z + r * np.sin(phi) * np.sin(theta)
        
        points.append([x, y, z])
        
        green = 0.3 + np.random.rand() * 0.5
        colors.append([0.1 + np.random.rand() * 0.2, green, 0.1 + np.random.rand() * 0.2])
    
    positions = np.array(points, dtype=np.float32)
    colors_array = np.array(colors, dtype=np.float32)
    
    logger.info(f"Generated {len(positions)} points")
    logger.info(f"Bounds: X[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}], "
                f"Y[{positions[:,1].min():.2f}, {positions[:,1].max():.2f}], "
                f"Z[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    
    output_dir = Path('./pointclouds')
    output_dir.mkdir(exist_ok=True)
    
    if output_format == 'npy':
        data = np.hstack([positions, colors_array * 255])
        output_file = output_dir / 'rainforest_0.npy'
        np.save(output_file, data)
        logger.info(f"Saved to: {output_file}")
    
    elif output_format == 'xyz':
        data = np.hstack([positions, colors_array * 255])
        output_file = output_dir / 'rainforest_0.xyz'
        np.savetxt(output_file, data, fmt='%.6f')
        logger.info(f"Saved to: {output_file}")
    
    elif output_format == 'ply':
        output_file = output_dir / 'rainforest_0.ply'
        with open(output_file, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(positions)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            
            for i in range(len(positions)):
                pos = positions[i]
                col = (colors_array[i] * 255).astype(np.uint8)
                f.write(f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} {col[0]} {col[1]} {col[2]}\n")
        
        logger.info(f"Saved to: {output_file}")
    
    return positions, colors_array


if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Rainforest Point Cloud Generator")
    logger.info("="*60)
    
    generate_rainforest_pointcloud(num_points=100000, output_format='npy')
    
    logger.info("\nDone! You can now run: python main.py")
