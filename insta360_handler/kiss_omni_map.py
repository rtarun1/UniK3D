from unik3d.models import UniK3D
import numpy as np
from pathlib import Path
import open3d as o3d
from tqdm import tqdm


def get_points(filepath):
    """Load a .ply point cloud (RGB optional) and return Nx6 array (xyz + rgb)."""
    pcd = o3d.io.read_point_cloud(filepath)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if colors.shape[0] == 0:
        colors = np.zeros_like(points)
    return np.hstack((points, colors))


def get_transformation_matrix(filepath):
    """Load a 4x4 transformation matrix from a text file."""
    matrix = np.loadtxt(filepath).reshape(4, 4)
    return matrix


def apply_transformation(points, transformation):
    """Apply a 4x4 transformation to Nx3 xyz points, keep RGB if available."""
    points_h = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))  # Nx4
    transformed_xyz = (transformation @ points_h.T).T[:, :3]
    return np.hstack((transformed_xyz, points[:, 3:]))  # Nx6


def main():
    rgb_dir = Path("/home/tarun/OmniSLAM/data/rrc_lab/equirectangular")
    depth_dir = Path("/home/tarun/OmniSLAM/data/rrc_lab/depth")
    pose_dir = Path("/home/tarun/OmniSLAM/data/rrc_lab/poses")  # 4x4 txt per frame
    output_map_path = Path("/home/tarun/OmniSLAM/data/rrc_lab/global_map.ply")

    accumulated_points = []

    num_frames = 100
    for timestamp in tqdm(range(num_frames), desc="Processing frames"):
        ply_path = depth_dir / f"{timestamp:06d}.ply"
        pose_path = pose_dir / f"{timestamp:06d}.txt"

        if not ply_path.exists() or not pose_path.exists():
            continue

        # Load point cloud and transformation
        points = get_points(ply_path)
        transformation = get_transformation_matrix(pose_path)

        # Apply transformation
        transformed_points = apply_transformation(points, transformation)

        # Accumulate
        accumulated_points.append(transformed_points)

    # Merge all points
    if accumulated_points:
        merged_points = np.vstack(accumulated_points)
        # Create Open3D point cloud
        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(merged_points[:, :3])
        final_pcd.colors = o3d.utility.Vector3dVector(merged_points[:, 3:])
        # Save
        o3d.io.write_point_cloud(str(output_map_path), final_pcd)
        print(f"Saved merged map to {output_map_path}")
    else:
        print("No points to save!")


if __name__ == "__main__":
    main()
