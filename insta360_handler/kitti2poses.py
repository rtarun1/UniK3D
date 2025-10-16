import numpy as np
import os

def split_kitti_poses(pose_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with open(pose_file, 'r') as f:
        for i, line in enumerate(f):
            numbers = list(map(float, line.strip().split()))
            if len(numbers) != 12:
                print(f"Skipping malformed line {i}")
                continue

            # Construct 4x4 matrix
            T = np.eye(4)
            T[:3, :4] = np.array(numbers).reshape(3, 4)

            # Save to a file like 000000.txt, 000001.txt, ...
            out_path = os.path.join(output_dir, f"{i:06d}.txt")
            np.savetxt(out_path, T, fmt="%.18e")
            print(f"Saved: {out_path}")

# Example usage
pose_file = "/home/tarun/OmniSLAM/data/rrc_lab/pcd_poses_kitti.txt"
output_dir = "/home/tarun/OmniSLAM/data/rrc_lab/poses"

split_kitti_poses(pose_file, output_dir)
