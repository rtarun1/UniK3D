from unik3d.models import UniK3D
import numpy as np
from PIL import Image
import torch
import open3d as o3d
from unik3d.utils.camera import Spherical
import cv2
from pathlib import Path
from tqdm import tqdm

def get_xyz_sphere(height, width):
    j, i = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    u = (i + 0.5) / width * 2 * np.pi
    v = ((j + 0.5) / height - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    return np.stack([-x, y, z], -1)

def estimate_depth(rgb_path, depth_path, model, camera, height=960, width=1920):


    image = cv2.imread(rgb_path)
    image = cv2.resize(image, (width, height))
    image_torch = torch.from_numpy(image).permute(2, 0, 1).to(model.device)
    
    with torch.no_grad():
        outputs = model.infer(rgb=image_torch, camera=camera, normalize=True, rays=None)
    
    points = outputs["points"].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
    pred_depth = np.linalg.norm(points, axis=1).reshape((height, width))
    
    xyz = pred_depth[..., None] * get_xyz_sphere(height, width)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    xyzrgb = np.concatenate([xyz, rgb / 255.0], axis=2).reshape(-1, 6)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])

    Path(depth_path).parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(depth_path, pcd, write_ascii=False) 
    
    return xyz

def main():
    device = 'cuda'
    height, width = 960, 1920

    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device).eval()
    model.resolution_level = 9
    model.interpolation_mode = 'bilinear'
    
    hfov = np.pi  
    vfov = height / width * hfov
    camera = Spherical(params=torch.tensor([0, 0, 0, 0, width, height, hfov, vfov]))

    rgb_dir = Path("/home/tarun/OmniSLAM/data/rrc_lab/equirectangular")
    depth_dir = Path("/home/tarun/OmniSLAM/data/rrc_lab/depth")
    depth_dir.mkdir(parents=True, exist_ok=True)

    num_frames = 1696
    for timestamp in tqdm(range(num_frames), desc="Processing frames"):
        rgb_path = rgb_dir / f"{timestamp:06d}.png"
        depth_path = depth_dir / f"{timestamp:06d}.ply"
        estimate_depth(str(rgb_path), str(depth_path), model, camera, height, width)

if __name__ == "__main__":
    main()