from unik3d.models import UniK3D
import numpy as np
from PIL import Image
import torch
import open3d as o3d
from unik3d.utils.camera import Spherical
import cv2

class Unik3DModel:
    def __init__(self, backbone: str = 'vitl', resolution_level: int = 9, interpolation_mode: str = 'bilinear', height: int = 960, width: int = 1920) -> None:
        self.height = height
        self.width = width
        self.device = 'cuda'

        self.hfov = 360 * np.pi / 180.0 / 2
        self.vfov = self.height / self.width * self.hfov
        self.camera = eval("Spherical")(params=torch.tensor([0, 0, 0, 0, width, height, self.hfov, self.vfov]))

        self.model = UniK3D.from_pretrained(f"lpiccinelli/unik3d-{backbone}")
        self.model.resolution_level = resolution_level
        self.model.interpolation_mode = interpolation_mode
        self.model = self.model.to(self.device).eval()

    def estimate_depth(self, rgb_path: str, depth_path: str) -> None:
        image = cv2.resize(cv2.imread(rgb_path), (self.width, self.height))
        image_torch = torch.from_numpy(image).permute(2, 0, 1)

        with torch.no_grad():
            outputs = self.model.infer(rgb=image_torch, camera=self.camera, normalize=True, rays=None)
        
        points = outputs["points"].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        pred_depth = np.linalg.norm(points, axis=1).reshape((self.height, self.width))
        # np.save(depth_path, pred_depth)
        return

def get_uni_sphere_xyz(height, width):
    j, i = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    u = (i+0.5) / width * 2 * np.pi
    v = ((j+0.5) / height - 0.5) * np.pi
    z = -np.sin(v)
    c = np.cos(v)
    y = c * np.sin(u)
    x = c * np.cos(u)
    return np.stack([-x, y, z], -1)

def pcd_from_rgb_depth(rgb, depth):
    height, width = rgb.shape[:2]
    xyz = depth * get_uni_sphere_xyz(height, width)
    xyzrgb = np.concatenate([xyz, rgb/255.], 2).reshape(-1, 6)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:])
    return pcd

# Usage
height, width = 960, 1920
rgb_path = "/home/tarun/motionseg/datasets/custom_dataset/data/18bit/equirectangular/000000.png"
depth_path = "depth.npy"

depth_model = Unik3DModel(height=height, width=width)
depth_model.estimate_depth(rgb_path, depth_path)

rgb = cv2.resize(cv2.imread(rgb_path), (width, height))[:, :, :3][:, :, ::-1]
depth = cv2.resize(np.load(depth_path), (width, height))[..., None]

pcd = pcd_from_rgb_depth(rgb, depth)

o3d.visualization.draw_geometries([pcd])
