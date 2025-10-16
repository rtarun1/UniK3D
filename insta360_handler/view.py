import open3d as o3d

ply_file = "/home/tarun/OmniSLAM/data/rrc_lab/depth/000055.ply"
pcd = o3d.io.read_point_cloud(ply_file)
o3d.visualization.draw_geometries([pcd])
