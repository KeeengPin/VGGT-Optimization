import torch
import numpy as np
import open3d as o3d

# Load the data and images
data = torch.load(r"Project\eth3d-pointcloud-subset\bridge\vggt_random_pt_cloud.pt", map_location=torch.device('cpu'))  # Tensor, not dict
images = torch.load(r"Project\eth3d-pointcloud-subset\bridge\random_sample_images.pt", map_location=torch.device('cpu'))  # Tensor

# Squeeze batch dimension from world_points tensor
world_points = data.squeeze(0)  # shape: [3, 518, 518, 3]

print(f"world_points shape: {world_points.shape}")
print(f"images shape: {images.shape}")

# images shape: [3, 3, 518, 518]
# Permute channels last to [3, 518, 518, 3]
images = images.permute(0, 2, 3, 1)  # [3, 518, 518, 3]
print(images.shape)
# Convert to numpy
world_points_np = world_points.numpy()
images_np = images.numpy()

# Check if colors are in [0,1] or [0,255]
if images_np.max() > 1.0:
    print("Normalizing colors to [0,1]")
    images_np = images_np / 255.0
else:
    print("Colors already normalized")

all_points = []
all_colors = []

for i in range(world_points_np.shape[0]):  # 3 views
    pts = world_points_np[i].reshape(-1, 3)
    cols = images_np[i].reshape(-1, 3)

    # Filter valid points
    valid_mask = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-5)
    pts = pts[valid_mask]
    cols = cols[valid_mask]

    all_points.append(pts)
    all_colors.append(cols)

# Concatenate all points and colors
points = np.concatenate(all_points, axis=0)
colors = np.concatenate(all_colors, axis=0)

# Create Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# Visualize
# o3d.visualization.draw_geometries([pcd], window_name="Colored Point Cloud")
# target coordinate to look at
target = np.array([0.0, 0.0, 0.0])  # change to your coordinate

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Colored Point Cloud")
vis.add_geometry(pcd)

# Add coordinate axes: one at the world origin and one at the target point
# axes_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0.0, 0.0, 0.0])
# axes_target = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=target.tolist())
# vis.add_geometry(axes_origin)
# vis.add_geometry(axes_target)

ctr = vis.get_view_control()

# Compute a sensible camera offset so the camera is not exactly at the lookat point.
# We place the camera a bit along +Z relative to the target; front must point toward the target.
bbox_diag = np.linalg.norm(np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
if bbox_diag == 0:
    bbox_diag = 1.0
camera_offset = np.array([0.0, 0.0, bbox_diag * 1.2])  # camera placed along +Z
cam_pos = target + camera_offset

front = (target - cam_pos)
front = front / np.linalg.norm(front)  # normalized view direction toward the target
up = np.array([0.0, -1.0, 0.0])          # world up vector

ctr.set_lookat(target.tolist())     # center camera on `target`
ctr.set_front(front.tolist())       # direction camera looks along (toward target)
ctr.set_up(up.tolist())             # camera up vector
ctr.set_zoom(0.1)                   # adjust zoom (smaller -> farther)

vis.poll_events()
vis.update_renderer()
vis.run()
vis.destroy_window()
