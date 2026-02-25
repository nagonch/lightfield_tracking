import numpy as np
import torch
import open3d as o3d


def torch_points_to_o3d_tensor_pointcloud(
    points_tensor: torch.Tensor,
    device_str: str = "CPU:0",
) -> o3d.t.geometry.PointCloud:
    """
    points_tensor: torch.Tensor of shape [N, 3], float32/float64
    device_str: "CPU:0" or "CUDA:0"
    """
    if points_tensor.ndim != 2 or points_tensor.shape[1] != 3:
        raise ValueError(
            f"Expected points_tensor shape [N,3], got {tuple(points_tensor.shape)}"
        )

    # Open3D wants float32 typically
    points_cpu = (
        points_tensor.detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous()
        .numpy()
    )
    o3d_device = o3d.core.Device(device_str)

    point_cloud_t = o3d.t.geometry.PointCloud(device=o3d_device)
    point_cloud_t.point["positions"] = o3d.core.Tensor(
        points_cpu, dtype=o3d.core.float32, device=o3d_device
    )
    return point_cloud_t


def estimate_normals_o3d_tensor(
    point_cloud_t: o3d.t.geometry.PointCloud,
    radius: float = 0.05,
    max_nn: int = 30,
    orient_to_camera_location: np.ndarray | None = None,
) -> o3d.t.geometry.PointCloud:
    """
    Estimates normals into point_cloud_t.point["normals"].
    Optionally orients normals towards a camera location (legacy-style behavior).
    """
    point_cloud_t.estimate_normals(radius=radius, max_nn=max_nn)

    # Optional: orient normals consistently toward a viewpoint
    # (Tensor API currently has fewer orientation helpers; easiest is to convert and use legacy)
    if orient_to_camera_location is not None:
        point_cloud_legacy = point_cloud_t.to_legacy()
        point_cloud_legacy.orient_normals_towards_camera_location(
            orient_to_camera_location
        )
        point_cloud_t = o3d.t.geometry.PointCloud.from_legacy(
            point_cloud_legacy, device=point_cloud_t.device
        )

    return point_cloud_t


def make_normals_lineset(pcd, normal_length=0.01, stride=10, color=(1, 0, 0)):
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)

    # --- subsample here ---
    indices = np.arange(0, points.shape[0], stride)
    points = points[indices]
    normals = normals[indices]

    start = points
    end = points + normal_length * normals

    line_points = np.vstack([start, end])
    n = start.shape[0]
    lines = np.column_stack([np.arange(n), np.arange(n) + n])

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (n, 1)))
    return ls


def visualize_pointcloud_with_normals(
    points_tensor: torch.Tensor,
    colors_tensor: torch.Tensor | None = None,
    device_str: str = "CPU:0",  # try "CUDA:0" if your Open3D supports it
    normal_radius: float = 0.05,
    normal_max_nn: int = 30,
    normal_length: float = 0.05,
):
    """
    points_tensor: [N,3] torch tensor
    colors_tensor: optional [N,3] torch tensor in [0,1] float (RGB)
    """
    # 1) Torch -> Open3D tensor point cloud
    point_cloud_t = torch_points_to_o3d_tensor_pointcloud(
        points_tensor, device_str=device_str
    )

    # 2) Estimate normals (tensor API)
    point_cloud_t = estimate_normals_o3d_tensor(
        point_cloud_t,
        radius=normal_radius,
        max_nn=normal_max_nn,
    )

    # 3) Convert to legacy for visualization
    point_cloud_legacy = point_cloud_t.to_legacy()

    # 4) Add colors to the legacy point cloud
    num_points = len(point_cloud_legacy.points)
    if colors_tensor is None:
        # single color for all points (blue-ish)
        uniform_color = np.array([0.2, 0.6, 1.0], dtype=np.float64)
        colors_np = np.tile(uniform_color, (num_points, 1))
    else:
        if colors_tensor.ndim != 2 or colors_tensor.shape[1] != 3:
            raise ValueError(
                f"Expected colors_tensor shape [N,3], got {tuple(colors_tensor.shape)}"
            )
        if colors_tensor.shape[0] != num_points:
            raise ValueError(
                f"colors_tensor N={colors_tensor.shape[0]} does not match points N={num_points}"
            )
        colors_np = (
            colors_tensor.detach()
            .to(dtype=torch.float32, device="cpu")
            .contiguous()
            .numpy()
        )
        colors_np = np.clip(colors_np, 0.0, 1.0).astype(np.float64)

    point_cloud_legacy.colors = o3d.utility.Vector3dVector(colors_np)
    point_cloud_legacy.orient_normals_towards_camera_location(
        camera_location=np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )

    # 5) Build a LineSet to show normals
    normals_lineset = make_normals_lineset(
        point_cloud_legacy,
        normal_length=normal_length,
        # normal_color_rgb=(1.0, 0.2, 0.2),
    )

    # 6) Visualize
    o3d.visualization.draw_geometries([point_cloud_legacy, normals_lineset])


if __name__ == "__main__":
    points = torch.load("pc_0.pt")
    color = torch.load("color_0.pt")

    visualize_pointcloud_with_normals(
        points_tensor=points,
        colors_tensor=color,
        device_str="CPU:0",  # change to "CUDA:0" if available
        normal_radius=0.15,
        normal_max_nn=30,
        normal_length=0.08,
    )
