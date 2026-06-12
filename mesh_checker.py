from src.dataset import LFDataset
from src.utilities import Visualizer, backproject_depth_to_pointcloud
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np

if __name__ == "__main__":
    dataset = LFDataset(
        "/home/ngoncharov/SpecTrack_dataset/cube_0.0/tomato_soup_can_yalehand0"
    )
    frame0 = dataset[0]
    mesh = dataset.get_mesh()

    depth0 = frame0["depth"].cuda()
    image0 = frame0["LF"][2, 2].cuda()
    mask = frame0["masks"][2, 2].cuda()
    inds = (mask == 1).cuda()
    pc = backproject_depth_to_pointcloud(
        None, depths=depth0, camera_matrix=dataset.camera_matrix.cuda()
    )
    colors = image0.reshape(-1, 3)
    gt_pose = frame0["object_pose"].cpu().numpy()
    T = np.eye(4)
    T[:3, :3] = R.from_euler("xyz", [270, 0, 0], degrees=True).as_matrix()
    mesh.apply_transform(T)

    mesh.apply_transform(trimesh.transformations.scale_matrix(1.5))
    mesh.apply_transform(gt_pose)
    vis = Visualizer()
    vis.add_point_cloud("yo", pc.cpu().numpy(), colors.cpu().numpy())
    vis.add_mesh("mesh", mesh)
    vis.run()
