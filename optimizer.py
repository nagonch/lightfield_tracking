from surface_lf import SurfaceLF, SurfaceLFRig
import torch

if __name__ == "__main__":
    K = torch.load("K.pt")
    poses = torch.load("poses_4x4.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=(1280, 720),
    )
    for i in range(5):
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pc_{i:04d}.pt"),
            images=torch.load(f"images_{i:04d}.pt"),
        )
        values = surface_lf.transform(torch.eye(4).cuda())
        image, depth = surface_lf.rasterize(values)
