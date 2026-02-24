import torch
import torch.nn.functional as F

# ImageNet normalization used by DA3 visualization "denormalize" implies this normalize. :contentReference[oaicite:2]{index=2}
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)


# def _upper_bound_resize_nchw(
#     images_nchw: torch.Tensor, process_res: int = 504
# ) -> torch.Tensor:
#     # Matches the idea of "upper_bound_resize": max(H, W) -> process_res, keep aspect.
#     _, _, height, width = images_nchw.shape
#     scale = process_res / float(max(height, width))
#     new_h = max(1, int(round(height * scale)))
#     new_w = max(1, int(round(width * scale)))
#     return F.interpolate(
#         images_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False
#     )


def resize_max_side_and_make_divisible(
    images_nchw: torch.Tensor,
    max_side: int = 504,
    divisor: int = 14,
) -> torch.Tensor:
    # images_nchw: (N,3,H,W)
    _, _, H, W = images_nchw.shape

    scale = max_side / float(max(H, W))
    new_h = max(1, int(round(H * scale)))
    new_w = max(1, int(round(W * scale)))

    # round to nearest divisible (could also floor/ceil; nearest is usually fine)
    new_h = max(divisor, int(round(new_h / divisor)) * divisor)
    new_w = max(divisor, int(round(new_w / divisor)) * divisor)

    return F.interpolate(
        images_nchw, size=(new_h, new_w), mode="bilinear", align_corners=False
    )


@torch.inference_mode()
def da3_run_from_tensors(
    da3_model,  # DepthAnything3 instance
    images_mhw3: torch.Tensor,  # (m, h, w, 3), float32, cuda, range [0,1]
    intrinsics_4x4: torch.Tensor,  # (4,4) or (3,3), float32, cuda
    poses_m44: torch.Tensor,  # (m,4,4), float32, cuda
    poses_are_c2w: bool = False,  # set True if your poses are camera->world
    process_res: int = 504,
    export_feat_layers=(),
    infer_gs: bool = False,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "saddle_balanced",
):
    device = next(da3_model.parameters()).device
    m, orig_h, orig_w, _ = images_mhw3.shape

    # --- images: (m,h,w,3) -> (m,3,h,w) -> resize -> imagenet normalize -> add batch -> (1,m,3,H,W)
    images_mhw3 = images_mhw3.to(device=device, dtype=torch.float32)
    images_mchw = images_mhw3.permute(0, 3, 1, 2).contiguous()
    images_mchw = resize_max_side_and_make_divisible(
        images_mchw, max_side=504, divisor=14
    )
    images_mchw = (images_mchw - _IMAGENET_MEAN) / _IMAGENET_STD
    imgs_bmchw = images_mchw.unsqueeze(0)  # (B=1, N=m, 3, H, W)

    # --- intrinsics: want (B,N,3,3)
    if intrinsics_4x4.shape == (4, 4):
        intrinsics_33 = intrinsics_4x4[:3, :3]
    else:
        intrinsics_33 = intrinsics_4x4
    intrinsics_33 = intrinsics_33.to(device=device, dtype=torch.float32)
    in_t = (
        intrinsics_33.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, poses_m44.shape[0], 3, 3)
        .contiguous()
    )

    # --- extrinsics: want (B,N,4,4) and DA3 normalizes them internally. :contentReference[oaicite:3]{index=3}
    ex = poses_m44.to(device=device, dtype=torch.float32)
    if poses_are_c2w:
        ex = torch.linalg.inv(ex)  # DA3 expects world->camera (w2c) in most pipelines
    ex_t = ex.unsqueeze(0)  # (1,m,4,4)

    # Same normalization DA3 uses in inference() before forward. :contentReference[oaicite:4]{index=4}
    ex_t_norm = da3_model._normalize_extrinsics(ex_t.clone())

    # Call exactly what you wanted: _run_model_forward. :contentReference[oaicite:5]{index=5}
    assert imgs_bmchw.shape[-2] % 14 == 0 and imgs_bmchw.shape[-1] % 14 == 0
    raw_output = da3_model._run_model_forward(
        imgs_bmchw,
        ex_t_norm,
        in_t,
        export_feat_layers=list(export_feat_layers),
        infer_gs=infer_gs,
        use_ray_pose=use_ray_pose,
        ref_view_strategy=ref_view_strategy,
    )
    depth = raw_output["depth"][0]
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),  # [n,1,h',w']
        size=(orig_h, orig_w),
        mode="bilinear",
        align_corners=False,
    )[:, 0]

    return depth


if __name__ == "__main__":
    pass
