import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from sh_helpers import fit_sh_coeffs_per_point


def _render_surface_lf(surface_lf, rel_pose=None):
    if rel_pose is None:
        rel_pose = torch.eye(
            4,
            device=surface_lf.values["means"].device,
            dtype=surface_lf.values["means"].dtype,
        )
    image, _, _ = surface_lf.rasterize(rel_pose)
    image = torch.clamp(image.detach().float(), 0.0, 1.0)
    return (image.cpu().numpy() * 255.0).astype("uint8")


def _save_render_comparison(
    original_rgb,
    separated_rgb,
    save_path,
    diffuse_rgb=None,
    view_dependent_rgb=None,
):
    original_image = Image.fromarray(original_rgb)
    separated_image = Image.fromarray(separated_rgb)

    panel_images = [original_image, separated_image]
    panel_labels = ["original", "separated"]
    if diffuse_rgb is not None and view_dependent_rgb is not None:
        panel_images.extend(
            [Image.fromarray(diffuse_rgb), Image.fromarray(view_dependent_rgb)]
        )
        panel_labels.extend(["diffuse (c_d)", "view-dependent"])

    panel_w = max(img.width for img in panel_images)
    panel_h = max(img.height for img in panel_images)

    cols = 2
    rows = (len(panel_images) + cols - 1) // cols
    label_height = 24
    canvas = Image.new(
        "RGB",
        (cols * panel_w, rows * (panel_h + label_height)),
        color=(0, 0, 0),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (img, label) in enumerate(zip(panel_images, panel_labels)):
        row = idx // cols
        col = idx % cols
        x0 = col * panel_w
        y0 = row * (panel_h + label_height)
        canvas.paste(img, (x0, y0 + label_height))
        draw.text((x0 + 8, y0 + 6), label, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    canvas.save(save_path)


def _render_with_temporary_harmonics(surface_lf, harmonics, rel_pose=None):
    old_harmonics = surface_lf.values["harmonics"]
    old_use_relight = surface_lf.use_relight
    try:
        surface_lf.values["harmonics"] = harmonics
        surface_lf.use_relight = False
        return _render_surface_lf(surface_lf, rel_pose=rel_pose)
    finally:
        surface_lf.values["harmonics"] = old_harmonics
        surface_lf.use_relight = old_use_relight


def _masked_nmf_rgb(
    colors: torch.Tensor,
    rank: int = 2,
    iters: int = 10,
    lambda_cd: float = 1e-4,
    eps: float = 1e-8,
):
    """Batched NMF for [points, views, 3] color stacks."""
    colors = colors.clamp(0.0, 1.0)

    n_points, n_views, n_channels = colors.shape
    rank = max(1, min(rank, n_views, n_channels))

    valid_counts = torch.full(
        (n_points, 1),
        float(n_views),
        device=colors.device,
        dtype=colors.dtype,
    )
    mean_color = colors.mean(dim=1)
    mean_color = mean_color.clamp(0.0, 1.0)

    weights = torch.ones(
        (n_points, n_views, rank),
        device=colors.device,
        dtype=colors.dtype,
    )
    weights = weights * (1.0 + 0.05 * torch.rand_like(weights))

    bases = mean_color.unsqueeze(1).expand(n_points, rank, n_channels).clone()
    bases = bases * (1.0 + 0.05 * torch.rand_like(bases))

    for _ in range(iters):
        reconstruction = torch.einsum("pnr,prc->pnc", weights, bases)

        bases_numer = torch.einsum("pnr,pnc->prc", weights, colors)
        bases_denom = torch.einsum("pnr,pnc->prc", weights, reconstruction)
        bases = bases * bases_numer / (bases_denom + lambda_cd).clamp_min(eps)
        bases = bases.clamp_min(eps)

        reconstruction = torch.einsum("pnr,prc->pnc", weights, bases)

        weights_numer = torch.einsum("pnc,prc->pnr", colors, bases)
        weights_denom = torch.einsum("pnc,prc->pnr", reconstruction, bases)
        weights = weights * weights_numer / (weights_denom + lambda_cd).clamp_min(eps)
        weights = weights.clamp_min(eps)

    return weights, bases, mean_color, valid_counts


def reflection_separation(
    surface_lf,
    alpha=0.5,
    lambda_cd=1e-4,
    lambda_alpha=1e-2,
    iters=5,
    save_render_path: str | None = "reflection_separation_compare.png",
    visualize_components: bool = True,
    reflection_viz_gain: float | None = None,
    diffuse_viz_gain: float | None = None,
):
    original_rgb = None
    specular_viz_harmonics = None
    diffuse_viz_harmonics = None
    if save_render_path is not None:
        original_rgb = _render_surface_lf(surface_lf)

    colors = surface_lf.colors.permute(1, 0, 2).contiguous()

    alpha = min(max(float(alpha), 0.0), 1.0)

    # Exact boundary case: C = C_d when alpha == 1, so keep the model unchanged.
    if alpha >= 1.0 - 1e-8:
        if save_render_path is not None:
            diffuse_rgb = None
            view_dep_rgb = None
            if visualize_components:
                zeros = torch.zeros_like(surface_lf.values["harmonics"])
                diffuse_rgb = _render_with_temporary_harmonics(
                    surface_lf,
                    harmonics=surface_lf.values["harmonics"],
                )
                view_dep_rgb = _render_with_temporary_harmonics(
                    surface_lf, harmonics=zeros
                )
            _save_render_comparison(
                original_rgb,
                original_rgb,
                save_render_path,
                diffuse_rgb=diffuse_rgb,
                view_dependent_rgb=view_dep_rgb,
            )
        return surface_lf

    weights, bases, mean_color, valid_counts = _masked_nmf_rgb(
        colors=colors,
        rank=2,
        iters=max(1, iters),
        lambda_cd=lambda_cd,
    )

    weight_mean = weights.mean(dim=1)
    weight_var = ((weights - weight_mean[:, None, :]) ** 2).mean(dim=1)
    const_score = weight_var / (weight_mean**2 + 1e-8)
    diffuse_index = const_score.argmin(dim=-1)

    gather_index = diffuse_index[:, None, None].expand(-1, 1, colors.shape[-1])
    diffuse_basis = bases.gather(1, gather_index).squeeze(1)
    diffuse_weight = weight_mean.gather(1, diffuse_index[:, None]).squeeze(1)
    diffuse_color = diffuse_weight.unsqueeze(-1) * diffuse_basis
    diffuse_color = (1.0 - lambda_alpha) * diffuse_color + lambda_alpha * mean_color
    diffuse_color = diffuse_color.clamp(0.0, 1.0)

    diffuse_colors = diffuse_color[:, None, :].expand_as(colors).contiguous()
    diffuse_contrib = alpha * diffuse_colors
    mix_denom = max(1.0 - alpha, 1e-8)
    specular_colors = (colors - diffuse_contrib) / mix_denom
    specular_colors = torch.clamp(specular_colors, 0.0, 1.0)

    surface_lf.colors = diffuse_contrib.permute(1, 0, 2).contiguous()

    if hasattr(surface_lf, "view_dirs") and hasattr(surface_lf, "surface_normals"):
        diffuse_base_vp = diffuse_colors.permute(1, 0, 2).contiguous()
        diffuse_colors_vp = diffuse_contrib.permute(1, 0, 2).contiguous()
        view_mask_vp = torch.ones(
            diffuse_colors_vp.shape[:2],
            device=diffuse_colors_vp.device,
            dtype=diffuse_colors_vp.dtype,
        )
        specular_contrib_vp = ((1.0 - alpha) * specular_colors).permute(1, 0, 2)

        if diffuse_viz_gain is None:
            diffuse_viz_gain = 1.0 / max(alpha, 1e-3)
        diffuse_viz_gain = max(float(diffuse_viz_gain), 0.0)

        if reflection_viz_gain is None:
            reflection_viz_gain = 1.0 / max(1.0 - alpha, 1e-3)
        reflection_viz_gain = max(float(reflection_viz_gain), 0.0)

        harmonics = fit_sh_coeffs_per_point(
            diffuse_colors_vp.float(),
            surface_lf.view_dirs.float(),
            view_mask_vp.float(),
            max_degree=2,
            lambda_reg=1e-3,
        )
        surface_lf.values["harmonics"] = harmonics.to(
            device=surface_lf.values["harmonics"].device,
            dtype=surface_lf.values["harmonics"].dtype,
        )

        diffuse_viz_vp = torch.clamp(
            diffuse_base_vp * diffuse_viz_gain,
            0.0,
            1.0,
        )
        diffuse_viz_harmonics = fit_sh_coeffs_per_point(
            diffuse_viz_vp.float(),
            surface_lf.view_dirs.float(),
            view_mask_vp.float(),
            max_degree=2,
            lambda_reg=1e-3,
        ).to(
            device=surface_lf.values["harmonics"].device,
            dtype=surface_lf.values["harmonics"].dtype,
        )

        specular_viz_vp = torch.clamp(
            specular_contrib_vp * reflection_viz_gain,
            0.0,
            1.0,
        )
        specular_viz_harmonics = fit_sh_coeffs_per_point(
            specular_viz_vp.float(),
            surface_lf.view_dirs.float(),
            view_mask_vp.float(),
            max_degree=2,
            lambda_reg=1e-3,
        ).to(
            device=surface_lf.values["harmonics"].device,
            dtype=surface_lf.values["harmonics"].dtype,
        )

        normals = surface_lf.surface_normals.to(
            device=colors.device, dtype=colors.dtype
        )
        view_dirs = surface_lf.view_dirs.to(device=colors.device, dtype=colors.dtype)
        normals_rep = normals.unsqueeze(0).expand_as(view_dirs)
        reflected_dirs = F.normalize(
            view_dirs
            - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep,
            dim=-1,
            eps=1e-8,
        )
        surface_lf.environment_map = surface_lf._build_environment_map(
            reflected_dirs=reflected_dirs,
            colors=specular_colors.permute(1, 0, 2).contiguous(),
            valid=view_mask_vp,
            view_dirs=view_dirs,
            normals=normals,
        )

    if save_render_path is not None:
        separated_rgb = _render_surface_lf(surface_lf)
        diffuse_rgb = None
        view_dep_rgb = None
        if visualize_components and specular_viz_harmonics is not None:
            diffuse_rgb = _render_with_temporary_harmonics(
                surface_lf,
                harmonics=diffuse_viz_harmonics,
            )
            view_dep_rgb = _render_with_temporary_harmonics(
                surface_lf,
                harmonics=specular_viz_harmonics,
            )
        _save_render_comparison(
            original_rgb,
            separated_rgb,
            save_render_path,
            diffuse_rgb=diffuse_rgb,
            view_dependent_rgb=view_dep_rgb,
        )

    return surface_lf


if __name__ == "__main__":
    pass
