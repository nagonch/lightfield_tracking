import torch


def reflection_separation_stat(colors):
    n, d, _ = colors.shape

    # 1. Compute aggregate intensity (luminance) for each direction
    # We use a simple mean across channels to identify specular outliers.[3]
    intensity = colors.mean(dim=-1)  # [n, d]

    # 2. Identify "Specular-Free" samples using the 75th percentile rule.
    # Psychophysical research suggests diffuse information is found in the lower
    # percentiles, typically below the 80th.
    thresholds = torch.quantile(intensity, 0.75, dim=1, keepdim=True)  # [n, 1]
    is_diffuse_mask = intensity <= thresholds  # [n, d]

    # 3. Estimate f_dc (matte background) via masked median.
    # The median of low-intensity samples provides a robust estimate of
    # the invariant body reflection.
    f_dc = torch.zeros((n, 3), device=colors.device, dtype=colors.dtype)
    for i in range(n):
        # Filter samples for function i that are below the percentile threshold
        samples_i = colors[i, is_diffuse_mask[i], :]  # [k_filtered, 3]
        if samples_i.shape[0] > 0:
            f_dc[i] = torch.median(samples_i, dim=0).values
        else:
            # Fallback to minimum if percentile is empty (unlikely with 0.75)
            f_dc[i] = colors[i].min(dim=0).values

    # 4. Estimate shared Alpha using the Intensity Ratio Heuristic.
    # For diffuse reflection, the ratio max/(max-min) is independent of
    # geometry. We use the relationship between the estimated
    # baseline (B = alpha * f_dc) and the total signal.
    min_vals = intensity.min(dim=1).values  # [n]
    max_vals = intensity.max(dim=1).values  # [n]

    # Alpha represents the weight of the DC component in the mixture.
    # Per-function estimate: alpha_i ~ min_intensity / max_intensity
    per_function_alphas = min_vals / (max_vals + 1e-8)

    # Since alpha is known to be shared across all functions, we take the mean.
    global_alpha = torch.mean(per_function_alphas).item()
    global_alpha = max(0.0, min(1.0, global_alpha))  # Clamp to [0, 1]

    # 5. Isolate f_spec via the model: f(d) = alpha * f_dc + (1 - alpha) * f_spec(d)
    # f_spec(d) = (f(d) - alpha * f_dc) / (1 - alpha)
    weighted_dc = global_alpha * f_dc.unsqueeze(1)  # [n, 1, 3]
    one_minus_alpha = 1.0 - global_alpha

    # Avoid division by zero if alpha is 1 (purely diffuse material)
    if one_minus_alpha < 1e-6:
        f_spec = torch.zeros_like(colors)
    else:
        f_spec = (colors - weighted_dc) / one_minus_alpha

    # Apply positivity constraint [6]
    f_spec = torch.clamp(f_spec, min=0.0)

    return global_alpha, f_dc, f_spec


def reflection_separation(surface_lf):
    colors = surface_lf.colors
    view_dirs = surface_lf.view_dirs
    valid = surface_lf.valid.all(axis=0)
    colors = colors[:, valid].permute(1, 0, 2)
    global_alpha, f_dc, f_spec = reflection_separation_stat(colors)
    view_dirs = view_dirs[:, valid].permute(1, 0, 2)
    harmonics = surface_lf.values["harmonics"]

    return surface_lf


if __name__ == "__main__":
    pass
