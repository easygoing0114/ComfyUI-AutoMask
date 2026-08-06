import os
import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from pymatting import estimate_alpha_cf, fix_trimap
import folder_paths


# ============================================================================
# MaskRefine
# ============================================================================

DEFAULT_BLACKPOINT = 0.01
DEFAULT_WHITEPOINT = 0.99
DEFAULT_MAX_ITERATIONS = 1000


class MaskRefine:
    """
    Node that refines an alpha matte from an image and a mask, with
    preblur as the only configurable setting.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "preblur": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "refine"
    CATEGORY = "Image-Filters/mask"

    def refine(self, image, mask, preblur):
        d = preblur * 2 + 1

        i_dup = image.cpu().numpy().astype(np.float64)
        m_dup = mask.cpu().numpy().astype(np.float64)

        # If the mask batch size differs from the image batch size, align it
        if m_dup.shape[0] < i_dup.shape[0]:
            reps = i_dup.shape[0] // m_dup.shape[0]
            m_dup = np.tile(m_dup, (reps, 1, 1))
        m_dup = m_dup[: i_dup.shape[0]]

        out = m_dup.copy()

        for index, img in enumerate(i_dup):
            trimap = m_dup[index]

            if preblur > 0:
                trimap = cv2.GaussianBlur(trimap, (d, d), 0)

            trimap = fix_trimap(trimap, DEFAULT_BLACKPOINT, DEFAULT_WHITEPOINT)

            alpha = estimate_alpha_cf(
                img,
                trimap,
                laplacian_kwargs={"epsilon": 1e-6},
                cg_kwargs={"maxiter": DEFAULT_MAX_ITERATIONS},
            )

            out[index] = alpha

        return (torch.from_numpy(out.astype(np.float32)),)


# ============================================================================
# AutoMaskThreshold
# ============================================================================

def _mask_to_hist(mask_np: np.ndarray, bins: int = 256):
    values = mask_np.reshape(-1).astype(np.float32)
    hist, bin_edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    return hist, bin_edges


def _strip_leading_zero_bin(hist: np.ndarray, bin_edges: np.ndarray):
    if len(hist) <= 1:
        return hist, bin_edges
    trimmed_hist = hist[1:]
    trimmed_edges = bin_edges[1:]
    return trimmed_hist, trimmed_edges


def _raw_local_minima(hist: np.ndarray):
    n = len(hist)
    minima_idx = []
    i = 1
    while i < n - 1:
        if hist[i] < hist[i - 1]:
            j = i
            while j < n - 1 and hist[j + 1] == hist[i]:
                j += 1
            if j < n - 1 and hist[j + 1] > hist[i]:
                center = (i + j) // 2
                minima_idx.append(center)
                i = j + 1
                continue
        i += 1
    return minima_idx


def _topographic_prominence_for_minimum(hist: np.ndarray, idx: int):
    n = len(hist)
    base = hist[idx]

    left_peak = hist[idx]
    for i in range(idx - 1, -1, -1):
        if hist[i] > left_peak:
            left_peak = hist[i]
        if hist[i] < base:
            break

    right_peak = hist[idx]
    for i in range(idx + 1, n):
        if hist[i] > right_peak:
            right_peak = hist[i]
        if hist[i] < base:
            break

    prominence = min(left_peak, right_peak) - base
    return prominence


def _find_peak_regions_by_level_cutoff(hist: np.ndarray, n_peaks_target: int = 5, max_iter: int = 60):
    n = len(hist)
    if n == 0:
        return [], 0.0

    hist_max = float(hist.max())
    if hist_max <= 0:
        return [], 0.0

    def regions_at_level(level):
        regions = []
        in_region = False
        start = 0
        for i in range(n):
            above = hist[i] > level
            if above and not in_region:
                in_region = True
                start = i
            elif not above and in_region:
                in_region = False
                regions.append((start, i - 1))
        if in_region:
            regions.append((start, n - 1))
        return regions

    lo, hi = 0.0, hist_max * 0.999
    best_regions = regions_at_level(lo)
    best_level = lo

    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        regions = regions_at_level(mid)
        count = len(regions)
        if count <= n_peaks_target and count > 0:
            best_regions = regions
            best_level = mid
            hi = mid
        elif count > n_peaks_target:
            lo = mid
        else:
            hi = mid

    candidates = []
    for frac in np.linspace(0.0, 0.999, 200):
        level = hist_max * frac
        regions = regions_at_level(level)
        if len(regions) > 0:
            candidates.append((abs(len(regions) - n_peaks_target), level, regions))

    if candidates:
        candidates.sort(key=lambda c: (c[0], c[1]))
        best_diff = abs(len(best_regions) - n_peaks_target)
        top_diff, top_level, top_regions = candidates[0]
        if top_diff < best_diff or (top_diff == best_diff and top_level < best_level):
            best_regions, best_level = top_regions, top_level

    return best_regions, best_level


def _find_valley_boundary_for_region(hist: np.ndarray, region, prev_region_end):
    start, end = region
    search_from = 0 if prev_region_end is None else prev_region_end + 1
    search_to = start

    if search_to <= search_from:
        return start

    sub_hist = hist[search_from:search_to + 1]
    if len(sub_hist) < 3:
        return start

    raw_minima_local = _raw_local_minima(sub_hist)
    if not raw_minima_local:
        return start

    scored = []
    for idx in raw_minima_local:
        prom = _topographic_prominence_for_minimum(sub_hist, idx)
        scored.append((prom, idx))
    scored.sort(key=lambda x: (-x[0], x[1]))
    best_local_idx = scored[0][1]

    return search_from + best_local_idx


def _find_minor_minima_in_largest_peak(hist: np.ndarray, region, n_minima: int = 3):
    start, end = region
    if end <= start + 1:
        return []

    sub_hist = hist[start:end + 1]

    raw_minima_local = _raw_local_minima(sub_hist)
    scored = [(_topographic_prominence_for_minimum(sub_hist, idx), idx) for idx in raw_minima_local]
    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen_local = sorted(idx for _, idx in scored[:n_minima])
    chosen_global = [start + idx for idx in chosen_local]

    if len(chosen_global) < n_minima:
        fallback = _equally_spaced_fallback(start, end, n_minima)
        chosen_global = sorted(set(chosen_global) | set(fallback))

    return chosen_global[:n_minima]


def _equally_spaced_fallback(start, end, n_minima):
    span = end - start
    if span <= 0:
        return []
    positions = []
    for k in range(1, n_minima + 1):
        pos = start + int(round(span * k / (n_minima + 1)))
        pos = max(start + 1, min(end - 1, pos))
        positions.append(pos)
    return sorted(set(positions))


def _get_font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _blend(fg, bg, alpha):
    """Alpha-blend fg over bg (both RGB tuples), alpha in [0, 1]."""
    return tuple(int(round(f * alpha + b * (1.0 - alpha))) for f, b in zip(fg, bg))


def _draw_dashed_vline(draw, x, y0, y1, color, width, dash, gap):
    y = y0
    while y < y1:
        y_end = min(y + dash, y1)
        draw.line([(x, y), (x, y_end)], fill=color, width=width)
        y = y_end + gap


def _draw_vertical_text(img, draw, text, font, color, cx, cy):
    """Draws text rotated 90 degrees (bottom-to-top), centered at (cx, cy)."""
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    label_img = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((-bbox[0] + 2, -bbox[1] + 2), text, font=font, fill=color + (255,))
    rotated = label_img.rotate(90, expand=True)
    paste_x = cx - rotated.width // 2
    paste_y = cy - rotated.height // 2
    img.paste(rotated, (paste_x, paste_y), rotated)


def _render_histogram_image(hist, bin_edges, minima_idx, threshold_values, selected_idx=None):
    SILVER_RATIO = 1.41421356
    width_px = 1200
    height_px = int(width_px / SILVER_RATIO)

    bg_color = (255, 255, 250)
    text_color = (54, 69, 79)
    grid_color = (54, 69, 79)
    bar_color = (135, 206, 235)
    green = (76, 187, 23)
    red = (215, 0, 64)

    pad_top = 104
    pad_bottom = 104
    pad_left = 112
    pad_right = 60

    plot_w = width_px - pad_left - pad_right
    plot_h = height_px - pad_top - pad_bottom
    plot_x0 = pad_left
    plot_y0 = pad_top
    plot_x1 = pad_left + plot_w
    plot_y1 = pad_top + plot_h

    img = Image.new("RGB", (width_px, height_px), color=bg_color)
    draw = ImageDraw.Draw(img)

    font_label = _get_font(32)
    font_tick = _get_font(24)
    font_annot = _get_font(24)

    hist_max = hist.max() if hist.max() > 0 else 1.0
    hist_norm = hist.astype(np.float64) / hist_max * 100.0

    centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    x_min = float(bin_edges[0])
    x_max = 1.0

    def x_to_px(x):
        frac = (x - x_min) / (x_max - x_min) if x_max > x_min else 0.0
        return plot_x0 + frac * plot_w

    def y_to_px(y):
        frac = y / 100.0
        return plot_y1 - frac * plot_h

    y_ticks = [0, 25, 50, 75, 100]
    for yt in y_ticks:
        py = y_to_px(yt)
        draw.line([(plot_x0, py), (plot_x1, py)], fill=_blend(grid_color, bg_color, 0.25), width=1)

    n_xticks = 8
    x_ticks = list(np.linspace(x_min, x_max, n_xticks))
    for xt in x_ticks:
        px = x_to_px(xt)
        draw.line([(px, plot_y0), (px, plot_y1)], fill=_blend(grid_color, bg_color, 0.25), width=1)

    draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], outline=_blend((0, 0, 0), bg_color, 0.4), width=2)

    bin_width_data = bin_edges[1] - bin_edges[0]
    bar_px_w = max(1.0, (bin_width_data / (x_max - x_min)) * plot_w)
    bar_fill = _blend(bar_color, bg_color, 0.75)
    for c, h in zip(centers, hist_norm):
        if h <= 0:
            continue
        left = x_to_px(c) - bar_px_w / 2.0
        right = x_to_px(c) + bar_px_w / 2.0
        top = y_to_px(h)
        bottom = y_to_px(0)
        draw.rectangle([left, top, right, bottom], fill=bar_fill)

    for k, idx in enumerate(minima_idx):
        x = centers[idx]
        is_selected = (selected_idx is not None and idx == selected_idx)
        color = red if is_selected else green
        px = x_to_px(x)
        width = 3 if is_selected else 2
        _draw_dashed_vline(draw, px, plot_y0, plot_y1, color, width, dash=8, gap=5)

    for xt in x_ticks:
        px = x_to_px(xt)
        label = f"{xt * 255:.0f}"
        tw = draw.textlength(label, font=font_tick)
        draw.text((px - tw / 2, plot_y1 + 8), label, font=font_tick, fill=text_color)

    for yt in y_ticks:
        py = y_to_px(yt)
        label = f"{yt:.0f}"
        tw = draw.textlength(label, font=font_tick)
        draw.text((plot_x0 - tw - 10, py - 8), label, font=font_tick, fill=text_color)

    x_label = "brightness (0-255)"
    tw = draw.textlength(x_label, font=font_label)
    draw.text(((plot_x0 + plot_x1) / 2 - tw / 2, plot_y1 + 42), x_label, font=font_label, fill=text_color)

    y_label = "count (% of max)"
    _draw_vertical_text(img, draw, y_label, font_label, text_color,
                         cx=36, cy=(plot_y0 + plot_y1) // 2)

    for k, idx in enumerate(minima_idx):
        x = centers[idx]
        is_selected = (selected_idx is not None and idx == selected_idx)
        color = red if is_selected else green
        tval = threshold_values[k]
        label = f"{k + 1}: {tval * 255:.0f}"
        px = x_to_px(x)
        tw = draw.textlength(label, font=font_annot)
        y_offset = int(15 + (k % 3) * 28)
        draw.text((px - tw / 2, plot_y0 - y_offset - 16), label, font=font_annot, fill=color)

    return img


class AutoMaskThreshold:
    CATEGORY = "mask/threshold"
    FUNCTION = "run"
    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("mask_batch (x8)", "mask", "histgram_image")
    OUTPUT_NODE = True

    BINS = 256
    PEAKS_TARGET = 5
    MINOR_MINIMA_IN_LARGEST_PEAK = 3
    TOTAL_THRESHOLDS = PEAKS_TARGET + MINOR_MINIMA_IN_LARGEST_PEAK

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": cls.TOTAL_THRESHOLDS,
                    "step": 1,
                    "tooltip": f"Selects the n-th boundary value (ascending, 1-{cls.TOTAL_THRESHOLDS}) as the single-mask output.",
                }),
            },
        }

    def run(self, mask, threshold=5):
        bins = self.BINS

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        mask_np_full = mask.detach().cpu().numpy().astype(np.float32)

        hist_full, bin_edges_full = _mask_to_hist(mask_np_full, bins=bins)
        hist, bin_edges = _strip_leading_zero_bin(hist_full, bin_edges_full)

        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        regions, level = _find_peak_regions_by_level_cutoff(
            hist, n_peaks_target=self.PEAKS_TARGET
        )

        if len(regions) == 0:
            threshold_values = [0.5]
            boundary_idx = [len(centers) // 2]
        else:
            valley_idx = []
            for i, (start, end) in enumerate(regions):
                prev_end = regions[i - 1][1] if i > 0 else None
                v_idx = _find_valley_boundary_for_region(hist, (start, end), prev_end)
                valley_idx.append(v_idx)

            peak_values = [hist[start:end + 1].max() for (start, end) in regions]
            largest_peak_region_i = int(np.argmax(peak_values))
            largest_region = regions[largest_peak_region_i]

            minor_idx = _find_minor_minima_in_largest_peak(
                hist, largest_region, n_minima=self.MINOR_MINIMA_IN_LARGEST_PEAK
            )

            boundary_idx = sorted(set(valley_idx) | set(minor_idx))

            if len(boundary_idx) < self.TOTAL_THRESHOLDS:
                n = len(hist)
                extra_needed = self.TOTAL_THRESHOLDS - len(boundary_idx)
                candidates = [i for i in range(1, n - 1) if i not in boundary_idx]
                def min_dist(i):
                    return min(abs(i - b) for b in boundary_idx) if boundary_idx else i
                candidates.sort(key=min_dist, reverse=True)
                boundary_idx = sorted(set(boundary_idx) | set(candidates[:extra_needed]))

            boundary_idx = boundary_idx[:self.TOTAL_THRESHOLDS]
            threshold_values = [float(centers[idx]) for idx in boundary_idx]

        binary_masks = []
        for tval in threshold_values:
            bm = (mask >= tval).float()
            binary_masks.append(bm)

        mask_batch = torch.cat(binary_masks, dim=0)

        n_thresholds = len(threshold_values)
        selected_index = min(max(threshold - 1, 0), n_thresholds - 1)

        selected_mask = binary_masks[selected_index]
        selected_boundary_idx = boundary_idx[selected_index]

        hist_pil = _render_histogram_image(
            hist, bin_edges, boundary_idx, threshold_values, selected_idx=selected_boundary_idx
        )

        hist_np = np.array(hist_pil).astype(np.float32) / 255.0
        hist_tensor = torch.from_numpy(hist_np).unsqueeze(0)

        temp_dir = folder_paths.get_temp_directory()
        filename = f"auto_mask_threshold_{random.randint(0, 1000000)}.png"
        hist_pil.save(os.path.join(temp_dir, filename), compress_level=4)

        return {
            "ui": {
                "images": [{"filename": filename, "subfolder": "", "type": "temp"}],
                "num_thresholds": [len(threshold_values)],
            },
            "result": (mask_batch, selected_mask, hist_tensor),
        }


NODE_CLASS_MAPPINGS = {
    "MaskRefine": MaskRefine,
    "AutoMaskThreshold": AutoMaskThreshold,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskRefine": "Mask Refine",
    "AutoMaskThreshold": "Auto Mask Threshold",
}