# -*- coding: utf-8 -*-
"""
Fast fibre azimuthal angle relative to circumferential direction (0..90 deg),
using the closest of 12 contour lines per slice extracted from a 3D binary tif.

12 contours per slice:
- 3 contours inside the true inner contour:
    inner_inset_100
    inner_inset_200
    inner_inset_300
- original 9 contours:
    inner
    inner_to_center_1
    inner_to_center_2
    inner_to_center_3
    centerline
    center_to_outer_1
    center_to_outer_2
    center_to_outer_3
    outer

Change in this version
----------------------
Tangent calculation has been stabilised, while leaving the contour construction
and nearest-contour selection logic unchanged.

Instead of using a 1-point central difference only, tangents are now computed from:
1) circular smoothing of the contour coordinates, then
2) a wider-window central difference over multiple points.

This reduces jitter / flipping in tangents, especially on inner and inset contours.
"""

import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from tifffile import imread
import matplotlib.pyplot as plt
from itertools import permutations


# ----------------------- USER CONFIG -----------------------
orientation_rotated_csv = r"L4L5_fibres_inc4_orientation_rotated_polycurv_3scales_old.csv"
tif_path = r"SmoothSegmentation/L4L5_rotated_smoothsegmentation_new.tif"
output_csv = r"L4L5_fibres_inc4_angle_to_circumferential_0to90_closest12_old.csv"

# Mask origin in the coordinate units/frame used by the fibre CSV:
tif_zero_xyz = np.array([-541.322, 1641.11, -420.226], dtype=float)

rotationconfig = {
    "rotation_angle_deg": 0, # <-- EDIT THIS TO YOUR REAL VALUE (if already rotated, set to 0)
    "rotation_axis": np.array([0, 0, 1], dtype=float),
    "rotation_point": np.array([2587.581, 2851.675, 350.0], dtype=float),
}

# Expected annular-mask voxel counts (X,Y,Z), not maximum indices.
# IMPORTANT: X=width (columns), Y=height (rows), Z=number of slices.
# Put the known dimensions of your donut mask tif here:
DONUT_MASK_DIMS_XYZ = (5009, 3204, 1920)  # <-- EDIT THIS TO YOUR REAL VALUES (use voxel counts, not maximum indices)

N_CONTOUR_POINTS = 1000

# New inner contours, inward from the TRUE inner contour
INNER_INSET_GAPS_VOX = [100, 200, 300]

# Fixed contour numbering for CSV output
CONTOUR_NAME_TO_NUMBER = {
    "inner_inset_100": 1,
    "inner_inset_200": 2,
    "inner_inset_300": 3,
    "inner": 4,
    "inner_to_center_1": 5,
    "inner_to_center_2": 6,
    "inner_to_center_3": 7,
    "centerline": 8,
    "center_to_outer_1": 9,
    "center_to_outer_2": 10,
    "center_to_outer_3": 11,
    "outer": 12,
}

# Chunk size for streaming CSV processing (tune based on RAM)
CSV_CHUNKSIZE = 1_000_000

# If fibre tangent is nearly vertical, azimuth is undefined; set threshold on horiz magnitude
HORIZ_EPS = 1e-6

# Tangent stabilisation parameters
# Moving-average smoothing window on the closed contour (odd number recommended)
TANGENT_SMOOTH_WINDOW = 9
# Wider central-difference half-span: tangent ~ p[i+span] - p[i-span]
TANGENT_DIFF_SPAN = 4

# Sanity PNGs
SANITY_PNG_CONTOURS = r"L4L5_sanity_contours12_centre_slice.png"
SANITY_PNG_CONTOURS_POINTS = r"L4L5_sanity_contours12_plus_points_centre_slice.png"

# For points sanity plot
SANITY_POINT_SUBSAMPLE = 8000
SANITY_ARROW_COUNT = 60
SANITY_ARROW_SCALE = 30.0
SANITY_RANDOM_SEED = 42

# Optional debug print of contour usage counts
PRINT_DEBUG_COUNTS = True
# ----------------------------------------------------------


# ------------------ Rotation utilities (same as Code 1) ------------------ #
def rodrigues_rotation_matrix(direction, theta_deg):
    theta = np.deg2rad(theta_deg)
    direction = np.asarray(direction, dtype=float)
    u, v, w = direction / np.linalg.norm(direction)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c + u*u*(1-c),     u*v*(1-c) - w*s, u*w*(1-c) + v*s],
        [v*u*(1-c) + w*s,   c + v*v*(1-c),   v*w*(1-c) - u*s],
        [w*u*(1-c) - v*s,   w*v*(1-c) + u*s, c + w*w*(1-c)]
    ], dtype=float)

def apply_rotation(points_xyz, R, centre_xyz):
    points_xyz = np.asarray(points_xyz, dtype=float)
    centre_xyz = np.asarray(centre_xyz, dtype=float)
    return ((R @ (points_xyz - centre_xyz).T).T + centre_xyz)


# ------------------ Mask reading with expected dims ------------------ #
def reorder_mask_to_yxz(mask, expected_xyz):
    """
    Ensure mask is in array order (Y, X, Z) matching expected_xyz=(X,Y,Z).
    Tries all axis permutations and returns the one that matches exactly.
    """
    exp_x, exp_y, exp_z = map(int, expected_xyz)
    target = (exp_y, exp_x, exp_z)

    if mask.ndim != 3:
        raise ValueError(f"Expected 3D mask, got shape {mask.shape}")

    if mask.shape == target:
        return mask

    for perm in permutations([0, 1, 2], 3):
        cand = np.transpose(mask, perm)
        if cand.shape == target:
            return cand

    raise ValueError(
        f"Could not reorder mask to (Y,X,Z)={target} from {mask.shape}. "
        f"Check DONUT_MASK_DIMS_XYZ and/or tif contents."
    )

def read_AF_mask(tif_path, expected_xyz):
    donut_mask = imread(tif_path)
    donut_mask = (donut_mask > 0).astype(np.uint8)
    donut_mask = reorder_mask_to_yxz(donut_mask, expected_xyz)
    return donut_mask


# ------------------ Contour extraction ------------------ #
def extract_inner_outer_contours(slice_img):
    """Extract boundaries assuming a single annulus with one hole per slice.

    Multiple objects/holes can overwrite these selections; inspect diagnostics.
    """
    slice_img = (slice_img * 255).astype(np.uint8)
    contours, hierarchy = cv2.findContours(slice_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return None, None

    contours = [cnt[:, 0, :] for cnt in contours]
    hierarchy = hierarchy[0]

    outer_contour = None
    inner_contour = None

    for i, h in enumerate(hierarchy):
        _, _, child_idx, parent_idx = h
        if parent_idx == -1 and child_idx != -1:
            outer_contour = contours[i]
        elif parent_idx != -1:
            inner_contour = contours[i]

    if outer_contour is None or inner_contour is None:
        return None, None
    return inner_contour, outer_contour

def resample_contour(contour, n_points=1000):
    contour = np.vstack([contour, contour[0]])
    distances = np.cumsum(np.r_[0, np.linalg.norm(np.diff(contour, axis=0), axis=1)])
    total_length = distances[-1]
    if total_length <= 0:
        return contour[:1].astype(float)
    interp_points = np.linspace(0, total_length, n_points)
    x = np.interp(interp_points, distances, contour[:, 0])
    y = np.interp(interp_points, distances, contour[:, 1])
    return np.vstack((x, y)).T

def align_contours(inner_contour, outer_contour, n_points=1000):
    inner_pts = resample_contour(inner_contour, n_points)
    outer_pts = resample_contour(outer_contour, n_points)

    dist_normal = np.linalg.norm(inner_pts[0] - outer_pts[0])
    dist_flipped = np.linalg.norm(inner_pts[-1] - outer_pts[0])
    if dist_flipped < dist_normal:
        inner_pts = inner_pts[::-1]
    return inner_pts, outer_pts

def smooth_closed_contour(points_xy, window=9):
    """
    Circular moving-average smoothing for a closed contour.
    """
    pts = np.asarray(points_xy, dtype=float)
    n = len(pts)
    if n == 0 or window <= 1:
        return pts.copy()

    # Ensure sensible odd window, not larger than contour length
    window = int(window)
    window = max(1, window)
    if window % 2 == 0:
        window += 1
    if window > n:
        window = n if n % 2 == 1 else n - 1
        window = max(window, 1)

    half = window // 2
    kernel = np.ones(window, dtype=float) / window

    padded_x = np.pad(pts[:, 0], (half, half), mode="wrap")
    padded_y = np.pad(pts[:, 1], (half, half), mode="wrap")

    x_s = np.convolve(padded_x, kernel, mode="valid")
    y_s = np.convolve(padded_y, kernel, mode="valid")

    return np.column_stack([x_s, y_s])

def compute_tangent_closed_xy(points_xy, smooth_window=TANGENT_SMOOTH_WINDOW, diff_span=TANGENT_DIFF_SPAN):
    """
    Stabilised tangents for a closed contour:
    1) circular smoothing of contour coordinates
    2) wider-span central difference

    This leaves the contour-selection logic unchanged, but makes tangent
    directions less noisy / less likely to flip locally.
    """
    pts = np.asarray(points_xy, dtype=float)
    n = len(pts)
    if n == 0:
        return np.zeros((0, 2), dtype=float)
    if n == 1:
        return np.zeros((1, 2), dtype=float)

    pts_s = smooth_closed_contour(pts, window=smooth_window)

    span = int(max(1, diff_span))
    if 2 * span >= n:
        span = max(1, (n - 1) // 2)

    prev = np.roll(pts_s, span, axis=0)
    nxt = np.roll(pts_s, -span, axis=0)
    tan = nxt - prev

    norms = np.linalg.norm(tan, axis=1, keepdims=True)
    bad = norms[:, 0] < 1e-12
    norms[bad] = 1.0
    tan = tan / norms

    # Fallback for any degenerate points: use simple 1-step difference on smoothed contour
    if np.any(bad):
        prev1 = np.roll(pts_s, 1, axis=0)
        nxt1 = np.roll(pts_s, -1, axis=0)
        tan1 = nxt1 - prev1
        n1 = np.linalg.norm(tan1, axis=1, keepdims=True)
        n1[n1[:, 0] < 1e-12] = 1.0
        tan1 = tan1 / n1
        tan[bad] = tan1[bad]

    return tan

def compute_paired_inner_for_outer(inner_pts, outer_pts):
    tree = cKDTree(inner_pts)
    _, idx = tree.query(outer_pts)
    return inner_pts[idx]

def make_9_contour_lines(inner_pts, outer_pts):
    matched_inner = compute_paired_inner_for_outer(inner_pts, outer_pts)

    fractions = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    names = [
        "inner",
        "inner_to_center_1",
        "inner_to_center_2",
        "inner_to_center_3",
        "centerline",
        "center_to_outer_1",
        "center_to_outer_2",
        "center_to_outer_3",
        "outer"
    ]

    return {name: matched_inner + f * (outer_pts - matched_inner) for f, name in zip(fractions, names)}


# ------------------ Additional inner contours ------------------ #
def build_hole_mask_from_inner_contour(shape_yx, inner_contour_xy):
    """
    Binary mask of the region enclosed by the true inner contour.
    """
    mask = np.zeros(shape_yx, dtype=np.uint8)
    poly = np.round(inner_contour_xy).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 1)
    return mask

def extract_largest_external_contour(binary_img):
    """
    Extract the largest external contour from a binary image.
    Returns contour as (N,2) or None.
    """
    binary_img = (binary_img > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    return cnt[:, 0, :]

def make_inner_inset_contours(inner_pts_true, shape_yx, gaps_vox, n_points=1000):
    """
    Build contours inside the true inner contour using a distance transform
    of the enclosed hole region.

    For each gap g:
        inset_region = dist_transform(hole_mask) >= g
        inset contour = outer boundary of inset_region

    Returns:
        dict: name -> resampled contour_xy
    """
    hole_mask = build_hole_mask_from_inner_contour(shape_yx, inner_pts_true)

    # Distance to boundary, measured inside the hole
    dist = cv2.distanceTransform(hole_mask.astype(np.uint8), cv2.DIST_L2, 5)

    inset_lines = {}
    for g in gaps_vox:
        inset_region = (dist >= float(g)).astype(np.uint8)

        # Too small to contour
        if inset_region.sum() < 10:
            continue

        cnt = extract_largest_external_contour(inset_region)
        if cnt is None or len(cnt) < 10:
            continue

        inset_lines[f"inner_inset_{int(g)}"] = resample_contour(cnt, n_points=n_points)

    return inset_lines


def build_rotated_contours_12lines_FAST(
    donut_mask_yxz,
    tif_zero_xyz,
    R,
    centre_xyz,
    n_points=1000,
    inner_inset_gaps_vox=(100, 200, 300)
):
    """
    Builds per-slice data with:
      - original 9 contours
      - 3 extra contours inside the true inner contour (if they exist on that slice)
      - ONE combined KDTree per slice over all contour points (XY)

    Returns:
      z_map[z_key] = {
        "lines": dict(name -> rotated_xyz Nx3),
        "tree_all_xy": KDTree over stacked XY,
        "tan_all_xy": tangent for each stacked point,
        "name_all": line name for each stacked point,
        "xyz_all": stacked rotated xyz
      }
    """
    z_map = {}
    Z = donut_mask_yxz.shape[2]

    for z in range(Z):
        slice_img = donut_mask_yxz[:, :, z]
        inner_c, outer_c = extract_inner_outer_contours(slice_img)
        if inner_c is None:
            continue

        inner_pts_true, outer_pts_true = align_contours(inner_c, outer_c, n_points=n_points)

        # Original 9 contours
        lines_xy_9 = make_9_contour_lines(inner_pts_true, outer_pts_true)

        # Additional inner contours
        inset_lines_xy = make_inner_inset_contours(
            inner_pts_true=inner_pts_true,
            shape_yx=slice_img.shape,
            gaps_vox=inner_inset_gaps_vox,
            n_points=n_points
        )

        # Combine: add inset lines first, then original 9
        lines_xy = {}
        lines_xy.update(inset_lines_xy)
        lines_xy.update(lines_xy_9)

        lines_rot_xyz = {}
        stacked_xy = []
        stacked_tan_xy = []
        stacked_names = []
        stacked_xyz = []

        for name, xy in lines_xy.items():
            tan_xy = compute_tangent_closed_xy(xy)

            xyz = np.column_stack([
                xy[:, 0] + tif_zero_xyz[0],
                xy[:, 1] + tif_zero_xyz[1],
                np.full(xy.shape[0], z + tif_zero_xyz[2], dtype=float),
            ])

            xyz_rot = apply_rotation(xyz, R, centre_xyz)

            # rotate tangents (z=0) and normalize XY
            tan_xyz = np.column_stack([tan_xy[:, 0], tan_xy[:, 1], np.zeros(len(tan_xy))])
            tan_xyz_rot = (R @ tan_xyz.T).T
            tan_xy_rot = tan_xyz_rot[:, :2]
            tn = np.linalg.norm(tan_xy_rot, axis=1, keepdims=True)
            tn[tn == 0] = 1.0
            tan_xy_rot /= tn

            lines_rot_xyz[name] = xyz_rot

            stacked_xy.append(xyz_rot[:, :2])
            stacked_tan_xy.append(tan_xy_rot)
            stacked_names.append(np.full(len(xyz_rot), name, dtype=object))
            stacked_xyz.append(xyz_rot)

        if len(stacked_xy) == 0:
            continue

        all_xy = np.vstack(stacked_xy)
        all_tan_xy = np.vstack(stacked_tan_xy)
        all_names = np.concatenate(stacked_names)
        all_xyz = np.vstack(stacked_xyz)

        tree_all = cKDTree(all_xy)

        z_key = int(round(z + tif_zero_xyz[2]))
        z_map[z_key] = {
            "lines": lines_rot_xyz,
            "tree_all_xy": tree_all,
            "tan_all_xy": all_tan_xy,
            "name_all": all_names,
            "xyz_all": all_xyz
        }

    return z_map


# ------------------ Nearest-z lookup (no skipping) ------------------ #
def nearest_available_z(z_values, available_z_sorted):
    """
    Vectorized: map each z in z_values to nearest entry in available_z_sorted.
    """
    z_values = np.asarray(z_values, dtype=int)
    a = available_z_sorted

    idx = np.searchsorted(a, z_values, side="left")
    idx0 = np.clip(idx - 1, 0, len(a) - 1)
    idx1 = np.clip(idx,     0, len(a) - 1)

    z0 = a[idx0]
    z1 = a[idx1]

    choose1 = np.abs(z_values - z1) < np.abs(z_values - z0)
    return np.where(choose1, z1, z0)


# ------------------ Fast fibre XY direction + angle ------------------ #
def fibre_xy_from_phi_theta(phi_deg, theta_deg, horiz_eps=1e-6):
    """
    For azimuthal direction in XY, the UNIT direction is just (cos(phi), sin(phi))
    provided the tangent has nonzero horizontal component.
    We detect near-vertical tangents from theta (via alpha) and mark invalid.

    Returns:
      fib_xy: (N,2)
      valid: (N,) bool
    """
    phi = np.deg2rad(phi_deg.astype(float))

    # detect near-vertical: horiz = sin(alpha)
    theta = theta_deg.astype(float)
    alpha = np.where(theta >= 0.0, theta, theta + 180.0)
    horiz = np.sin(np.deg2rad(alpha))

    valid = np.abs(horiz) > horiz_eps

    fib_xy = np.column_stack([np.cos(phi), np.sin(phi)])
    return fib_xy, valid

# This angle compares XY projections, not the full 3D fibre tangent.
def angle0to90_from_unit_xy(fib_xy, circ_xy):
    """
    fib_xy: (N,2) unit
    circ_xy: (N,2) unit
    angle 0..90 = arccos(abs(dot))
    """
    dot = fib_xy[:, 0] * circ_xy[:, 0] + fib_xy[:, 1] * circ_xy[:, 1]
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


# ------------------ Sanity plots ------------------ #
def choose_center_z_key(z_map, donut_mask_yxz, tif_zero_xyz):
    z_center = int(round(tif_zero_xyz[2] + (donut_mask_yxz.shape[2] - 1) / 2.0))
    if z_center in z_map:
        return z_center
    z_available = np.array(sorted(z_map.keys()))
    return int(z_available[np.argmin(np.abs(z_available - z_center))])

def save_sanity_contours_png(z_map, z_key, png_path):
    data = z_map[z_key]
    lines = data["lines"]

    plt.figure(figsize=(8, 8))
    for name, xyz in lines.items():
        plt.plot(xyz[:, 0], xyz[:, 1], linewidth=1, label=name)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Rotated contour lines (12) at z = {z_key}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved sanity PNG (contours): {png_path}")

def save_sanity_contours_plus_points_png(
    z_map,
    z_key,
    points_df,                 # must contain x,y,z,closest_line,closest_contour_x,closest_contour_y,circ_tx,circ_ty
    png_path,
    subsample=8000,
    arrow_count=60,
    arrow_scale=30.0,
    seed=42
):
    data = z_map[z_key]
    lines = data["lines"]

    slice_df = points_df[np.isclose(points_df["z"].values, float(z_key))].copy()
    if len(slice_df) == 0:
        print(f"No fibre points at z={z_key} for sanity plot.")
        return

    rng = np.random.default_rng(seed)

    if len(slice_df) > subsample:
        take = rng.choice(len(slice_df), size=subsample, replace=False)
        slice_df = slice_df.iloc[take].copy()

    # categorical color mapping
    contour_nums = sorted(points_df["closest_line"].dropna().unique().tolist())
    num_to_code = {n: i for i, n in enumerate(contour_nums)}
    codes = slice_df["closest_line"].map(num_to_code).values

    plt.figure(figsize=(9, 9))

    # plot contours
    for name, xyz in lines.items():
        plt.plot(xyz[:, 0], xyz[:, 1], linewidth=1, label=name)

    # plot points
    sc = plt.scatter(slice_df["x"].values, slice_df["y"].values, c=codes, s=6, alpha=0.75)

    # draw tangent arrows at closest contour points (a few)
    if len(slice_df) > 0:
        m = min(arrow_count, len(slice_df))
        arrow_idx = rng.choice(len(slice_df), size=m, replace=False)
        sub = slice_df.iloc[arrow_idx]

        # anchor at closest contour point
        x0 = sub["closest_contour_x"].values.astype(float)
        y0 = sub["closest_contour_y"].values.astype(float)

        # tangent direction
        u = sub["circ_tx"].values.astype(float)
        v = sub["circ_ty"].values.astype(float)

        # quiver
        plt.quiver(
            x0, y0, u, v,
            angles="xy", scale_units="xy", scale=1.0/arrow_scale,
            width=0.003, headwidth=3, headlength=4, headaxislength=3,
            alpha=0.9
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Rotated contours + fibre points (n={len(slice_df)}) + tangent arrows at z = {z_key}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=7, ncol=2)

    cbar = plt.colorbar(sc, fraction=0.046, pad=0.04)
    cbar.set_label("closest_line (categorical code)")
    cbar.set_ticks(list(num_to_code.values()))
    cbar.set_ticklabels([str(n) for n in contour_nums])

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"Saved sanity PNG (contours + points + arrows): {png_path}")


# ------------------ Main (FAST + no skipping) ------------------ #
def main():
    # Rotation matrix
    R = rodrigues_rotation_matrix(rotationconfig["rotation_axis"], rotationconfig["rotation_angle_deg"])
    centre = rotationconfig["rotation_point"]

    # Load mask with expected dims handling
    donut_mask_yxz = read_AF_mask(tif_path, expected_xyz=DONUT_MASK_DIMS_XYZ)
    print(f"Loaded donut mask (Y,X,Z) = {donut_mask_yxz.shape}")

    # Build contours + combined KDTree per slice
    print("Building rotated contours and per-slice KDTree (combined 12 lines)...")
    z_map = build_rotated_contours_12lines_FAST(
        donut_mask_yxz,
        tif_zero_xyz=tif_zero_xyz,
        R=R,
        centre_xyz=centre,
        n_points=N_CONTOUR_POINTS,
        inner_inset_gaps_vox=INNER_INSET_GAPS_VOX
    )
    if not z_map:
        raise RuntimeError("No valid contours extracted from tif. Check mask/dims/threshold.")

    available_z = np.array(sorted(z_map.keys()), dtype=int)
    print(f"Slices with contours: {len(available_z)} (z from {available_z[0]} to {available_z[-1]})")

    # Sanity PNG (contours only) at centre slice
    z_centre_key = choose_center_z_key(z_map, donut_mask_yxz, tif_zero_xyz)
    save_sanity_contours_png(z_map, z_centre_key, SANITY_PNG_CONTOURS)

    # Stream fibre CSV in chunks and write output incrementally
    usecols = ["x", "y", "z", "fibre_id", "theta", "phi"]
    reader = pd.read_csv(orientation_rotated_csv, usecols=usecols, chunksize=CSV_CHUNKSIZE)

    first_write = True
    sanity_buffer = []

    print("Processing fibre CSV in chunks...")
    for chunk_idx, df in enumerate(reader):
        # basic arrays
        x = df["x"].values.astype(float)
        y = df["y"].values.astype(float)
        z = df["z"].values.astype(float)
        z_int = np.rint(z).astype(int)

        # No maximum z gap: out-of-range points also use the nearest contour slice.
        # Slice labels are valid for z-preserving rotations only.
        z_near = nearest_available_z(z_int, available_z)

        # fibre azimuth direction in XY (unit) from phi (fast)
        fib_xy, valid = fibre_xy_from_phi_theta(df["phi"].values, df["theta"].values, horiz_eps=HORIZ_EPS)

        # Prepare outputs
        angle90 = np.full(len(df), np.nan, dtype=float)
        closest_line = np.empty(len(df), dtype=np.int32)
        closest_dist = np.empty(len(df), dtype=float)
        circ_tx = np.empty(len(df), dtype=float)
        circ_ty = np.empty(len(df), dtype=float)
        closest_contour_x = np.empty(len(df), dtype=float)
        closest_contour_y = np.empty(len(df), dtype=float)
        z_used = z_near.astype(float)

        # Process per unique mapped z
        for z_key in np.unique(z_near):
            idx = np.where(z_near == z_key)[0]
            if idx.size == 0:
                continue

            tree = z_map[int(z_key)]["tree_all_xy"]
            # No XY distance threshold; nearest sampled point, not exact curve projection.
            dist, nn = tree.query(np.column_stack([x[idx], y[idx]]))

            nn = nn.astype(int)
            closest_dist[idx] = dist

            tan_xy = z_map[int(z_key)]["tan_all_xy"][nn]
            names = z_map[int(z_key)]["name_all"][nn]
            xyz_anchor = z_map[int(z_key)]["xyz_all"][nn]

            contour_nums = np.array([CONTOUR_NAME_TO_NUMBER[str(n)] for n in names], dtype=np.int32)

            circ_tx[idx] = tan_xy[:, 0]
            circ_ty[idx] = tan_xy[:, 1]
            closest_line[idx] = contour_nums
            closest_contour_x[idx] = xyz_anchor[:, 0]
            closest_contour_y[idx] = xyz_anchor[:, 1]

            vmask = valid[idx]
            if np.any(vmask):
                angle90[idx[vmask]] = angle0to90_from_unit_xy(fib_xy[idx[vmask]], tan_xy[vmask])

        out_df = pd.DataFrame({
            "x": x,
            "y": y,
            "z": z,                  # original z
            "z_used": z_used,        # nearest contour slice z used
            "fibre_id": df["fibre_id"].values.astype(int),
            "theta_rot": df["theta"].values.astype(float),
            "phi_rot": df["phi"].values.astype(float),

            "closest_line": closest_line,
            "closest_line_dist_xy": closest_dist,

            "circ_tx": circ_tx,
            "circ_ty": circ_ty,

            "closest_contour_x": closest_contour_x,
            "closest_contour_y": closest_contour_y,

            "angle_to_circumferential_0to90_deg": angle90
        })

        if PRINT_DEBUG_COUNTS:
            unique_nums, counts = np.unique(closest_line, return_counts=True)
            count_str = ", ".join([f"{int(n)}={c:,}" for n, c in zip(unique_nums, counts)])
            print(f"    closest_line counts: {count_str}")

        # write incrementally
        out_df.to_csv(output_csv, mode="w" if first_write else "a", index=False, header=first_write)
        first_write = False

        # Collect points for centre-slice sanity plot
        centre_mask = np.isclose(out_df["z_used"].values, float(z_centre_key))
        if np.any(centre_mask):
            tmp = out_df.loc[centre_mask].copy()
            if len(tmp) > 2000:
                tmp = tmp.sample(n=2000, random_state=SANITY_RANDOM_SEED + chunk_idx)
            sanity_buffer.append(tmp)

        print(f"  Chunk {chunk_idx+1}: wrote {len(out_df):,} rows")

    print(f"Done. Output written to: {output_csv}")

    # Build sanity points dataframe
    if len(sanity_buffer) > 0:
        sanity_df = pd.concat(sanity_buffer, ignore_index=True)
        if len(sanity_df) > SANITY_POINT_SUBSAMPLE:
            sanity_df = sanity_df.sample(n=SANITY_POINT_SUBSAMPLE, random_state=SANITY_RANDOM_SEED)

        # Plot against z_used
        sanity_df_plot = sanity_df.copy()
        sanity_df_plot["z"] = sanity_df_plot["z_used"]

        save_sanity_contours_plus_points_png(
            z_map=z_map,
            z_key=z_centre_key,
            points_df=sanity_df_plot,
            png_path=SANITY_PNG_CONTOURS_POINTS,
            subsample=SANITY_POINT_SUBSAMPLE,
            arrow_count=SANITY_ARROW_COUNT,
            arrow_scale=SANITY_ARROW_SCALE,
            seed=SANITY_RANDOM_SEED
        )
    else:
        print("No points found near centre slice for sanity plot with points/arrows.")

    print("All done.")


if __name__ == "__main__":
    main()