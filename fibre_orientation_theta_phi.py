# -*- coding: utf-8 -*-
r"""
Fibre orientation extraction from Excel-XML Avizo fibre tracing.

Preserves the original point-cloud generation, spline fitting, tangent
normalisation, angle conventions, parallel processing, and rotated outputs.

Orientation CSV columns: x, y, z, fibre_id, theta, phi.
Theta is in [-90, 90] degrees; phi is in [0, 360) degrees.

Required packages: numpy, pandas, scipy, joblib, lxml.
Temporary files use D:\Temp; edit the paths below if needed.
"""

import os

# ------------------------------------------------------------------
# Environment: temp on D:, use all threads for BLAS / LAPACK
# ------------------------------------------------------------------
os.environ["TEMP"] = r"D:\Temp"
os.environ["TMP"] = r"D:\Temp"
os.environ["JOBLIB_TEMP_FOLDER"] = r"D:\Temp"

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())
os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())

import numpy as np
import pandas as pd
from numpy.linalg import norm
import time
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional
from joblib import Parallel, delayed, dump, load
import tempfile
import shutil
from lxml import etree
from scipy.interpolate import splprep, splev

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
os.makedirs(r"D:\Temp", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
class Config:
    """Configuration parameters for fibre orientation extraction"""
    def __init__(self):
        # -------------------------
        # I/O
        # -------------------------
        self.input_filename = "L2L3_fibres.xml"

        self.pointcloud_output = "L2L3_fibres_inc4_pointcloud.txt"
        self.pointcloud_output_rotated = "L2L3_fibres_inc4_pointcloud_rotated.txt"

        self.orientation_output = "L2L3_fibres_inc4_orientation.csv"
        self.orientation_output_rotated = "L2L3_fibres_inc4_orientation_rotated.csv"

        self.spacecurve_output = "L2L3_fibres_inc4_spacecurve.npz"

        # -------------------------
        # Processing parameters
        # -------------------------
        self.spacing_increment = 4.0          # Segment stepping distance in input units; see resampling caveat below.
        self.min_points_for_fit = 4

        # -------------------------
        # Parallelism
        # -------------------------
        self.max_workers = os.cpu_count()
        self.fit_chunk_size = 200
        self.tangent_chunk_size = 200

rotationconfig = {
    "rotation_angle_deg": 39,
    "rotation_axis": np.array([0, 0, 1], dtype=float),
    "rotation_point": np.array([2587.581, 2851.675, 350.0], dtype=float),
}

# ------------------ Rotation utilities ------------------ #
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

def apply_rotation(points, R, centre):
    points = np.asarray(points, dtype=float)
    centre = np.asarray(centre, dtype=float)
    return ((R @ (points - centre).T).T + centre)

def rotate_vectors(vectors: np.ndarray, R: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    return (R @ vectors.T).T

# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------
def validate_input_file(filepath: str) -> Path:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    return path

# ------------------------------------------------------------------
# Streaming XML loader
# ------------------------------------------------------------------
def load_data_from_xml(config: Config):
    """
    Memory-safe streaming XML loader for very large Excel-XML files.
    Reads only Points + Segments sheets.
    """
    ns = "{urn:schemas-microsoft-com:office:spreadsheet}"

    points = []
    point_ids = []

    current_sheet = None
    header = []
    col_index = {}

    def parse_row(row_cells):
        return [cell.text if cell is not None else None for cell in row_cells]

    context = etree.iterparse(
        config.input_filename,
        events=("start", "end"),
        huge_tree=True
    )

    for event, elem in context:
        tag = elem.tag

        if event == "start" and tag.endswith("Worksheet"):
            current_sheet = elem.attrib.get(f"{ns}Name")

        if event == "end" and tag.endswith("Row"):
            cells = elem.findall(f"{ns}Cell/{ns}Data")
            row = parse_row(cells)

            if current_sheet == "Points":
                if not header:
                    header = row
                    col_index = {name: i for i, name in enumerate(header)}
                else:
                    x = float(row[col_index["X Coord"]])
                    y = float(row[col_index["Y Coord"]])
                    z = float(row[col_index["Z Coord"]])
                    points.append((x, y, z))

            elif current_sheet == "Segments":
                if not header:
                    header = row
                    col_index = {name: i for i, name in enumerate(header)}
                else:
                    raw_id = row[col_index["Point IDs"]]
                    pid = np.array([int(x.strip()) for x in raw_id.split(",")], dtype=int)
                    point_ids.append(pid)

            elem.clear()

        if event == "end" and tag.endswith("Worksheet"):
            current_sheet = None
            header = []
            col_index = {}

    coords = np.array(points, dtype=float)
    point_ids = list(point_ids)
    lamella = np.ones(len(point_ids), dtype=int)

    logger.info(f"Loaded {len(coords)} raw points (streaming XML)")
    logger.info(f"Loaded {len(point_ids)} fibres (streaming XML)")
    return coords, point_ids, lamella

# ------------------------------------------------------------------
# Geometry / point cloud
# ------------------------------------------------------------------
def interpolate_fibre(fibre_coords: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    """Step within trace segments; residual distance is not carried across vertices.

    This preserves the original algorithm, not uniform whole-fibre arc-length
    sampling. Use a positive spacing in the same units as the input coordinates.
    """
    if len(fibre_coords) < 2:
        return fibre_coords, np.zeros((len(fibre_coords), 3), dtype=np.float32)

    positions = [fibre_coords[0]]
    directions = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]

    current_pos = fibre_coords[0]
    target_idx = 1

    while target_idx < len(fibre_coords):
        next_point = fibre_coords[target_idx]
        vec = next_point - current_pos
        dist = norm(vec)

        if dist < 1e-10:
            target_idx += 1
            continue

        if dist >= spacing:
            direction = (vec / dist).astype(np.float32)
            new_pos = current_pos + spacing * direction
            positions.append(new_pos)
            directions.append(direction)
            current_pos = new_pos
        else:
            target_idx += 1
            if target_idx < len(fibre_coords):
                current_pos = next_point

    positions = np.array(positions, dtype=np.float32)
    directions = np.array(directions, dtype=np.float32)

    if len(directions) < len(positions):
        padding = np.zeros((len(positions) - len(directions), 3), dtype=np.float32)
        if len(directions) > 0:
            padding[:] = directions[-1]
        directions = np.vstack([directions, padding])

    return positions, directions

def build_point_cloud(coord: np.ndarray, point_ids, lamella: np.ndarray, config: Config) -> np.ndarray:
    logger.info("Building point cloud with interpolated points")

    estimated_total_points = len(point_ids) * 100
    point_cloud = np.empty((estimated_total_points, 10), dtype=np.float32)

    global_point_count = 0

    for fibre_idx, pid in enumerate(point_ids):
        # Assumes Point IDs are zero-based row indices, not arbitrary Avizo IDs.
        point_indices = np.array(pid, dtype=int)
        fibre_coords = coord[point_indices, :]

        positions, directions = interpolate_fibre(
            fibre_coords.astype(np.float32),
            config.spacing_increment
        )
        num_points = len(positions)
        if num_points == 0:
            continue

        if global_point_count + num_points > len(point_cloud):
            new_size = max(len(point_cloud) * 2, global_point_count + num_points)
            point_cloud = np.resize(point_cloud, (new_size, 10))

        pc_slice = slice(global_point_count, global_point_count + num_points)
        point_cloud[pc_slice, 0] = lamella[fibre_idx]
        point_cloud[pc_slice, 1] = fibre_idx + 1
        point_cloud[pc_slice, 2] = np.arange(1, num_points + 1, dtype=np.float32)
        point_cloud[pc_slice, 3] = np.arange(global_point_count + 1,
                                             global_point_count + num_points + 1,
                                             dtype=np.float32)
        point_cloud[pc_slice, 4:7] = positions
        point_cloud[pc_slice, 7:10] = directions

        global_point_count += num_points

        if fibre_idx % 5000 == 0 and fibre_idx > 0:
            logger.info(f"Processed {fibre_idx}/{len(point_ids)} fibres, "
                        f"{global_point_count} points generated")

    point_cloud = point_cloud[:global_point_count]
    logger.info(f"Generated {len(point_cloud)} interpolated points")
    return point_cloud

def build_fibre_index(point_cloud: np.ndarray) -> Dict[int, np.ndarray]:
    logger.info("Building fibre index")
    index: Dict[int, list] = {}
    fibre_ids = point_cloud[:, 1].astype(int)

    for i, fid in enumerate(fibre_ids):
        index.setdefault(fid, []).append(i)

    for fid in index:
        index[fid] = np.array(index[fid], dtype=np.int64)

    logger.info("Fibre index built")
    return index

# ------------------------------------------------------------------
# Spline fitting for orientation
# ------------------------------------------------------------------
def fit_space_curve(fibre_data: np.ndarray, config: Config) -> Optional[Dict]:
    npts = len(fibre_data)
    if npts < config.min_points_for_fit:
        return None

    positions = fibre_data[:, 4:7].astype(np.float64)

    segments = positions[1:] - positions[:-1]
    distances = norm(segments, axis=1)
    arc_length = np.insert(np.cumsum(distances), 0, 0.0)
    total_length = arc_length[-1]
    if total_length <= 0:
        return None

    u = arc_length / total_length
    k = 3
    if npts <= k:
        k = max(1, npts - 1)

    try:
        s_local = 0.02 * len(positions)
        tck_local, u_out = splprep(positions.T, u=u, s=s_local, k=k)
        return {"tck_local": tck_local, "u": u_out, "order": k}
    except Exception:
        # Fit failures become missing angles; the original exception is not logged.
        return None

def chunked(iterable, chunk_size):
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i:i + chunk_size]

def fit_all_fibres_joblib(point_cloud: np.ndarray,
                          fibre_index: Dict[int, np.ndarray],
                          config: Config) -> Dict[int, Dict]:
    logger.info("Fitting space curves (splines) using joblib + chunking (for tangents only)")

    fibre_ids = np.array(sorted(fibre_index.keys()), dtype=int)

    def process_chunk(fid_chunk):
        out = []
        for fid in fid_chunk:
            idx = fibre_index[fid]
            fibre_data = point_cloud[idx]
            fit = fit_space_curve(fibre_data, config)
            if fit is not None:
                out.append((fid, fit))
        return out

    chunks = list(chunked(fibre_ids, config.fit_chunk_size))

    results = Parallel(
        n_jobs=config.max_workers,
        backend="loky",
        verbose=10,
        batch_size=1
    )(
        delayed(process_chunk)(chunk)
        for chunk in chunks
    )

    curve_fits: Dict[int, Dict] = {}
    for chunk_result in results:
        for fid, fit in chunk_result:
            curve_fits[fid] = fit

    logger.info(f"Successfully fit {len(curve_fits)}/{len(fibre_ids)} fibres (splines)")
    return curve_fits

# ------------------------------------------------------------------
# Tangents from the fitted spline
# ------------------------------------------------------------------
def compute_tangent_vector_spline(fit: Dict) -> np.ndarray:
    tck = fit["tck_local"]
    u = fit["u"]

    dx, dy, dz = splev(u, tck, der=1)
    tangents = np.column_stack([dx, dy, dz]).astype(np.float64)

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms
    return tangents.astype(np.float32)

def compute_all_tangents_joblib(point_cloud: np.ndarray,
                                fibre_index: Dict[int, np.ndarray],
                                curve_fits: Dict[int, Dict],
                                config: Config) -> np.ndarray:
    logger.info("Computing tangents using joblib + chunking")

    tangents = np.zeros((len(point_cloud), 3), dtype=np.float32)
    fibre_ids = np.array(sorted(fibre_index.keys()), dtype=int)

    def process_chunk(fid_chunk):
        out = []
        for fid in fid_chunk:
            idx = fibre_index[fid]
            if fid not in curve_fits:
                tangent = np.full((len(idx), 3), np.nan, dtype=np.float32)
            else:
                tangent = compute_tangent_vector_spline(curve_fits[fid])
            out.append((idx, tangent))
        return out

    chunks = list(chunked(fibre_ids, config.tangent_chunk_size))

    results = Parallel(
        n_jobs=config.max_workers,
        backend="loky",
        verbose=10,
        batch_size=1
    )(
        delayed(process_chunk)(chunk)
        for chunk in chunks
    )

    for chunk_result in results:
        for idx, tan in chunk_result:
            tangents[idx] = tan

    return tangents

# ------------------------------------------------------------------
# Orientation
# ------------------------------------------------------------------
def compute_orientation_angles(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logger.info("Computing orientation angles (theta [-90,90], phi [0,360))")

    t = tangents.astype(np.float64, copy=False)
    tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]

    horiz = np.sqrt(tx * tx + ty * ty)
    # Polar angle from +z, folded across 90 degrees; not elevation above XY.
    # Phi depends on trace direction and is undefined for vertical tangents.
    alpha = np.degrees(np.arctan2(horiz, tz))
    theta = np.where(alpha <= 90.0, alpha, alpha - 180.0)
    phi = (np.degrees(np.arctan2(ty, tx)) + 360.0) % 360.0

    return theta.astype(np.float32), phi.astype(np.float32)

# ------------------------------------------------------------------
# Saving results
# ------------------------------------------------------------------
def save_pointcloud_txt(path: str, point_cloud: np.ndarray):
    arr = np.column_stack([
        point_cloud[:, 3].astype(np.int64),
        point_cloud[:, 4].astype(np.float64),
        point_cloud[:, 5].astype(np.float64),
        point_cloud[:, 6].astype(np.float64),
    ])
    np.savetxt(path, arr, fmt="%d\t%.6f\t%.6f\t%.6f", delimiter="\t")

def save_spacecurve_npz(path: str, point_cloud: np.ndarray):
    np.savez_compressed(
        path,
        point_cloud=point_cloud,
        lamella=point_cloud[:, 0],
        fibre_id=point_cloud[:, 1],
        local_id=point_cloud[:, 2],
        global_id=point_cloud[:, 3],
        coordinates=point_cloud[:, 4:7],
        directions=point_cloud[:, 7:10]
    )

def save_orientation_csv(orientation_csv_path: str,
                         point_cloud: np.ndarray,
                         theta: np.ndarray,
                         phi: np.ndarray):
    data = {
        "x": point_cloud[:, 4],
        "y": point_cloud[:, 5],
        "z": point_cloud[:, 6],
        "fibre_id": point_cloud[:, 1].astype(int),
        "theta": theta,
        "phi": phi,
    }
    pd.DataFrame(data).to_csv(orientation_csv_path, index=False)

# ------------------------------------------------------------------
# Disk helpers for memmap on D:
# ------------------------------------------------------------------
def check_disk_requirements(array: np.ndarray, path: Path, safety_factor: float = 2.0):
    required_bytes = int(array.nbytes * safety_factor)
    drive = path.anchor
    free = shutil.disk_usage(drive).free

    logger.info(f"Point cloud size: {array.nbytes / 1e9:.2f} GB")
    logger.info(f"Estimated required space: {required_bytes / 1e9:.2f} GB")
    logger.info(f"Free space on {drive}: {free / 1e9:.2f} GB")

    if free < required_bytes:
        raise RuntimeError(
            f"Not enough disk space on {drive}. "
            f"Need {required_bytes / 1e9:.2f} GB, free {free / 1e9:.2f} GB."
        )

def cleanup_temp_folder(path: Path):
    try:
        if path.exists() and path.is_dir():
            logger.info(f"Cleaning up temporary folder: {path}")
            shutil.rmtree(path)
    except Exception as e:
        logger.warning(f"Could not delete temp folder {path}: {e}")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    start_time = time.time()
    config = Config()

    TMP_ROOT = Path(r"D:\Temp\Joblib")
    TMP_ROOT.mkdir(parents=True, exist_ok=True)

    tmpdir = None

    try:
        logger.info(f"Using temp root: {TMP_ROOT}")
        validate_input_file(config.input_filename)

        # Load
        coords, point_ids, lamella = load_data_from_xml(config)

        # Point cloud
        point_cloud = build_point_cloud(coords, point_ids, lamella, config)

        # Fibre index
        fibre_index = build_fibre_index(point_cloud)

        # Memmap
        tmpdir = Path(tempfile.mkdtemp(dir=TMP_ROOT))
        memmap_path = tmpdir / "point_cloud_memmap.dat"
        check_disk_requirements(point_cloud, memmap_path, safety_factor=2.0)

        logger.info(f"Creating memory-mapped point cloud at {memmap_path}")
        dump(point_cloud, memmap_path)
        point_cloud_memmap = load(memmap_path, mmap_mode="r")

        # Fit curves for tangents and orientation
        curve_fits = fit_all_fibres_joblib(
            point_cloud_memmap,
            fibre_index,
            config
        )

        # Tangents (original)
        tangents = compute_all_tangents_joblib(
            point_cloud_memmap,
            fibre_index,
            curve_fits,
            config
        )

        # Orientation (original)
        theta, phi = compute_orientation_angles(tangents)

        # Save original outputs
        logger.info("Saving ORIGINAL outputs")
        save_pointcloud_txt(config.pointcloud_output, point_cloud)
        save_spacecurve_npz(config.spacecurve_output, point_cloud)

        save_orientation_csv(
            config.orientation_output,
            point_cloud,
            theta, phi
        )

        # -------------------------
        # Rotated orientation CSV
        # -------------------------
        logger.info("Computing ROTATED outputs (positions + tangents rotated)")
        R = rodrigues_rotation_matrix(rotationconfig["rotation_axis"], rotationconfig["rotation_angle_deg"])
        centre = rotationconfig["rotation_point"]

        point_cloud_rot = point_cloud.copy()
        point_cloud_rot[:, 4:7] = apply_rotation(point_cloud_rot[:, 4:7], R, centre).astype(np.float32)

        tangents_rot = rotate_vectors(tangents, R).astype(np.float32)
        theta_rot, phi_rot = compute_orientation_angles(tangents_rot)

        save_pointcloud_txt(config.pointcloud_output_rotated, point_cloud_rot)

        save_orientation_csv(
            config.orientation_output_rotated,
            point_cloud_rot,
            theta_rot, phi_rot
        )

        logger.info("Results saved to:")
        logger.info(f"  - {config.pointcloud_output}")
        logger.info(f"  - {config.orientation_output}")
        logger.info(f"  - {config.spacecurve_output}")
        logger.info(f"  - {config.pointcloud_output_rotated}")
        logger.info(f"  - {config.orientation_output_rotated}")

        elapsed = time.time() - start_time
        logger.info(f"Total processing time: {elapsed / 60:.2f} minutes")

    except Exception as e:
        logger.error(f"Processing failed: {e}", exc_info=True)
        raise

    finally:
        if tmpdir is not None:
            cleanup_temp_folder(tmpdir)

if __name__ == "__main__":
    main()
