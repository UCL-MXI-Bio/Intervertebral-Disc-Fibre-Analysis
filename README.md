# Intervertebral-Disc-Fibre-Analysis

# Fibre orientation analysis

Python scripts for calculating local fibre orientation from Avizo fibre traces and the angle of fibres relative to a mask-derived circumferential direction.

## Installation

Install the dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

## 1. Fibre orientation

`fibre_orientation_theta_phi.py` generates a resampled point cloud, fits splines to the fibre coordinates and calculates local orientation from the spline tangents. It also exports rotated coordinates and orientations using the specified rotation.

**Input:** an Avizo Excel-XML export containing a `Points` worksheet with `X Coord`, `Y Coord`, `Z Coord`, and a `Segments` worksheet with comma-separated `Point IDs`. The script treats point IDs as zero-based indices into the point-coordinate array.

Paths, sampling increment and processing settings are defined in `Config`; rotation settings are defined in `rotationconfig`. The temporary directories are configured for `D:\Temp` and `D:\Temp\Joblib`.

Run:

```bash
python fibre_orientation_theta_phi.py
```

**Outputs:**

- Original and rotated point-cloud text files: point ID, x, y, z; tab-separated, without a header.
- Original and rotated orientation CSV files: `x, y, z, fibre_id, theta, phi`.
- A compressed `.npz` file containing the original resampled point cloud and associated identifiers and interpolation directions.

### Angle conventions

Angles are reported in degrees.

| Angle | Definition |
| --- | --- |
| `theta` | Polar angle from the z axis, folded into [-90, 90]. Vertical tangents give 0 degrees and horizontal tangents give +90 degrees. |
| `phi` | Azimuth in the XY plane, measured from +x towards +y, in [0, 360). |

These angles follow the tangent direction defined by the ordering of points along each trace. Fibres without a successful spline fit receive missing orientation values.

## 2. Orientation relative to the circumference

`fibre_orientation_to_circumference.py` compares each fibre's XY direction with a local circumferential tangent derived from a binary annular mask.

**Inputs:**

- An orientation CSV with columns `x, y, z, fibre_id, theta, phi`.
- A binary 3D TIFF mask with an annular region and central hole in the relevant XY slices.

Set the input/output paths, mask dimensions `(X, Y, Z)`, mask origin (`tif_zero_xyz`) and rotation in the `USER CONFIG` section. Use the appropriate orientation CSV from the first script.

Run:

```bash
python fibre_orientation_to_circumference.py
```

The script constructs up to 12 contour lines per slice, calculates smoothed contour tangents, and selects the nearest sampled contour point in XY on the nearest available z slice.

**Outputs:** a CSV containing fibre coordinates, orientation angles, selected contour information and `angle_to_circumferential_0to90_deg`, plus diagnostic PNGs showing contours and fibre points.

The circumferential angle ranges from **0 degrees** (parallel) to **90 degrees** (perpendicular). It compares XY directions rather than full 3D tangents. Near-vertical fibres receive a missing circumferential angle.

## Coordinate requirements

Fibre and mask coordinates must use consistent units and occupy the same coordinate system. Mask dimensions are voxel counts, not maximum indices. The mask calculation uses voxel-index coordinates without voxel-size scaling; its distance interpretation assumes isotropic voxels. Slice-based contour matching assumes rotations preserve z planes.

The supplied paths and settings are dataset-specific. Retain the settings used for the published analysis when reproducing its results, and configure them for other datasets.

## Citation

If you use these scripts, please cite the publication associated with this repository.
