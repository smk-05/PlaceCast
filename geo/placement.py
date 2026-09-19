"""
The mesh -> ENU placement matrix. Spec 13.1, 13.2.

The solver's FitResult is defined on CANONICAL coordinates: the mesh after its
up-axis is rotated to +Z, its base dropped to z=0 and its XY centred
(geo.outline.canonical_offsets). A viewer holding only the raw glb and the
FitResult therefore cannot place the mesh — it does not know those offsets.

This module composes the whole chain into one 4x4 matrix, written into the
placement record as `mesh_to_enu`:

    ENU = T(tx, ty, z0) . Rz(theta) . S(sx, sy, sz) . T(-o) . R_canon . v

Spec 13.1's rule is kept: nothing is baked into the vertices. The mesh stays in
its native glTF frame and every scrap of georeferencing is in this matrix.
The viewer multiplies it by Cesium's ENU->ECEF frame and loads the glb with
`upAxis: Axis.Z` so Cesium does not ALSO apply its own Y-up rotation (R_canon
already contains it).

tests/test_mesh_to_enu.py asserts the matrix reproduces the solver's placement
exactly, which is what guarantees the map shows what the overlay shows.
"""

from __future__ import annotations

import numpy as np

from contracts import OMBB, FitResult


def homogeneous(r: np.ndarray, t: np.ndarray | None = None) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = r
    if t is not None:
        m[:3, 3] = t
    return m


def mesh_to_canonical(rot: np.ndarray, o: np.ndarray) -> np.ndarray:
    """4x4 for canonical = R @ v - o (from geo.outline.canonical_offsets)."""
    return homogeneous(rot, -np.asarray(o, dtype=float))


def unit_box_to_canonical(ombb: OMBB) -> np.ndarray:
    """4x4 taking Cesium's unit box (centred at the origin, z in [-0.5, 0.5]) to
    the dry-run outline: the footprint OMBB, centred, base at z=0, height 1.

    Matches geo.outline.outline_from_footprint, whose M_0 corners are
    +-u*a/2 +- v*b/2 about the origin with mesh_height_units = 1.
    """
    m = np.eye(4)
    m[:2, 0] = ombb.u * ombb.a
    m[:2, 1] = ombb.v * ombb.b
    m[2, 3] = 0.5          # lift the centred box so its base sits at z=0
    return m


def canonical_to_enu(fit: FitResult) -> np.ndarray:
    """4x4 for the solved similarity: Rz(theta) . S(sx, sy, sz), then translate."""
    c, s = np.cos(fit.theta), np.sin(fit.theta)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale = np.diag([fit.scale_x, fit.scale_y, fit.scale_z])
    return homogeneous(rz @ scale, np.array([fit.tx, fit.ty, fit.z_offset]))


def mesh_to_enu(fit: FitResult, to_canonical: np.ndarray) -> np.ndarray:
    """The full mesh -> ENU metres matrix."""
    return canonical_to_enu(fit) @ to_canonical


def record_entry(matrix: np.ndarray, mesh_frame: str) -> dict:
    """JSON form for the placement record.

    `rows` is row-major for humans; `column_major` is what
    Cesium.Matrix4.fromColumnMajorArray takes. Both describe the same matrix.
    """
    return {
        "mesh_frame": mesh_frame,
        "rows": [[float(x) for x in row] for row in matrix],
        "column_major": [float(x) for x in matrix.T.reshape(-1)],
    }
