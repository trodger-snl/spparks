#!/usr/bin/env python3
"""Convert crystal directions (Miller indices) to SPPARKS quaternions.

SPPARKS uses Hamilton quaternions with scalar-first ordering
(q0, qx, qy, qz), stored as the d1..d4 site fields in app_style
additive/texture and app_style potts/quaternion.  The rotation matrix
M(q) returned by SPPARKS' quaternion::to_rotation_matrix has columns
equal to the crystal axes expressed in the lab frame, i.e.

    M(q) @ v_crystal = v_lab

The natural way to specify a substrate orientation is therefore:

    "the crystal direction [hkl] should point along lab +Z (the
     build direction); the crystal direction [uvw] should point
     along lab +X (the scan direction)"

with [uvw] orthogonalized against [hkl] so the two crystal vectors
form a right-handed orthonormal pair.

Examples (run from this directory)::

    # Identity (cube-on-face)
    python crystal_to_quat.py --normal 001 --inplane 100

    # Goss orientation: {110}<001>
    python crystal_to_quat.py --normal 110 --inplane 001

    # 111-fiber with [1-10] along scan
    python crystal_to_quat.py --normal 111 --inplane "1 -1 0"

    # Bunge Euler angles (phi1, Phi, phi2) in degrees, ZXZ convention
    python crystal_to_quat.py --euler-bunge 30 45 0

    # Print as a SPPARKS `set` command for one substrate spin
    python crystal_to_quat.py --normal 110 --inplane 001 --as-set 7
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from typing import Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation


# ----------------------------------------------------------------------
# Miller-index parsing
# ----------------------------------------------------------------------

_MILLER_RE = re.compile(r"-?\d")  # one digit per index in the glued form


def parse_miller(s: str) -> np.ndarray:
    """Parse '[hkl]', '(hkl)', 'hkl', or 'h k l' into a length-3 vector.

    Negative indices may be written as bar-prefixed (``-1`` for 1-bar)
    or whitespace-separated.  Single-digit forms like '111' are read as
    (1, 1, 1).  For two- or three-digit indices, separate with spaces:
    e.g. ``"1 10 0"`` is (1, 10, 0).  The result is NOT normalized;
    the caller decides whether to treat it as a direction or vector.
    """
    s = s.strip().strip("[](){}")
    # Case 1: explicit separators (whitespace or commas)
    if any(c in s for c in " ,\t"):
        toks = re.split(r"[\s,]+", s)
        idx = [int(t) for t in toks if t]
    else:
        # Case 2: glued single-digit form like '111' or '-1-10'
        idx = [int(m.group()) for m in _MILLER_RE.finditer(s)]
    if len(idx) != 3:
        raise ValueError(
            f"could not parse {s!r} as a 3-component Miller index; "
            f"got {idx}.  Use space-separated form for multi-digit "
            f"indices, e.g. '1 10 0'."
        )
    return np.array(idx, dtype=float)


# ----------------------------------------------------------------------
# Direction pair -> SPPARKS quaternion
# ----------------------------------------------------------------------

def directions_to_quaternion(
    crystal_normal: Sequence[float],
    crystal_inplane: Sequence[float],
    tol: float = 1e-9,
) -> np.ndarray:
    """Quaternion that orients the crystal so:

        crystal direction `crystal_normal`  points along lab +Z
        crystal direction `crystal_inplane` points along lab +X
                                            (after orthogonalization)

    Returns ``np.array([q0, qx, qy, qz])`` (scalar-first, SPPARKS form).
    """
    n = np.asarray(crystal_normal, dtype=float)
    p = np.asarray(crystal_inplane, dtype=float)
    if np.linalg.norm(n) < tol or np.linalg.norm(p) < tol:
        raise ValueError("normal and inplane directions must be non-zero")

    n = n / np.linalg.norm(n)
    # Orthogonalize p against n (Gram-Schmidt) and normalize.
    p_perp = p - np.dot(p, n) * n
    if np.linalg.norm(p_perp) < tol:
        raise ValueError(
            "inplane direction is parallel to the normal direction; "
            "pick a non-collinear crystal vector for the in-plane axis"
        )
    p_perp /= np.linalg.norm(p_perp)
    # Right-handed third axis
    third = np.cross(n, p_perp)

    # M maps crystal-frame vectors to lab-frame components.
    # Given (n, p_perp, third) are crystal-frame unit vectors that
    # should align with lab (+Z, +X, +Y) respectively, the rows of M
    # are exactly (p_perp, third, n) so that:
    #     M @ p_perp = (1,0,0),  M @ third = (0,1,0),  M @ n = (0,0,1).
    M = np.vstack([p_perp, third, n])

    rot = Rotation.from_matrix(M)
    qx, qy, qz, q0 = rot.as_quat()  # scipy uses scalar-LAST
    q = np.array([q0, qx, qy, qz])
    # Canonicalize to q0 >= 0 (q and -q represent the same rotation;
    # this makes the output reproducible).
    if q[0] < 0:
        q = -q
    return q


def euler_bunge_to_quaternion(phi1_deg: float, Phi_deg: float,
                              phi2_deg: float) -> np.ndarray:
    """Bunge ZXZ Euler angles (degrees) -> SPPARKS quaternion.

    The Bunge convention rotates the sample frame to the crystal frame
    by phi1 about Z, then Phi about the new X, then phi2 about the new
    Z.  We invert it to go crystal->lab so the result is consistent
    with the directions_to_quaternion convention above.
    """
    g_sample_to_crystal = Rotation.from_euler(
        "ZXZ", [phi1_deg, Phi_deg, phi2_deg], degrees=True
    )
    g_crystal_to_lab = g_sample_to_crystal.inv()
    qx, qy, qz, q0 = g_crystal_to_lab.as_quat()
    q = np.array([q0, qx, qy, qz])
    if q[0] < 0:
        q = -q
    return q


# ----------------------------------------------------------------------
# Inverse: quaternion -> reported crystal directions
# ----------------------------------------------------------------------

def quaternion_to_directions(q: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (crystal_normal_unit, crystal_inplane_unit) from a quaternion.

    Useful for sanity-checking: feed back a quaternion you already have
    and confirm the [hkl] interpretation.
    """
    q = np.asarray(q, dtype=float)
    q0, qx, qy, qz = q / np.linalg.norm(q)
    rot = Rotation.from_quat([qx, qy, qz, q0])
    M = rot.as_matrix()
    # rows of M are (d_x_crystal, d_y_crystal, d_z_crystal) in crystal frame
    return M[2].copy(), M[0].copy()


def quaternion_to_euler_bunge(q: Sequence[float]) -> Tuple[float, float, float]:
    """SPPARKS quaternion -> Bunge ZXZ Euler angles (phi1, Phi, phi2) in degrees.

    Inverse of :func:`euler_bunge_to_quaternion`.  Output is wrapped to
    the canonical Bunge ranges:
        phi1 in [0, 360),  Phi in [0, 180],  phi2 in [0, 360).

    Note: at the gimbal-lock cases Phi == 0 or Phi == 180 the split
    between phi1 and phi2 is degenerate (only their sum or difference
    is well-defined).  scipy resolves the ambiguity by zeroing one of
    them; the returned triple still represents the correct rotation.
    """
    q = np.asarray(q, dtype=float)
    q0, qx, qy, qz = q / np.linalg.norm(q)
    g_crystal_to_lab = Rotation.from_quat([qx, qy, qz, q0])
    g_sample_to_crystal = g_crystal_to_lab.inv()
    # scipy warns on gimbal lock (Phi == 0 or 180) where the phi1/phi2
    # split is ambiguous. The triple still represents the correct
    # rotation; the docstring documents this. Suppress the noise.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected", category=UserWarning
        )
        phi1, Phi, phi2 = g_sample_to_crystal.as_euler("ZXZ", degrees=True)
    return float(phi1 % 360.0), float(Phi), float(phi2 % 360.0)


# ----------------------------------------------------------------------
# Output formatting
# ----------------------------------------------------------------------

def format_quat(q: np.ndarray, precision: int = 8) -> str:
    """Plain `q0 qx qy qz` string with the requested precision."""
    fmt = f"%.{precision}g"
    return " ".join(fmt % v for v in q)


def format_set_command(q: np.ndarray, spin: int, precision: int = 8) -> str:
    """SPPARKS `set` command line that assigns this quaternion to one spin."""
    fmt = f"%.{precision}g"
    return (
        f"set i1 {spin} "
        f"d1 {fmt % q[0]} d2 {fmt % q[1]} "
        f"d3 {fmt % q[2]} d4 {fmt % q[3]}"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert crystal directions to SPPARKS quaternions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--normal", metavar="[hkl]",
        help="crystal direction to align with lab +Z (build direction)",
    )
    src.add_argument(
        "--euler-bunge", nargs=3, type=float,
        metavar=("phi1", "Phi", "phi2"),
        help="Bunge ZXZ Euler angles in degrees",
    )
    src.add_argument(
        "--check", nargs=4, type=float,
        metavar=("q0", "qx", "qy", "qz"),
        help="invert: report which [hkl] this quaternion places along lab Z, X",
    )
    p.add_argument(
        "--inplane", metavar="[uvw]",
        help="crystal direction to align with lab +X (scan direction); "
             "required with --normal",
    )
    p.add_argument(
        "--as-set", type=int, metavar="SPIN",
        help="emit a SPPARKS `set i1 SPIN d1 ... d4 ...` line",
    )
    p.add_argument(
        "--precision", type=int, default=8,
        help="decimal precision for output (default 8)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.check is not None:
        q = np.array(args.check, dtype=float)
        n_dir, x_dir = quaternion_to_directions(q)
        phi1, Phi, phi2 = quaternion_to_euler_bunge(q)
        print(f"normal  (crystal direction along lab +Z): "
              f"{n_dir[0]:.6f} {n_dir[1]:.6f} {n_dir[2]:.6f}")
        print(f"inplane (crystal direction along lab +X): "
              f"{x_dir[0]:.6f} {x_dir[1]:.6f} {x_dir[2]:.6f}")
        print(f"Bunge ZXZ Euler angles (deg): "
              f"phi1 = {phi1:.4f}, Phi = {Phi:.4f}, phi2 = {phi2:.4f}")
        return 0

    if args.normal is not None:
        if args.inplane is None:
            print("error: --normal requires --inplane", file=sys.stderr)
            return 2
        n = parse_miller(args.normal)
        p = parse_miller(args.inplane)
        q = directions_to_quaternion(n, p)
    else:
        phi1, Phi, phi2 = args.euler_bunge
        q = euler_bunge_to_quaternion(phi1, Phi, phi2)

    if args.as_set is not None:
        print(format_set_command(q, args.as_set, args.precision))
    else:
        print(format_quat(q, args.precision))
    return 0


# ----------------------------------------------------------------------
# Self-tests (run with `python crystal_to_quat.py --selftest`)
# ----------------------------------------------------------------------

def _selftest() -> int:
    """Sanity checks. Prints PASS/FAIL per case; returns nonzero on failure."""
    cases: list[tuple[str, np.ndarray, np.ndarray]] = [
        # (description, normal_hkl, inplane_uvw, expected q0_sign)
    ]

    def check(desc, n, p):
        q = directions_to_quaternion(n, p)
        # Sanity: round-trip.
        n_back, x_back = quaternion_to_directions(q)
        n_unit = np.array(n) / np.linalg.norm(n)
        ok_n = np.allclose(n_back, n_unit, atol=1e-9)
        ok_norm = np.isclose(np.linalg.norm(q), 1.0, atol=1e-12)
        status = "PASS" if (ok_n and ok_norm) else "FAIL"
        print(f"  [{status}] {desc}: q = {format_quat(q, 6)}")
        return ok_n and ok_norm

    def check_euler(desc, phi1, Phi, phi2):
        q = euler_bunge_to_quaternion(phi1, Phi, phi2)
        phi1_back, Phi_back, phi2_back = quaternion_to_euler_bunge(q)
        # Re-encode and compare quaternions; this is robust to the
        # gimbal-lock split ambiguity (the input/output triples may
        # differ but must represent the same rotation).
        q_back = euler_bunge_to_quaternion(phi1_back, Phi_back, phi2_back)
        same = np.allclose(q, q_back, atol=1e-9) or np.allclose(q, -q_back, atol=1e-9)
        status = "PASS" if same else "FAIL"
        print(f"  [{status}] Euler round-trip ({desc}): "
              f"({phi1:.1f}, {Phi:.1f}, {phi2:.1f}) -> "
              f"({phi1_back:.4f}, {Phi_back:.4f}, {phi2_back:.4f})")
        return same

    print("Self-tests:")
    ok = True
    ok &= check("identity ([001] || z, [100] || x)", [0, 0, 1], [1, 0, 0])
    ok &= check("Goss ([110] || z, [001] || x)", [1, 1, 0], [0, 0, 1])
    ok &= check("[111] || z, [1-10] || x", [1, 1, 1], [1, -1, 0])
    ok &= check("non-orthogonal inplane gets orthogonalized",
                [0, 0, 1], [1, 1, 0.3])  # third comp irrelevant after GS
    ok &= check_euler("generic", 30.0, 45.0, 17.0)
    ok &= check_euler("generic 2", 217.5, 102.3, 333.0)
    ok &= check_euler("Phi=0 (gimbal lock)", 30.0, 0.0, 0.0)
    print("  OK." if ok else "  FAILED.")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    sys.exit(main())
