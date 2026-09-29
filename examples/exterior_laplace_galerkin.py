#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exterior Laplace Dirichlet-to-Neumann solve with the 2-D Galerkin boundary operator.

The analytic exterior field ``u = d·r/|r|²`` with ``r = x - x0`` (a dipole of
direction ``d`` at ``x0`` inside the square ``[-1, 1]²``) is harmonic outside
the square, decays at infinity, and has zero total conormal. Its Dirichlet
trace is L2-projected onto continuous P1 (a declared projection, reported with
its defect), the bordered exterior system recovers the DP0 conormal and the
far-field constant under a declared decay condition, and the Green
representation ``u = c + Dφ - Sq`` is evaluated near and far from the
boundary. The dipole is off-center so that the discrete far-field constant is
a genuine discretization error rather than zero by symmetry. Panel refinement
reports the observed convergence of every error.
"""

from __future__ import annotations

import math
import time

import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

import phydrax as phx


type _Floats = npt.NDArray[np.float64]

_CENTER = np.asarray((0.25, -0.15))
_DIRECTION = np.asarray((0.8, 0.6))
_NEAR = np.asarray(((1.05, 0.3), (-0.4, -1.08)))
_FAR = np.asarray(((10.0, 4.0), (100.0, -30.0)))


def _square(panels_per_side: int) -> _Floats:
    steps = np.linspace(-1.0, 1.0, panels_per_side + 1)[:-1]
    ones = np.ones(panels_per_side)
    return np.concatenate(
        (
            np.stack((steps, -ones), axis=1),
            np.stack((ones, steps), axis=1),
            np.stack((-steps, ones), axis=1),
            np.stack((-ones, -steps), axis=1),
        )
    )


def _dipole(points: _Floats) -> tuple[_Floats, _Floats]:
    offset = points - _CENTER
    squared = np.sum(offset * offset, axis=-1)
    projection = offset @ _DIRECTION
    gradient = (
        _DIRECTION / squared[..., None]
        - 2.0 * (projection / squared**2)[..., None] * offset
    )
    return projection / squared, gradient


def _refinement(panels_per_side: int) -> dict[str, float | int | bool]:
    started = time.perf_counter()
    curve = phx.operators.ClosedPolygonalCurve2D(
        _square(panels_per_side), source_id="dipole-exterior-square"
    )
    galerkin = phx.operators.prepare_scalar_laplace_galerkin_2d(curve)
    projection = phx.operators.prepare_boundary_trace_projection_2d(
        galerkin.spaces, order=8
    )
    prepared = phx.operators.prepare_exterior_laplace_dirichlet_2d(
        galerkin, far_field="decaying", far_field_tolerance=1.0e-2
    )
    preparation_seconds = time.perf_counter() - started

    samples = np.asarray(projection.sample_points)
    value, gradient = _dipole(samples)
    normals = np.asarray(curve.normals)[:, None, :]
    dirichlet = projection.project_dirichlet(jnp.asarray(value))
    reference = projection.project_conormal(jnp.asarray(np.sum(gradient * normals, -1)))
    started = time.perf_counter()
    result = phx.operators.solve_exterior_laplace_dirichlet_2d(
        prepared, dirichlet.coefficients
    )
    solve_seconds = time.perf_counter() - started

    lengths = np.asarray(curve.lengths)
    conormal_error = float(
        np.sqrt(
            np.sum(lengths * (np.asarray(result.conormal - reference.coefficients)) ** 2)
        )
    )
    targets = np.concatenate((_NEAR, _FAR))
    field = galerkin.evaluate_field(
        targets,
        side="exterior",
        dirichlet=dirichlet.coefficients,
        conormal=result.conormal,
        far_field_constant=result.far_field_constant,
    )
    errors = np.abs(np.asarray(field.values) - _dipole(targets)[0])
    report = galerkin.report
    return {
        "panels": curve.panel_count,
        "near_pairs": report.pair_counts[2],
        "regular_pairs": report.pair_counts[3],
        "promoted_pairs": report.promoted_pair_count,
        "maximum_quadrature_error": float(jnp.max(report.maximum_errors)),
        "quadrature_supported": bool(report.accuracy_supported),
        "resident_bytes": report.resident_bytes,
        "action_workspace_bytes": report.action_workspace_bytes_per_rhs,
        "trace_projection_relative_defect": float(dirichlet.relative_defect),
        "gmres_iterations": int(result.linear.diagnostics.iterations),
        "equation_residual": float(result.equation_residual),
        "far_field_constant": float(result.far_field_constant),
        "total_conormal": float(result.total_conormal),
        "conormal_l2_error": conormal_error,
        "near_field_error": float(np.max(errors[: _NEAR.shape[0]])),
        "far_field_error": float(np.max(errors[_NEAR.shape[0] :])),
        "accepted": bool(result.accepted) and bool(field.accepted),
        "preparation_seconds": preparation_seconds,
        "solve_seconds": solve_seconds,
    }


def _orders(rows: list[dict[str, float | int | bool]], key: str, /) -> list[float]:
    return [
        math.log2(abs(float(coarse[key])) / abs(float(fine[key])))
        for coarse, fine in zip(rows[:-1], rows[1:], strict=True)
    ]


def run() -> dict[str, object]:
    rows = [_refinement(panels_per_side) for panels_per_side in (4, 8, 16, 32)]
    orders = {
        key: _orders(rows, key)
        for key in (
            "far_field_constant",
            "conormal_l2_error",
            "near_field_error",
            "far_field_error",
        )
    }
    for row in rows:
        print(
            f"panels={row['panels']:4d} near={row['near_pairs']:5d} "
            f"regular={row['regular_pairs']:6d} promoted={row['promoted_pairs']:3d} "
            f"quad_err={row['maximum_quadrature_error']:.1e} "
            f"gmres={row['gmres_iterations']:4d} residual={row['equation_residual']:.1e} "
            f"c_inf={row['far_field_constant']:+.2e} total_q={row['total_conormal']:+.1e} "
            f"q_err={row['conormal_l2_error']:.2e} near_err={row['near_field_error']:.2e} "
            f"far_err={row['far_field_error']:.2e} "
            f"prepare={row['preparation_seconds']:.1f}s solve={row['solve_seconds']:.2f}s"
        )
    for key, values in orders.items():
        print(f"observed order {key}: " + ", ".join(f"{value:.2f}" for value in values))
    if not all(
        bool(row["accepted"]) and bool(row["quadrature_supported"]) for row in rows
    ):
        raise RuntimeError("Exterior Galerkin solve or quadrature evidence was rejected.")
    # DP0 conormals converge at least at first order in L2; smooth-data
    # functionals (far-field constant, off-boundary values) at least at second
    # order. Near-boundary values are pre-asymptotic, so mean rates are checked.
    minimum_rates = {
        "far_field_constant": 2.0,
        "conormal_l2_error": 1.0,
        "near_field_error": 2.0,
        "far_field_error": 2.0,
    }
    if any(float(np.mean(orders[key])) < rate for key, rate in minimum_rates.items()):
        raise RuntimeError(
            "Exterior Galerkin refinement did not converge at the expected rate."
        )
    return {"refinements": rows, "observed_orders": orders}


if __name__ == "__main__":
    run()
