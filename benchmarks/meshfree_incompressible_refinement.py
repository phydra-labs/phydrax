"""Taylor-Green refinement study of the meshfree incompressible flow owner.

Periodic lattices ``n x n`` on ``[0, 2 pi)^2`` with the signed exact metric,
radius ``1.1 h``, ``nu = 0.05`` and ``t1 = 1``. Three separated studies:

* joint: ``dt = 0.1 * 12 / n`` for each transport scheme; timings,
  divergence, kinetic-energy decay against ``exp(-4 nu t1)`` and velocity
  errors against the exact vortex, with orders ``log2(err_n / err_2n)``;
* temporal (``--temporal-count``): fixed ``n`` and halved ``dt``; velocity,
  pressure and energy differences against the finest ``dt`` and the
  reference-free Richardson order ``log2(|u_dt - u_dt/2| / |u_dt/2 - u_dt/4|)``;
* spatial (``--spatial-step-size``): every ``n`` at one small ``dt``; velocity,
  pressure and energy errors against the exact vortex.

The pressure is the step pressure whose nodal gradient balances the momentum
update, a midpoint quantity: errors compare it with the exact pressure at
``t1 - dt / 2`` and temporal differences first carry it to ``t1`` with the
exact pressure decay ``exp(-4 nu dt / 2)``; means are removed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import MortonAddressPlan, PointCloudPlan
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeCoercivityPolicy,
    MeshfreeEdgeRelationPlan,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMetricPolicy,
    minimum_image_edge_charts,
    PreparedMeshfreeExteriorCalculus,
    TransportScheme,
)
from phydrax.linalg import SparseFactorizationPolicy
from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver import (
    FixedStepProblem,
    MeshfreeFlowEvidence,
    MeshfreeIncompressibleFlowPlan,
    MeshfreeIncompressibleState,
    PreparedMeshfreeIncompressibleFlow,
    solve_fixed_step,
)
from phydrax.solver.advanced import AdditiveIMEXScheme
from phydrax.typing import parse


LENGTH = 2.0 * np.pi
VISCOSITY = 0.05
FINAL_TIME = 1.0
EXTERIOR_LIMIT_SECONDS = 1800.0
ERROR_KEYS = (
    "relative_energy_error",
    "velocity_max_error",
    "velocity_rms_error",
)


def taylor_green(points: Array, t: float) -> Array:
    x, y = points[:, 0], points[:, 1]
    return jnp.exp(-2.0 * VISCOSITY * t) * jnp.stack(
        (jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=1
    )


def taylor_green_pressure(points: Array, t: float) -> Array:
    x, y = points[:, 0], points[:, 1]
    return 0.25 * jnp.exp(-4.0 * VISCOSITY * t) * (jnp.cos(2.0 * x) + jnp.cos(2.0 * y))


def prepare_exterior(
    count: int, symbolic_work: int
) -> tuple[PreparedMeshfreeExteriorCalculus, MortonAddressPlan, float]:
    spacing = LENGTH / count
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape(-1, 2)
    points = (lattice + 0.5) * spacing
    address = MortonAddressPlan(
        (0.0, 0.0), (LENGTH, LENGTH), 12, periodic_axes=(True, True)
    )
    capacity = 4 * count * count
    start = time.perf_counter()
    relation = MeshfreeEdgeRelationPlan(
        points, 1.1 * spacing, capacity, address=address
    ).prepare()
    exterior = MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        capacity,
        node_volumes=np.full(count * count, spacing**2),
        intrinsic_displacements=minimum_image_edge_charts(relation, points),
        metric_policy=MeshfreeMetricPolicy(),
        coercivity_policy=MeshfreeCoercivityPolicy(
            factorization=SparseFactorizationPolicy(
                "cholesky",
                ordering="approximate-minimum-degree",
                max_symbolic_work=symbolic_work,
            )
        ),
    ).prepare(edge_relation=relation)
    jax.block_until_ready(exterior.lengths)
    return exterior, address, time.perf_counter() - start


def prepare_flow(
    exterior: PreparedMeshfreeExteriorCalculus,
    address: MortonAddressPlan,
    scheme: TransportScheme,
    method: AdditiveIMEXScheme,
) -> PreparedMeshfreeIncompressibleFlow:
    cloud = PointCloudPlan(
        np.asarray(exterior.points),
        np.asarray(exterior.node_volumes),
        address=address,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
    ).prepare()
    return MeshfreeIncompressibleFlowPlan(
        exterior,
        cloud,
        viscosity=VISCOSITY,
        domain_betti_number=2,
        method=method,
        transport_scheme=scheme,
        cycle_tolerance=0.15,
    ).prepare()


def rollout(
    flow: PreparedMeshfreeIncompressibleFlow, step_size: float
) -> tuple[MeshfreeIncompressibleState, dict[str, float | int | bool], float]:
    """Run to ``t1``; return the final state, step evidence and warm seconds."""
    initial = flow.initialize(taylor_green(flow.points, 0.0))
    problem = FixedStepProblem(
        flow,
        initial.state,
        t0=0.0,
        t1=FINAL_TIME,
        step_size=step_size,
        state_geometry=EuclideanStateGeometry(),
    )
    start = time.perf_counter()
    solution = jax.block_until_ready(
        solve_fixed_step(problem, evidence_retention="steps")
    )
    seconds = time.perf_counter() - start
    evidence = solution.evidence.steps
    if not isinstance(evidence, MeshfreeFlowEvidence):
        raise TypeError(
            "Incompressible refinement requires retained MeshfreeFlowEvidence steps."
        )
    final = jax.tree.map(lambda leaf: leaf[-1], solution.states)
    record: dict[str, float | int | bool] = {
        "step_size": step_size,
        "steps_requested": round(FINAL_TIME / step_size),
        "steps_accepted": int(jnp.sum(solution.valid[1:])),
        "run_successful": bool(solution.successful),
        "energy_ratio": float(evidence.kinetic_energy_after[-1])
        / float(evidence.kinetic_energy_before[0]),
        "max_graph_divergence": float(jnp.max(evidence.graph_divergence_norm)),
        "max_nodal_divergence": float(jnp.max(evidence.nodal_divergence_after)),
        "max_momentum": float(jnp.max(jnp.abs(evidence.momentum_after))),
        "cycle_fraction": float(jnp.max(evidence.cycles.nonphysical_fraction)),
        "pressure_iterations": int(jnp.sum(evidence.pressure_iterations)),
    }
    return final, record, seconds


def rms(values: Array) -> float:
    """Root mean square over nodes of the pointwise Euclidean norm."""
    return float(jnp.sqrt(jnp.sum(values**2) / values.shape[0]))


def centered(values: Array) -> Array:
    return values - jnp.mean(values)


def exact_errors(
    flow: PreparedMeshfreeIncompressibleFlow,
    final: MeshfreeIncompressibleState,
    record: dict[str, float | int | bool],
) -> dict[str, float]:
    points = flow.points
    error = final.velocity - taylor_green(points, FINAL_TIME)
    pressure = taylor_green_pressure(
        points, FINAL_TIME - 0.5 * float(record["step_size"])
    )
    exact_ratio = float(np.exp(-4.0 * VISCOSITY * FINAL_TIME))
    return {
        "relative_energy_error": abs(float(record["energy_ratio"]) / exact_ratio - 1.0),
        "velocity_max_error": float(jnp.max(jnp.abs(error))),
        "velocity_rms_error": rms(error),
        "pressure_rms_error": rms(centered(final.pressure) - centered(pressure)),
    }


def order(coarse: float, fine: float) -> float:
    return float(np.log2(coarse / fine))


@dataclass
class _JointRun:
    nodes: int
    exterior_seconds: float
    schemes: dict[TransportScheme, dict[str, float | int | bool]] = field(
        default_factory=dict
    )
    status: str | None = None

    def payload(
        self,
    ) -> dict[str, str | float | int | dict[str, float | int | bool]]:
        result: dict[str, str | float | int | dict[str, float | int | bool]] = {
            "nodes": self.nodes,
            "exterior_seconds": self.exterior_seconds,
            **self.schemes,
        }
        if self.status is not None:
            result["status"] = self.status
        return result


def joint_study(
    counts: list[int],
    schemes: list[TransportScheme],
    method: AdditiveIMEXScheme,
    symbolic_work: int,
) -> dict[str, object]:
    runs: dict[str, _JointRun] = {}
    for count in counts:
        exterior, address, exterior_seconds = prepare_exterior(count, symbolic_work)
        entry = _JointRun(count * count, exterior_seconds)
        if exterior_seconds > EXTERIOR_LIMIT_SECONDS:
            entry.status = "refused: exterior preparation exceeded 30 min"
        else:
            for scheme in schemes:
                start = time.perf_counter()
                flow = prepare_flow(exterior, address, scheme, method)
                plan_seconds = time.perf_counter() - start
                step_size = 0.1 * 12.0 / count
                _, _, compile_seconds = rollout(flow, step_size)
                final, record, warm_seconds = rollout(flow, step_size)
                entry.schemes[scheme] = {
                    **record,
                    **exact_errors(flow, final, record),
                    "plan_seconds": plan_seconds,
                    "compile_seconds": compile_seconds,
                    "warm_seconds": warm_seconds,
                }
        runs[str(count)] = entry
        print(json.dumps({"joint": {str(count): entry.payload()}}), flush=True)
    orders: dict[str, dict[str, float]] = {}
    for scheme in schemes:
        for coarse, fine in zip(counts, counts[1:]):
            a, b = runs[str(coarse)], runs[str(fine)]
            if scheme in a.schemes and scheme in b.schemes:
                orders[f"{scheme}:{coarse}->{fine}"] = {
                    key: order(a.schemes[scheme][key], b.schemes[scheme][key])
                    for key in (*ERROR_KEYS, "pressure_rms_error")
                }
    print(
        f"{'n':>4} {'scheme':>13} {'ext s':>8} {'plan s':>7} {'comp s':>7} "
        f"{'warm s':>7} {'steps':>5} {'graphdiv':>9} {'nodaldiv':>9} "
        f"{'KE err':>9} {'u max':>9} {'u rms':>9} {'p rms':>9} {'mom':>9} {'cycle':>7}"
    )
    for count in counts:
        entry = runs[str(count)]
        for scheme in schemes:
            if scheme not in entry.schemes:
                print(f"{count:>4} {scheme:>13} {entry.exterior_seconds:>8.1f} refused")
                continue
            r = entry.schemes[scheme]
            print(
                f"{count:>4} {scheme:>13} {entry.exterior_seconds:>8.1f} "
                f"{r['plan_seconds']:>7.1f} {r['compile_seconds']:>7.1f} "
                f"{r['warm_seconds']:>7.2f} {r['steps_accepted']:>5} "
                f"{r['max_graph_divergence']:>9.2e} {r['max_nodal_divergence']:>9.2e} "
                f"{r['relative_energy_error']:>9.2e} {r['velocity_max_error']:>9.2e} "
                f"{r['velocity_rms_error']:>9.2e} {r['pressure_rms_error']:>9.2e} "
                f"{r['max_momentum']:>9.2e} {r['cycle_fraction']:>7.4f}"
            )
    for name, values in orders.items():
        print("joint", name, " ".join(f"{k}={v:.2f}" for k, v in values.items()))
    return {"runs": {key: run.payload() for key, run in runs.items()}, "orders": orders}


def temporal_study(
    count: int,
    step_sizes: list[float],
    scheme: TransportScheme,
    method: AdditiveIMEXScheme,
    symbolic_work: int,
) -> dict[str, object]:
    exterior, address, _ = prepare_exterior(count, symbolic_work)
    flow = prepare_flow(exterior, address, scheme, method)
    finals: list[MeshfreeIncompressibleState] = []
    records: list[dict[str, float | int | bool]] = []
    for step_size in step_sizes:
        final, record, _ = rollout(flow, step_size)
        finals.append(final)
        records.append(record)
    reference, reference_record = finals[-1], records[-1]

    def at_final_time(state: MeshfreeIncompressibleState, step_size: float) -> Array:
        # The step pressure is a midpoint quantity; carry it to t1 with the
        # exact Taylor-Green pressure decay so dt-differences are temporal error.
        return centered(state.pressure) * np.exp(-2.0 * VISCOSITY * step_size)

    rows: list[dict[str, float]] = []
    for index, (final, record) in enumerate(zip(finals[:-1], records[:-1], strict=True)):
        row = {
            "step_size": float(record["step_size"]),
            "velocity_difference": rms(final.velocity - reference.velocity),
            "pressure_difference": rms(
                at_final_time(final, float(record["step_size"]))
                - at_final_time(reference, float(reference_record["step_size"]))
            ),
            "energy_difference": abs(
                float(record["energy_ratio"]) / float(reference_record["energy_ratio"])
                - 1.0
            ),
        }
        if index + 2 < len(finals):
            row["richardson_velocity_order"] = order(
                rms(final.velocity - finals[index + 1].velocity),
                rms(finals[index + 1].velocity - finals[index + 2].velocity),
            )
        rows.append(row)
    for coarse, fine in zip(rows, rows[1:]):
        fine["velocity_order"] = order(
            coarse["velocity_difference"], fine["velocity_difference"]
        )
        fine["pressure_order"] = order(
            coarse["pressure_difference"], fine["pressure_difference"]
        )
        fine["energy_order"] = order(
            coarse["energy_difference"], fine["energy_difference"]
        )
    print(
        f"temporal n={count} scheme={scheme} method={method} reference dt={step_sizes[-1]}"
    )
    for row in rows:
        print("temporal", " ".join(f"{k}={v:.3e}" for k, v in row.items()))
    return {
        "count": count,
        "scheme": scheme,
        "reference_step_size": step_sizes[-1],
        "runs": records,
        "rows": rows,
    }


def spatial_study(
    counts: list[int],
    step_size: float,
    scheme: TransportScheme,
    method: AdditiveIMEXScheme,
    symbolic_work: int,
) -> dict[str, object]:
    rows: list[dict[str, float | int]] = []
    for count in counts:
        exterior, address, _ = prepare_exterior(count, symbolic_work)
        flow = prepare_flow(exterior, address, scheme, method)
        final, record, _ = rollout(flow, step_size)
        rows.append({"count": count, **record, **exact_errors(flow, final, record)})
    for coarse, fine in zip(rows, rows[1:]):
        for key in (*ERROR_KEYS, "pressure_rms_error"):
            fine[f"{key}_order"] = order(float(coarse[key]), float(fine[key]))
    print(f"spatial dt={step_size} scheme={scheme} method={method}")
    for row in rows:
        print(
            "spatial",
            " ".join(
                f"{k}={v:.3e}" if isinstance(v, float) else f"{k}={v}"
                for k, v in row.items()
            ),
        )
    return {"step_size": step_size, "scheme": scheme, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", type=int, nargs="*", default=[12, 24, 48])
    parser.add_argument("--schemes", nargs="+", default=["limited", "reconstructed"])
    parser.add_argument("--method", default="ars-222")
    parser.add_argument("--temporal-count", type=int)
    parser.add_argument(
        "--temporal-step-sizes",
        type=float,
        nargs="+",
        default=[0.1, 0.05, 0.025, 0.0125, 0.00625],
    )
    parser.add_argument("--spatial-step-size", type=float)
    parser.add_argument("--skip-joint", action="store_true")
    # The coercivity assessment's default budget (2e6 symbolic work) refuses
    # the n = 48 periodic lattice (AMD work 2.3e6); the study declares a
    # larger budget explicitly for every n.
    parser.add_argument("--symbolic-work", type=int, default=50_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    schemes: list[TransportScheme] = [
        parse(scheme, TransportScheme, "scheme") for scheme in args.schemes
    ]
    method = parse(args.method, AdditiveIMEXScheme, "method")
    report: dict[str, object] = {
        "symbolic_work_budget": args.symbolic_work,
        "method": method,
    }
    if not args.skip_joint:
        report["joint"] = joint_study(args.counts, schemes, method, args.symbolic_work)
    if args.temporal_count is not None:
        report["temporal"] = {
            scheme: temporal_study(
                args.temporal_count,
                args.temporal_step_sizes,
                scheme,
                method,
                args.symbolic_work,
            )
            for scheme in schemes
        }
    if args.spatial_step_size is not None:
        report["spatial"] = {
            scheme: spatial_study(
                args.counts, args.spatial_step_size, scheme, method, args.symbolic_work
            )
            for scheme in schemes
        }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
