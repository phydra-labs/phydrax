#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hard periodic seams composed with physical walls, initial data, and jumps.

Part 1 trains a free MLP for the Poisson problem ``-(u_xx + u_yy) = f`` on the
channel ``[0, 1]^2`` that is periodic in ``x`` with physical walls ``u = 0`` at
``y = 0`` and ``y = 1``. The manufactured solution is
``u = sin(2 pi x) sin(pi y)``. The seam (value and first ``x`` derivative) and the
walls selected by ``physical_boundary`` are enforced by one compiled
``EnforcementProgram``; only the PDE residual is trained.

Part 2 composes the same seam with walls and the initial slice of the heat
equation on ``[0, 1]^2 x [0, T]`` and prints the compiled evidence: route, scope,
endpoint rank and conditioning, required regularity, and the preserved wall and
initial contracts. Exactness is structural, so it is checked on the untrained
model.

Part 3 enforces a pressure-drop seam ``p(L) - p(0) = -dp`` with a periodic flux
and an antiperiodic seam ``w(L) = -w(0)`` and checks the traces against direct
endpoint evaluations.

Part 4 shows two refused compositions: a wall declared on the identified seam,
and incompatible constant jumps at the corner of two periodic directions.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx


jax.config.update("jax_enable_x64", True)

FINAL_TIME = 0.2
PRESSURE_DROP = 0.75
LENGTH = 2.0
TRAINING_STEPS = 300


def _max_abs(values: ArrayLike, /) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(values))))


def _rms(values: ArrayLike, /) -> float:
    return float(jnp.sqrt(jnp.mean(jnp.abs(jnp.asarray(values)) ** 2)))


def _unit_box(dimension: int, /) -> phx.domain.HyperRectangle:
    return phx.domain.HyperRectangle(
        jnp.zeros((dimension,), dtype=jnp.float64),
        jnp.ones((dimension,), dtype=jnp.float64),
    )


def _mlp(in_size: int, key: Array, /) -> phx.nn.models.MLP:
    return phx.nn.models.MLP(
        in_size=in_size,
        out_size="scalar",
        width_size=32,
        depth=2,
        activation=jnp.tanh,
        key=key,
    )


def _seam_conditions(
    seam: phx.domain.PairedSupport, /
) -> tuple[phx.conditions.Periodic, ...]:
    return (
        phx.conditions.Periodic("u", seam, label="seam-value"),
        phx.conditions.Periodic("u", seam, order=1, label="seam-flux"),
    )


def _report_residuals(
    conditions: tuple[phx.conditions.AbstractResidualCondition, ...],
    field: phx.domain.DomainFunction,
    /,
) -> None:
    for index, condition in enumerate(conditions):
        points = condition.on.sample(phx.domain.PointSampling(64), key=jr.key(index))
        residual = condition.residual({"u": field})(points).data
        print(f"  {condition.label}: max residual {_max_abs(residual):.1e}")


def _poisson_channel() -> None:
    box = _unit_box(2)
    periodic_x = phx.domain.PeriodicIdentification(box, "x", component=0)
    seam_conditions = _seam_conditions(periodic_x.pairing())
    functions = {"u": box.Model("x")(_mlp(2, jr.key(0)))}
    prepared = phx.enforcement.prepare_periodic_projection(
        functions, seam_conditions, route="analytic"
    )
    # The identified x faces are seams; the physical boundary is the two y walls.
    wall_conditions = tuple(
        phx.conditions.Dirichlet("u", wall, target=0.0, label=f"wall-{index}")
        for index, wall in enumerate(phx.domain.physical_boundary(box, (periodic_x,)))
    )
    program = phx.enforcement.compile(
        functions,
        (
            *(phx.enforcement.EnforcementSpec(value) for value in wall_conditions),
            phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),
        ),
        key=jr.key(1),
    )

    interior = box.component()
    source = box.Function("x")(
        lambda x: 5.0 * jnp.pi**2 * jnp.sin(2.0 * jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    )
    poisson = phx.conditions.Residual(
        "u", interior, lambda f: phx.operators.laplacian(f, var="x") + source
    )
    solver = phx.solver.FunctionalSolver(
        functions=functions,
        terms=(
            phx.terms.ResidualPenalty(
                poisson,
                phx.integration.per_step(
                    phx.integration.mean_over(interior), phx.domain.PointSampling(256)
                ),
            ),
        ),
        enforcement=program,
    )
    exact = box.Function("x")(
        lambda x: jnp.sin(2.0 * jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
    )
    check = interior.sample(phx.domain.PointSampling(1024), key=jr.key(2))

    def report(stage: str, field: phx.domain.DomainFunction, /) -> None:
        pde = _rms(poisson.residual({"u": field})(check).data)
        error = _rms((field - exact)(check).data)
        print(f"  {stage}: rms PDE residual={pde:.3e} rms |u - u_exact|={error:.3e}")

    print("Part 1: periodic Poisson channel")
    report("before training", solver.ansatz_functions()["u"])
    started = time.perf_counter()
    trained = solver.solve(
        num_iter=TRAINING_STEPS, optim=optax.adam(5e-3), seed=0, log_every=0
    )
    elapsed = time.perf_counter() - started
    u_hat = trained.ansatz_functions()["u"]
    report(f"after {TRAINING_STEPS} steps ({elapsed:.1f} s)", u_hat)
    _report_residuals((*seam_conditions, *wall_conditions), u_hat)


def _heat_composition() -> None:
    box = _unit_box(2)
    domain = box @ phx.domain.TimeInterval(0.0, FINAL_TIME)
    periodic_x = phx.domain.PeriodicIdentification(domain, "x", component=0)
    seam_conditions = _seam_conditions(periodic_x.pairing())
    functions = {"u": domain.Model("x", "t")(_mlp(3, jr.key(3)))}
    # The initial data sin(2 pi x) is periodic, but a generic function cannot be
    # proven seam-compatible; opt in to sampled compatibility evidence, which the
    # compiler records as "probed" rather than claiming preservation.
    prepared = phx.enforcement.prepare_periodic_projection(
        functions, seam_conditions, route="analytic", data_compatibility="probed"
    )
    # The physical boundary of the space-time box is the y walls plus the two
    # time slices; only the y walls carry Dirichlet data.
    walls = tuple(
        face
        for face in phx.domain.physical_boundary(domain, (periodic_x,))
        if isinstance(face.spec.selection_for("x"), phx.domain.CoordinateFace)
    )
    wall_conditions = tuple(
        phx.conditions.Dirichlet("u", wall, target=0.0, label=f"wall-{index}")
        for index, wall in enumerate(walls)
    )
    initial = phx.conditions.Initial(
        "u",
        domain.component({"t": phx.domain.FixedStart()}),
        target=domain.Function("x")(
            lambda x: jnp.sin(2.0 * jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1])
        ),
        label="initial",
    )
    program = phx.enforcement.compile(
        functions,
        (
            *(phx.enforcement.EnforcementSpec(value) for value in wall_conditions),
            phx.enforcement.EnforcementSpec(initial),
            phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),
        ),
        key=jr.key(4),
    )
    (spec,) = program.realization_specs
    if not isinstance(spec.realization, phx.enforcement.PreparedPeriodicProjection):
        raise TypeError("The compiled program lost its periodic realization.")
    evidence = spec.realization.evidence
    print("Part 2: periodic heat channel with walls and initial data")
    print(f"  route/scope: {evidence.route}/{evidence.scope}")
    for axis in evidence.axes:
        print(
            f"  seam axis: rows={axis.rows} rank={axis.rank} basis={axis.basis_size} "
            f"cond={axis.condition_number:.3e} max_order={axis.max_order}"
        )
    print(
        f"  required regularity (field, coordinate, order): {evidence.required_regularity}"
    )
    print(f"  endpoint evaluations per query: {evidence.endpoint_evaluations}")
    for record in evidence.preserved:
        print(
            f"  preserves {record.contract_id}: {record.status} "
            f"(probes={record.probes}, max defect={record.maximum_defect:.1e})"
        )
    enforced = program.apply(functions)["u"]
    _report_residuals((*seam_conditions, *wall_conditions, initial), enforced)


def _transported_seams() -> None:
    line = phx.domain.Interval1d(0.0, LENGTH)
    seam = phx.domain.PeriodicIdentification(line, "x").pairing()

    def free_model(key: Array, /) -> phx.domain.DomainFunction:
        return line.Model("x")(
            phx.nn.models.MLP(
                in_size="scalar",
                out_size="scalar",
                width_size=16,
                depth=2,
                activation=jnp.tanh,
                key=key,
            )
        )

    functions = {"p": free_model(jr.key(10)), "w": free_model(jr.key(11))}
    conditions = (
        phx.conditions.Periodic("p", seam, target=-PRESSURE_DROP, label="pressure-drop"),
        phx.conditions.Periodic("p", seam, order=1, label="periodic-flux"),
        phx.conditions.Periodic("w", seam, transport=-1.0, label="antiperiodic"),
        phx.conditions.Periodic(
            "w", seam, order=1, transport=-1.0, label="antiperiodic-flux"
        ),
    )
    prepared = phx.enforcement.prepare_periodic_projection(
        functions, conditions, route="analytic"
    )
    program = phx.enforcement.compile(
        functions,
        (phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),),
    )
    fields = program.apply(functions)

    endpoints = line.component().points({"x": jnp.asarray([[0.0], [LENGTH]])})
    p_values = jnp.asarray(fields["p"](endpoints).data)
    slope = phx.operators.partial_n(fields["p"], var="x", order=1)
    dp_values = jnp.asarray(slope(endpoints).data)
    w_values = jnp.asarray(fields["w"](endpoints).data)
    print("Part 3: affine and antiperiodic seams")
    print(
        f"  condition is certified linear: {prepared.condition.operator.capabilities.is_linear}"
    )
    print(
        f"  p(L) - p(0) = {float(p_values[1] - p_values[0]):+.15f} "
        f"(target {-PRESSURE_DROP:+.15f})"
    )
    print(f"  p'(L) - p'(0) = {float(dp_values[1] - dp_values[0]):+.1e}")
    print(f"  w(L) + w(0) = {float(w_values[1] + w_values[0]):+.1e}")
    for axis in prepared.evidence.axes:
        print(
            f"  {axis.field}: rows={axis.rows} rank={axis.rank} "
            f"cond={axis.condition_number:.3e}"
        )


def _refusals() -> None:
    print("Part 4: refused compositions")
    line = phx.domain.Interval1d(0.0, 1.0)
    seam = phx.domain.PeriodicIdentification(line, "x").pairing()
    u = line.Function("x")(lambda x: jnp.cos(x[0]))
    prepared = phx.enforcement.prepare_periodic_projection(
        {"u": u}, (phx.conditions.Periodic("u", seam),), route="analytic"
    )
    wall = phx.conditions.Dirichlet(
        "u", line.component({"x": phx.domain.Boundary()}), target=0.0, label="wall"
    )
    try:
        phx.enforcement.compile(
            {"u": u},
            (
                phx.enforcement.EnforcementSpec(wall),
                phx.enforcement.EnforcementSpec(prepared.condition, realization=prepared),
            ),
        )
    except ValueError as error:
        print(f"  seam used as a wall: {error}")
    else:
        raise RuntimeError("A wall on an identified seam was not refused.")

    box = _unit_box(2)
    v = box.Function("x")(lambda x: x[0] * x[1])
    jump_x = phx.conditions.Periodic(
        "v",
        phx.domain.PeriodicIdentification(box, "x", component=0).pairing(),
        target=1.0,
    )
    antiperiodic_y = phx.conditions.Periodic(
        "v",
        phx.domain.PeriodicIdentification(box, "x", component=1).pairing(),
        transport=-1.0,
    )
    try:
        phx.enforcement.prepare_periodic_projection(
            {"v": v}, (jump_x, antiperiodic_y), route="analytic"
        )
    except ValueError as error:
        print(f"  incompatible corner data: {error}")
    else:
        raise RuntimeError("Incompatible seam targets were not refused.")


def main() -> None:
    _poisson_channel()
    _heat_composition()
    _transported_seams()
    _refusals()


if __name__ == "__main__":
    main()
