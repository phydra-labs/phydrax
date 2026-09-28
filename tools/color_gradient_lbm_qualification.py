#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""N-color color-gradient lattice Boltzmann qualification campaigns (CPU scale).

Scenarios (``--scenario`` selects a subset; default runs all). All runs use D2Q9/BGK
with Guo forcing in lattice units (``dx = dt = rho0 = 1``). Baseline scenarios use
``nu = 1/6``:

- ``binary-regression``: the ``N = 2`` route on a static periodic drop against the
  Laplace law ``dp = sigma / R`` and against the pre-cutover binary CSF route measured
  on the identical configuration;
- ``ternary-neumann``: a liquid lens of ``c`` on a flat ``a|b`` interface; the cap
  angles ``theta = 2 atan(2 h / D)`` against the 3-4-5 Neumann triangle. It retains
  the ``nu = 1/6`` weakly damped run as mode evidence and qualifies equilibrium with
  declared higher BGK damping on two proportionally scaled periodic domains;
- ``momentum``: a ternary periodic rollout with near-contact repulsion; total momentum
  and component masses against their initial values;
- ``emulsion-no-merger``: two drops of one color pressed together across a thin film;
  the control without repulsion merges, the repelled pair stays separate for the
  declared interval, and the repulsion work ledger is recorded.

Every record carries the runtime identity, the discretization, the measurements, the
reference, the criterion and whether it passed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage

import phydrax as phx
from phydrax._fingerprint import canonical_fingerprint
from phydrax.discretization.lattice_boltzmann import (
    BGKCollisionPlan,
    color_gradient_candidate_profiles,
    ColorGradientLBMMethod,
    ColorGradientLBMRuntimeParameters,
    ColorGradientLBMState,
    D2Q9,
    GuoForcingPlan,
    LatticeBoltzmannBoundaryPlan,
    LatticeBoltzmannMethodPlan,
    LatticeBoltzmannPlan,
    NearContactRepulsionPlan,
)
from phydrax.equations import (
    ColorGradientLatticeBoltzmannProblem,
    compile_color_gradient_lattice_boltzmann_problem,
    CompiledColorGradientLatticeBoltzmannProblem,
)
from phydrax.interfacial_transport import InterfaceTensionMatrix


VISCOSITY = 1.0 / 6.0
EQUILIBRIUM_VISCOSITY = 1.0 / 3.0
# Pre-cutover binary CSF route (red/blue API) on the `binary-regression` configuration,
# measured in this worktree before the N-color cutover (2000 steps, float64, CPU).
PREVIOUS_BINARY_PRESSURE_JUMP = 6.128038734666394e-4
PREVIOUS_BINARY_MAXIMUM_SPEED = 1.1636176361664281e-05


def _runtime_identity() -> dict[str, object]:
    build_id = canonical_fingerprint(
        {
            "kind": "color-gradient-lbm-qualification-build",
            "phydrax": importlib.metadata.version("phydrax"),
            "phydrax_path": str(Path(phx.__file__).resolve().parent),
        }
    )
    environment_id = canonical_fingerprint(
        {
            "kind": "color-gradient-lbm-qualification-environment",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "jax": jax.__version__,
            "numpy": np.__version__,
        }
    )
    identity = phx.qualification.QualificationRuntimeIdentity(
        build_id,
        environment_id,
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return dict(identity.to_record())


def _problem(
    components: Sequence[str],
    shape: tuple[int, int],
    near_contact: NearContactRepulsionPlan | None = None,
) -> CompiledColorGradientLatticeBoltzmannProblem:
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(n, periodic=True) for n in shape),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (float(shape[0]), float(shape[1])))))
    discretization = LatticeBoltzmannPlan(grid, D2Q9()).prepare()
    method = ColorGradientLBMMethod(
        LatticeBoltzmannMethodPlan(BGKCollisionPlan(), forcing=GuoForcingPlan()),
        components,
        near_contact=near_contact,
        maximum_capillary_number=10.0,
    )
    return compile_color_gradient_lattice_boltzmann_problem(
        ColorGradientLatticeBoltzmannProblem("qualification", 2),
        discretization,
        method,
        LatticeBoltzmannBoundaryPlan(),
        time_step=1.0,
    )


def _coordinates(
    problem: CompiledColorGradientLatticeBoltzmannProblem,
) -> tuple[np.ndarray, np.ndarray]:
    grid = problem.discretization.grid
    points = np.asarray(grid.points)
    return points[:, 0].reshape(grid.shape), points[:, 1].reshape(grid.shape)


def _disk(
    x: np.ndarray, y: np.ndarray, center: tuple[float, float], radius: float
) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh((radius - np.hypot(x - center[0], y - center[1])) / 2.0))


def _rollout(
    problem: CompiledColorGradientLatticeBoltzmannProblem,
    parameters: ColorGradientLBMRuntimeParameters,
    steps: int,
) -> Callable[[ColorGradientLBMState], tuple[ColorGradientLBMState, Any]]:
    dynamics = problem.dynamics

    @jax.jit
    def run(state: ColorGradientLBMState) -> tuple[ColorGradientLBMState, Any]:
        def body(
            carry: tuple[ColorGradientLBMState, Any], index: Any
        ) -> tuple[tuple[ColorGradientLBMState, Any], None]:
            current, successful = carry
            result = dynamics.step_detailed(index, 0.0, current, 1.0, parameters)
            return (result.accepted_state, successful & result.successful), None

        carry, _ = jax.lax.scan(body, (state, jnp.asarray(True)), jnp.arange(steps))
        return carry

    return run


def _periodic_components(mask: np.ndarray) -> int:
    """Connected components of a boolean field on a doubly periodic grid."""

    labels, count = ndimage.label(mask)
    parent = list(range(count + 1))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for first, second in (
        (labels[0, :], labels[-1, :]),
        (labels[:, 0], labels[:, -1]),
    ):
        for left, right in zip(first, second, strict=True):
            if left and right:
                parent[find(int(left))] = find(int(right))
    return len({find(label) for label in range(1, count + 1)})


def _pair_tension(
    labels: Sequence[str], values: dict[tuple[str, str], float]
) -> InterfaceTensionMatrix:
    matrix = np.zeros((len(labels), len(labels)))
    for (first, second), value in values.items():
        i, j = labels.index(first), labels.index(second)
        matrix[i, j] = matrix[j, i] = value
    return InterfaceTensionMatrix(labels, matrix)


def binary_regression() -> dict[str, object]:
    tension = 0.01
    radius = 16.0
    steps = 2000
    problem = _problem(("red", "blue"), (64, 64))
    x, y = _coordinates(problem)
    red = _disk(x, y, (32.0, 32.0), radius)
    parameters = ColorGradientLBMRuntimeParameters(
        VISCOSITY, _pair_tension(("red", "blue"), {("red", "blue"): tension})
    )
    state = problem.initialize_state(
        np.stack((red, 1.0 - red)), jnp.zeros((2,)), parameters
    )
    final, successful = _rollout(problem, parameters, steps)(state)
    macro = problem.macroscopic_state(final, parameters)
    pressure = np.asarray(macro.pressure)
    distance = np.hypot(x - 32.0, y - 32.0)
    jump = float(pressure[distance < 8.0].mean() - pressure[distance > 26.0].mean())
    speed = float(np.max(np.linalg.norm(np.asarray(macro.velocity), axis=-1)))
    laplace = tension / radius
    laplace_error = abs(jump - laplace) / laplace
    previous_difference = abs(jump - PREVIOUS_BINARY_PRESSURE_JUMP) / abs(
        PREVIOUS_BINARY_PRESSURE_JUMP
    )
    speed_ratio = speed / PREVIOUS_BINARY_MAXIMUM_SPEED
    return {
        "discretization": {"grid": [64, 64], "radius": radius, "steps": steps},
        "tension": tension,
        "successful": bool(successful),
        "pressure_jump": jump,
        "laplace_reference": laplace,
        "laplace_relative_error": laplace_error,
        "previous_pressure_jump": PREVIOUS_BINARY_PRESSURE_JUMP,
        "previous_relative_difference": previous_difference,
        "maximum_speed": speed,
        "previous_maximum_speed": PREVIOUS_BINARY_MAXIMUM_SPEED,
        "criterion": "Laplace error <= 3%, |dp - dp_previous| <= 1%, "
        "spurious speed within 10% of the previous route",
        "passed": bool(successful)
        and laplace_error <= 0.03
        and previous_difference <= 0.01
        and abs(speed_ratio - 1.0) <= 0.1,
    }


def _crossing(values: np.ndarray, coordinates: np.ndarray, level: float) -> float:
    """Linear-interpolated coordinate of the first ``level`` crossing in ``values``."""

    above = values > level
    index = int(np.flatnonzero(above[1:] != above[:-1])[0])
    fraction = (level - values[index]) / (values[index + 1] - values[index])
    return float(
        coordinates[index] + fraction * (coordinates[index + 1] - coordinates[index])
    )


def _lens_angles(concentrations: np.ndarray, interface_guess: float) -> dict[str, float]:
    a, _, c = concentrations
    nx, ny = a.shape
    rows = np.arange(ny) + 0.5
    window = (rows > interface_guess - 20.0) & (rows < interface_guess + 20.0)
    level = _crossing(a[4, window], rows[window], 0.5)
    lower = int(np.floor(level - 0.5))
    weight = level - 0.5 - lower
    lens_row = (1.0 - weight) * c[:, lower] + weight * c[:, lower + 1]
    columns = np.arange(nx) + 0.5
    inside = np.flatnonzero(lens_row > 0.5)
    left = _crossing(
        lens_row[inside[0] - 3 : inside[0] + 1],
        columns[inside[0] - 3 : inside[0] + 1],
        0.5,
    )
    right = _crossing(
        lens_row[inside[-1] : inside[-1] + 4], columns[inside[-1] : inside[-1] + 4], 0.5
    )
    width = right - left
    center = int(np.floor(0.5 * (left + right)))
    column = c[center]
    occupied = np.flatnonzero(column > 0.5)
    bottom = _crossing(
        column[occupied[0] - 3 : occupied[0] + 1],
        rows[occupied[0] - 3 : occupied[0] + 1],
        0.5,
    )
    top = _crossing(
        column[occupied[-1] : occupied[-1] + 4],
        rows[occupied[-1] : occupied[-1] + 4],
        0.5,
    )
    depth = level - bottom
    height = top - level
    return {
        "interface_level": level,
        "width": width,
        "depth_in_a": depth,
        "height_in_b": height,
        "theta_a_degrees": float(np.degrees(2.0 * np.arctan(2.0 * depth / width))),
        "theta_b_degrees": float(np.degrees(2.0 * np.arctan(2.0 * height / width))),
    }


def _neumann_angles(ab: float, ac: float, bc: float) -> tuple[float, float]:
    theta_a = np.arccos((ab**2 + ac**2 - bc**2) / (2.0 * ab * ac))
    theta_b = np.arccos((ab**2 + bc**2 - ac**2) / (2.0 * ab * bc))
    return float(np.degrees(theta_a)), float(np.degrees(theta_b))


def _ternary_lens_case(
    shape: tuple[int, int],
    radius: float,
    viscosity: float,
    checkpoints: tuple[int, ...],
    /,
) -> dict[str, Any]:
    labels = ("a", "b", "c")
    tensions = {("a", "b"): 0.01, ("a", "c"): 0.008, ("b", "c"): 0.006}
    lower_interface = 0.25 * shape[1]
    lens_interface = 0.75 * shape[1]
    problem = _problem(labels, shape)
    x, y = _coordinates(problem)
    band = 0.5 * (
        np.tanh((y - lower_interface) / 2.0)
        - np.tanh((y - lens_interface) / 2.0)
    )
    lens = _disk(x, y, (0.5 * shape[0], lens_interface), radius)
    densities = np.stack(
        ((1.0 - lens) * band, (1.0 - lens) * (1.0 - band), lens)
    )
    matrix = _pair_tension(labels, tensions)
    admissibility = matrix.admissibility()
    parameters = ColorGradientLBMRuntimeParameters(viscosity, matrix)
    state = problem.initialize_state(densities, jnp.zeros((2,)), parameters)
    # Every declared campaign uses equal checkpoint intervals, so one compiled scan is
    # reused instead of compiling a different scan length for every observation.
    interval = checkpoints[0]
    if checkpoints != tuple(range(interval, checkpoints[-1] + 1, interval)):
        raise ValueError("Ternary-lens checkpoints must have equal intervals.")
    run = _rollout(problem, parameters, interval)
    history: list[dict[str, float]] = []
    successful = True
    for _ in checkpoints:
        state, ok = run(state)
        successful = successful and bool(ok)
        macro = problem.macroscopic_state(state, parameters)
        measurement = _lens_angles(np.asarray(macro.concentrations), lens_interface)
        measurement["maximum_speed"] = float(
            np.max(np.linalg.norm(np.asarray(macro.velocity), axis=-1))
        )
        history.append(measurement)
    expected_a, expected_b = _neumann_angles(
        tensions[("a", "b")], tensions[("a", "c")], tensions[("b", "c")]
    )
    final = history[-1]
    late = history[-3:]
    error = max(
        abs(final["theta_a_degrees"] - expected_a),
        abs(final["theta_b_degrees"] - expected_b),
    )
    drift = max(
        abs(history[-1]["theta_a_degrees"] - history[-2]["theta_a_degrees"]),
        abs(history[-1]["theta_b_degrees"] - history[-2]["theta_b_degrees"]),
    )
    late_range = max(
        float(np.ptp([item["theta_a_degrees"] for item in late])),
        float(np.ptp([item["theta_b_degrees"] for item in late])),
    )
    relaxation_time = 0.5 + 3.0 * viscosity
    return {
        "discretization": {
            "grid": list(shape),
            "lens_radius": radius,
            "domain_in_lens_radii": [shape[0] / radius, shape[1] / radius],
            "periodic_interface_separation_in_lens_radii": 0.5 * shape[1] / radius,
            "checkpoints": list(checkpoints),
        },
        "kinematic_viscosity": viscosity,
        "bgk_relaxation_time": relaxation_time,
        "bgk_relaxation_rate": 1.0 / relaxation_time,
        "reference_ohnesorge_number": (
            viscosity / float(np.sqrt(tensions[("a", "b")] * radius))
        ),
        "ohnesorge_reference": "rho = 1 and sigma_ab = 0.01 in lattice units",
        "tensions": {f"{k[0]}{k[1]}": value for k, value in tensions.items()},
        "strict_triangle_inequality": bool(admissibility.strict_triangle_inequality),
        "successful": successful,
        "measurements": dict(zip(map(str, checkpoints), history, strict=True)),
        "neumann_theta_a_degrees": expected_a,
        "neumann_theta_b_degrees": expected_b,
        "final_theta_a_degrees": final["theta_a_degrees"],
        "final_theta_b_degrees": final["theta_b_degrees"],
        "maximum_angle_error_degrees": error,
        "final_interval_angle_drift_degrees": drift,
        "late_angle_range_degrees": late_range,
    }


def ternary_neumann() -> dict[str, object]:
    low_damping = _ternary_lens_case(
        (160, 100), 18.0, VISCOSITY, (4000, 8000, 12000, 16000)
    )
    resolutions = {
        "radius-16": _ternary_lens_case(
            (144, 128),
            16.0,
            EQUILIBRIUM_VISCOSITY,
            tuple(range(2500, 17501, 2500)),
        ),
        "radius-18": _ternary_lens_case(
            (162, 144),
            18.0,
            EQUILIBRIUM_VISCOSITY,
            tuple(range(2500, 22501, 2500)),
        ),
    }
    for case in resolutions.values():
        case["passed"] = (
            bool(case["successful"])
            and float(case["maximum_angle_error_degrees"]) <= 5.0
            and float(case["final_interval_angle_drift_degrees"]) <= 1.0
            and float(case["late_angle_range_degrees"]) <= 1.0
        )
    coarse = resolutions["radius-16"]
    fine = resolutions["radius-18"]
    resolution_difference = max(
        abs(
            float(coarse["final_theta_a_degrees"])
            - float(fine["final_theta_a_degrees"])
        ),
        abs(
            float(coarse["final_theta_b_degrees"])
            - float(fine["final_theta_b_degrees"])
        ),
    )
    resolution_converged = resolution_difference <= 2.0
    return {
        "target": {
            "tension_triangle": "3-4-5",
            "theta_a_degrees": low_damping["neumann_theta_a_degrees"],
            "theta_b_degrees": low_damping["neumann_theta_b_degrees"],
            "viscosity_independent": True,
        },
        "low_damping_capillary_mode_evidence": {
            **low_damping,
            "within_angle_tolerance": (
                float(low_damping["maximum_angle_error_degrees"]) <= 5.0
            ),
            "steady": float(low_damping["late_angle_range_degrees"]) <= 1.0,
        },
        "equilibrium_damping_regime": {
            "kinematic_viscosity": EQUILIBRIUM_VISCOSITY,
            "bgk_relaxation_time": 1.5,
            "bgk_relaxation_rate": 2.0 / 3.0,
            "domain": "9R x 8R periodic, with the companion a|b interface 4R away",
            "transient_control": "physical BGK viscosity; no state smoothing or clipping",
        },
        "resolutions": resolutions,
        "resolution_angle_difference_degrees": resolution_difference,
        "resolution_converged": resolution_converged,
        "criterion": "at both resolutions: final cap angles within 5 degrees of "
        "the Neumann triangle, final-interval drift and last-three-checkpoint range "
        "<= 1 degree; final angles agree across resolutions within 2 degrees",
        "passed": resolution_converged
        and all(bool(case["passed"]) for case in resolutions.values()),
    }


def momentum() -> dict[str, object]:
    labels = ("a", "b", "c")
    shape = (64, 64)
    steps = 2000
    velocity = (0.02, -0.01)
    problem = _problem(
        labels, shape, NearContactRepulsionPlan((("a", "b"),), interaction_range=3)
    )
    x, y = _coordinates(problem)
    a = _disk(x, y, (22.0, 30.0), 11.0)
    b = (1.0 - a) * _disk(x, y, (40.0, 36.0), 9.0)
    densities = np.stack((a, b, 1.0 - a - b))
    parameters = ColorGradientLBMRuntimeParameters(
        VISCOSITY,
        _pair_tension(labels, {("a", "b"): 0.004, ("a", "c"): 0.006, ("b", "c"): 0.005}),
        near_contact_strength=np.asarray([0.002]),
    )
    state = problem.initialize_state(densities, jnp.asarray(velocity), parameters)
    initial = problem.dynamics.scalar_diagnostics(0, 0.0, state, parameters)
    final_state, successful = _rollout(problem, parameters, steps)(state)
    final = problem.dynamics.scalar_diagnostics(0, 0.0, final_state, parameters)
    momentum_drift = float(
        np.linalg.norm(np.asarray(final.total_momentum - initial.total_momentum))
        / np.linalg.norm(np.asarray(initial.total_momentum))
    )
    mass_drift = float(
        np.max(
            np.abs(np.asarray(final.component_masses - initial.component_masses))
            / np.asarray(initial.component_masses)
        )
    )
    return {
        "discretization": {"grid": list(shape), "steps": steps},
        "successful": bool(successful),
        "initial_momentum": np.asarray(initial.total_momentum).tolist(),
        "final_momentum": np.asarray(final.total_momentum).tolist(),
        "relative_momentum_drift": momentum_drift,
        "relative_component_mass_drift": mass_drift,
        "capillary_net_force_residual": float(final.capillary_net_force_residual),
        "near_contact_work": float(final_state.near_contact_work),
        "criterion": "relative momentum drift <= 1e-10 and component mass drift <= 1e-12",
        "passed": bool(successful)
        and momentum_drift <= 1.0e-10
        and mass_drift <= 1.0e-12,
    }


def _two_drop_emulsion(
    problem: CompiledColorGradientLatticeBoltzmannProblem,
    parameters: ColorGradientLBMRuntimeParameters,
    *,
    radius: float,
    gap: float,
    approach_speed: float,
    steps: int,
    chunk: int,
) -> tuple[bool, list[int], list[float]]:
    x, y = _coordinates(problem)
    nx, ny = problem.discretization.grid.shape
    offset = radius + 0.5 * gap
    left = _disk(x, y, (0.5 * nx - offset, 0.5 * ny), radius)
    right = _disk(x, y, (0.5 * nx + offset, 0.5 * ny), radius)
    velocity = np.zeros((nx, ny, 2))
    velocity[..., 0] = approach_speed * (left - right)
    oil = left + right
    state = problem.initialize_state(np.stack((1.0 - oil, oil)), velocity, parameters)
    run = _rollout(problem, parameters, chunk)
    counts = []
    work = []
    successful = True
    for _ in range(steps // chunk):
        state, ok = run(state)
        successful = successful and bool(ok)
        concentrations = np.asarray(
            problem.macroscopic_state(state, parameters).concentrations
        )
        counts.append(_periodic_components(concentrations[1] > 0.5))
        work.append(float(state.near_contact_work))
    return successful, counts, work


def emulsion_no_merger() -> dict[str, object]:
    shape = (96, 64)
    radius = 14.0
    gap = 4.0
    approach_speed = 0.01
    steps = 3000
    chunk = 250
    strength = 0.01
    labels = ("water", "oil")
    tension = _pair_tension(labels, {("water", "oil"): 0.01})
    control_successful, control_counts, _ = _two_drop_emulsion(
        _problem(labels, shape),
        ColorGradientLBMRuntimeParameters(VISCOSITY, tension),
        radius=radius,
        gap=gap,
        approach_speed=approach_speed,
        steps=steps,
        chunk=chunk,
    )
    repelled_successful, repelled_counts, repelled_work = _two_drop_emulsion(
        _problem(
            labels,
            shape,
            NearContactRepulsionPlan((("oil", "oil"),), interaction_range=4),
        ),
        ColorGradientLBMRuntimeParameters(
            VISCOSITY, tension, near_contact_strength=np.asarray([strength])
        ),
        radius=radius,
        gap=gap,
        approach_speed=approach_speed,
        steps=steps,
        chunk=chunk,
    )
    control_merged = min(control_counts) == 1
    repelled_separate = all(count == 2 for count in repelled_counts)
    return {
        "discretization": {
            "grid": list(shape),
            "radius": radius,
            "gap": gap,
            "approach_speed": approach_speed,
            "declared_interval_steps": steps,
            "sample_every": chunk,
            "interaction_range": 4,
            "strength": strength,
        },
        "control": {"successful": control_successful, "components": control_counts},
        "repelled": {
            "successful": repelled_successful,
            "components": repelled_counts,
            "near_contact_work": repelled_work,
        },
        "criterion": "control merges (one component) and the repelled pair keeps two "
        "components at every sample of the declared interval",
        "passed": control_successful
        and repelled_successful
        and control_merged
        and repelled_separate,
    }


SCENARIOS: dict[str, Callable[[], dict[str, object]]] = {
    "binary-regression": binary_regression,
    "ternary-neumann": ternary_neumann,
    "momentum": momentum,
    "emulsion-no-merger": emulsion_no_merger,
}


def run_qualification(selected: tuple[str, ...]) -> dict[str, object]:
    records = {}
    for name in selected:
        begin = time.perf_counter()
        record = SCENARIOS[name]()
        record["wall_seconds"] = time.perf_counter() - begin
        records[name] = record
    return {
        "kind": "color-gradient-lbm-qualification",
        "runtime": _runtime_identity(),
        "profiles": [
            profile.to_record() for profile in color_gradient_candidate_profiles()
        ],
        "scenarios": records,
        "successful": all(bool(record["passed"]) for record in records.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", action="append", choices=tuple(SCENARIOS))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    selected = tuple(arguments.scenario) if arguments.scenario else tuple(SCENARIOS)
    report = run_qualification(selected)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=float)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
