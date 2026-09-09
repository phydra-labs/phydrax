#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded A0 SPMe scientific observations through the actual battery facade.

Thresholds here are registry ceilings, not authority approval. Campaign criteria,
starts, execution signatures, reference rights and release decisions remain owned
by the qualification runner and release builders.
"""

from __future__ import annotations

from pathlib import Path

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.battery._experiment import (
    BatteryDiffraxSolvePlan,
    BatteryExperimentPlan,
    BatteryOutputPlan,
)
from phydrax.applications.battery._protocol import (
    BatteryProtocolPlan,
    BatteryProtocolValues,
    CurrentStepPlan,
    RestStepPlan,
)
from phydrax.applications.battery._qualification import (
    MARQUIS_2019_SPME_CANDIDATE,
    MARQUIS_2019_SPME_SUPPORT,
)
from phydrax.applications.battery._release_contracts import (
    CampaignCase,
    SPME_SCIENTIFIC_METRICS,
)
from phydrax.applications.battery._results import BatteryRunStatus
from phydrax.applications.battery._spme_marquis2019 import (
    Marquis2019SpmeAdapter,
    Marquis2019SpmeInitialCondition,
    Marquis2019SpmePlan,
)
from phydrax.qualification import ReferenceArtifactManifest
from tools._battery_spme_references import (
    FARADAY,
    native_parameters,
    REFERENCE_ENGINES,
    SyntheticSpmeData,
)
from tools.battery_campaign_registry import CampaignEntry, metric_key, PreparedCampaign


KIND = "spme-marquis2019-scientific"
DATA = SyntheticSpmeData()
TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)
BOUNDARIES = (0.0, 0.5, 0.75, 1.0)
CURRENTS = (0.2, 0.0, -0.05)
OBSERVABLES = (
    "voltage_v",
    "negative_surface_concentration_mol_m3",
    "positive_surface_concentration_mol_m3",
    "total_lithium_mol",
    "eq49:error_estimate",
    "current_a",
)
BASE_MESH = (12, (4, 3, 4))


def _case(case, **inputs):
    return CampaignCase(
        f"spme:{case}", tuple(sorted(inputs.items())), SPME_SCIENTIFIC_METRICS[case]
    )


CASES = (
    _case("equilibrium", current_a=0.0, times_s=TIMES),
    _case("current-rest", boundaries_s=BOUNDARIES, currents_a=CURRENTS, times_s=TIMES),
    _case("conservation", boundaries_s=BOUNDARIES, currents_a=CURRENTS),
    *(
        _case(f"{axis}-refinement", levels=levels, held_fixed=held)
        for axis, levels, held in (
            ("x", ((4, 3, 4), (8, 6, 8), (16, 12, 16)), f"r={BASE_MESH[0]};dt=1/128s"),
            (
                "r",
                tuple(BASE_MESH[0] * factor for factor in (1, 2, 4)),
                "x=(4,3,4);dt=1/128s",
            ),
            ("time", (128, 256, 512), f"r={BASE_MESH[0]};x=(4,3,4)"),
        )
    ),
    _case("eq49-table6", current_bound_a=DATA.maximum_current, parameter_id=DATA.data_id),
    _case(
        "fixed-path-jvp-vjp",
        currents_a=(0.2, -0.05),
        tangent=(0.4, -0.2),
        cotangent=(0.7, -0.3, 0.2),
        steps=128,
    ),
    _case(
        "references",
        engines=REFERENCE_ENGINES,
        parameter_id=DATA.data_id,
        radial_levels=(32, 64, 128),
        x_levels=((24, 12, 24), (48, 24, 48), (96, 48, 96)),
        times_s=TIMES,
    ),
)


def _prepare_experiment(
    *, shells=BASE_MESH[0], cells=BASE_MESH[1], steps=128, equilibrium=False
):
    adapter = Marquis2019SpmeAdapter(
        Marquis2019SpmePlan(
            shells,
            negative_electrolyte_cell_count=cells[0],
            separator_electrolyte_cell_count=cells[1],
            positive_electrolyte_cell_count=cells[2],
        )
    )
    protocol = BatteryProtocolPlan(
        (RestStepPlan(1.0),)
        if equilibrium
        else (CurrentStepPlan(0.5), RestStepPlan(0.25), CurrentStepPlan(0.25))
    )
    plan = BatteryExperimentPlan(
        adapter,
        protocol,
        BatteryOutputPlan(OBSERVABLES),
        BatteryDiffraxSolvePlan(
            stepsize_controller=dfx.StepTo(ts=jnp.linspace(0.0, 1.0, steps + 1)),
            adjoint=dfx.DirectAdjoint(),
            maximum_steps=steps + 8,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
        ),
        jnp.asarray(TIMES),
        MARQUIS_2019_SPME_CANDIDATE,
        MARQUIS_2019_SPME_SUPPORT,
    ).prepare()
    return plan, protocol


def _normalized_outputs(result):
    values = result.outputs.values
    return jnp.stack(
        (values[:, 0], values[:, 1] / DATA.cmax[0], values[:, 2] / DATA.cmax[1]), axis=-1
    )


def _difference(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def prepare(sample_times_s):
    if tuple(sample_times_s) != TIMES:
        raise ValueError(
            f"SPMe campaign requires the precommitted sample matrix {TIMES}."
        )
    parameters = native_parameters(DATA)
    initial = Marquis2019SpmeInitialCondition(*DATA.initial_stoichiometries)
    configurations = {
        "base": _prepare_experiment(),
        "equilibrium": _prepare_experiment(equilibrium=True),
        "x-medium": _prepare_experiment(cells=(8, 6, 8)),
        "x-fine": _prepare_experiment(cells=(16, 12, 16)),
        "r-medium": _prepare_experiment(shells=2 * BASE_MESH[0]),
        "r-fine": _prepare_experiment(shells=4 * BASE_MESH[0]),
        "time-medium": _prepare_experiment(steps=256),
        "time-fine": _prepare_experiment(steps=512),
    }

    def solve(name, currents=jnp.asarray((0.2, -0.05))):
        prepared, protocol = configurations[name]
        return prepared.run(
            parameters,
            initial,
            BatteryProtocolValues(
                protocol, jnp.asarray(()) if name == "equilibrium" else currents
            ),
        )

    def operation():
        return solve("base")

    def execute():
        results = {name: solve(name) for name in configurations}

        def objective(currents):
            result = solve("base", currents)
            return _normalized_outputs(result)[-1]

        point, tangent = jnp.asarray((0.2, -0.05)), jnp.asarray((0.4, -0.2))
        cotangent = jnp.asarray((0.7, -0.3, 0.2))
        _, jvp = jax.jvp(objective, (point,), (tangent,))
        _, transpose = jax.vjp(objective, point)
        vjp = transpose(cotangent)[0]
        h = 1e-4
        difference = (objective(point + h * tangent) - objective(point - h * tangent)) / (
            2 * h
        )

        def final_negative(current):
            return jnp.sum(
                solve(
                    "base", jnp.stack((current, point[1]))
                ).native_solution.states.negative_amount_mol[-1]
            )

        results["derivatives"] = {
            "jvp": jvp,
            "vjp": vjp,
            "finite_difference": difference,
            "duality_error": jnp.abs(jnp.vdot(cotangent, jvp) - jnp.vdot(tangent, vjp))
            / jnp.maximum(1.0, jnp.abs(jnp.vdot(cotangent, jvp))),
            "inventory_derivative": jax.grad(final_negative)(point[0]),
        }
        return results

    return PreparedCampaign(
        KIND,
        canonical_fingerprint(
            {"meshes": {k: v[0].preparation_id for k, v in configurations.items()}}
        ),
        parameters.parameter_id,
        MARQUIS_2019_SPME_CANDIDATE.profile_id,
        execute,
        operation,
    )


def _scalar_metric(metrics, case, name, value):
    value = float(value)
    metrics[metric_key(f"spme:{case}", name)] = {
        "value": value if np.isfinite(value) else None,
        "unavailable_reason": None
        if np.isfinite(value)
        else "nonfinite-scientific-observation",
    }


def _reference_matrix(level):
    return np.stack(
        (
            np.asarray(level["voltage_v"]),
            np.asarray(level["negative_surface_concentration_mol_m3"]) / DATA.cmax[0],
            np.asarray(level["positive_surface_concentration_mol_m3"]) / DATA.cmax[1],
        ),
        axis=-1,
    )


def _load_references(spec, directory):
    references = {}
    for binding in spec.payloads:
        if not isinstance(binding.manifest, ReferenceArtifactManifest):
            continue
        binding.verify_local(directory)
        from tools.battery_qualification import read_json_object

        payload = read_json_object(binding.local_path(directory))
        if (
            set(payload)
            != {"kind", "engine_id", "runtime_family", "mapping", "levels", "runtime"}
            or payload["kind"] != "spme-reference-observation"
        ):
            raise ValueError(
                "Reference payload is not the exact SPMe observation schema."
            )
        engine = payload["engine_id"]
        if engine not in REFERENCE_ENGINES or engine in references:
            raise ValueError(
                "References must name both distinct immutable engines exactly once."
            )
        if canonical_fingerprint(payload["mapping"]) != canonical_fingerprint(
            DATA.mapping_record()
        ):
            raise ValueError(
                "Reference parameters/geometry/initialization/hold/output audit differs."
            )
        if DATA.data_id not in binding.manifest.lineage_ids:
            raise ValueError(
                "Reference manifest must cite the self-authored parameter data identity."
            )
        runtime = dict(payload["runtime"])
        runtime_id = runtime.pop("runtime_id")
        if (
            canonical_fingerprint(runtime) != runtime_id
            or runtime_id not in binding.manifest.lineage_ids
        ):
            raise ValueError(
                "Reference manifest must bind the observed installed-code/runtime identity."
            )
        if (
            engine == REFERENCE_ENGINES[1]
            and runtime["packages"]["pybamm"]["version"] != "25.4.2"
        ):
            raise ValueError(
                "PyBaMM reference must use the precommitted installed package version."
            )
        if len(payload["levels"]) != 3:
            raise ValueError(
                "Reference must carry actual three-level self-convergence observations."
            )
        for level, radial, cells in zip(
            payload["levels"],
            (32, 64, 128),
            ((24, 12, 24), (48, 24, 48), (96, 48, 96)),
            strict=True,
        ):
            expected_keys = {
                "times_s",
                "current_a",
                "voltage_v",
                "negative_surface_concentration_mol_m3",
                "positive_surface_concentration_mol_m3",
                "radial_cells",
                "region_cells",
            }
            if engine == REFERENCE_ENGINES[1]:
                expected_keys.add("engine_terminal_voltage_v")
            if (
                set(level) != expected_keys
                or level["radial_cells"] != radial
                or tuple(level["region_cells"]) != cells
                or tuple(level["times_s"]) != TIMES
            ):
                raise ValueError(
                    "Reference mesh/time/output coverage differs from its precommitment."
                )
            if tuple(level["current_a"]) != (0.2, 0.2, 0.2, 0.0, -0.05):
                raise ValueError(
                    "Reference transition outputs must use the prescribed left hold."
                )
            matrix = _reference_matrix(level)
            if matrix.shape != (len(TIMES), 3) or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    "Reference numerical observations must be finite and complete."
                )
        references[engine] = (payload, binding.manifest.manifest_id)
    if (
        set(references) != set(REFERENCE_ENGINES)
        or len({v[0]["runtime_family"] for v in references.values()}) != 2
    ):
        raise ValueError("Exactly two distinct reference runtime families are required.")
    return references


def raw_output(spec, executed, campaign_directory: Path):
    metrics = {}
    base, equilibrium = executed["base"], executed["equilibrium"]
    add = lambda case, name, value: _scalar_metric(metrics, case, name, value)
    add(
        "equilibrium",
        "voltage-drift",
        np.max(
            np.abs(
                np.asarray(equilibrium.outputs.values[:, 0]) - (DATA.ocp[1] - DATA.ocp[0])
            )
        ),
    )
    add(
        "equilibrium",
        "inventory-drift",
        max(
            float(np.max(np.abs(np.asarray(v) - np.asarray(v)[0])))
            for v in jax.tree.leaves(equilibrium.native_solution.states)
        ),
    )
    add(
        "current-rest",
        "transition-current-error",
        _difference(base.outputs.values[:, -1], (0.2, 0.2, 0.2, 0.0, -0.05)),
    )
    ledger = base.ledger
    add(
        "current-rest",
        "charge-error",
        max(
            abs(float(ledger.negative_current_integral_residual_c)),
            abs(float(ledger.positive_current_integral_residual_c)),
        ),
    )
    provenance = []
    failures = 0
    for name, result in executed.items():
        if name == "derivatives":
            continue
        failures += (
            int(int(result.application_status) != int(BatteryRunStatus.SUCCESS))
            + int(not bool(result.ledger.successful))
            + int(not np.all(np.asarray(result.outputs.valid)))
        )
        failures += int(
            any(
                v is None
                for v in (
                    result.run_id,
                    result.protocol_values_digest,
                    result.initial_state_digest,
                    result.parameters_digest,
                )
            )
        )
        failures += int(
            result.numerical_admission_id is not None
            or result.predictive_admission_id is not None
        )
        provenance.append(
            {
                "case": name,
                "run_id": result.run_id,
                "preparation_id": result.preparation_id,
                "profile_id": result.profile_id,
                "support_tuple_id": result.support_tuple_id,
                "application_status": int(result.application_status),
                "ledger_successful": bool(result.ledger.successful),
            }
        )
    add("current-rest", "execution-contract-failures", failures)
    states = base.native_solution.states
    ns = np.asarray(states.negative_amount_mol).sum(axis=-1) + np.asarray(
        states.positive_amount_mol
    ).sum(axis=-1)
    ne = np.asarray(states.electrolyte_amount_mol).sum(axis=-1)
    for name, value in (
        ("solid-lithium-error", ns),
        ("electrolyte-lithium-error", ne),
        ("total-lithium-error", ns + ne),
    ):
        add("conservation", name, np.max(np.abs(value - value[0])))
    for name, value in (
        ("collector-flux-error", ledger.maximum_collector_flux_residual_mol_m2_s),
        (
            "interface-concentration-error",
            ledger.maximum_interface_concentration_jump_mol_m3,
        ),
        ("equation-mapping-error", ledger.maximum_equation_mapping_residual_mol_s),
        ("current-split-error", ledger.maximum_current_split_residual_a_m2),
    ):
        add("conservation", name, value)
    normalized = _normalized_outputs(base)
    for axis in ("x", "r", "time"):
        coarse = _difference(normalized, _normalized_outputs(executed[f"{axis}-medium"]))
        fine = _difference(
            _normalized_outputs(executed[f"{axis}-medium"]),
            _normalized_outputs(executed[f"{axis}-fine"]),
        )
        add(f"{axis}-refinement", "fine-difference", fine)
        add(f"{axis}-refinement", "contraction-excess", max(0.0, fine - 0.9 * coarse))
    expected = (
        np.abs(np.asarray(base.outputs.values[:, -1]))
        * sum(DATA.lengths)
        / (DATA.area * DATA.electrolyte_diffusivity * FARADAY * DATA.cmax[0])
    ) ** 2
    add(
        "eq49-table6",
        "eq49-formula-error",
        _difference(base.outputs.values[:, 4], expected),
    )
    add(
        "eq49-table6",
        "applicability-failures",
        int(not bool(ledger.eq49_applicable))
        + int(not bool(ledger.asymptotic_conditions_satisfied)),
    )
    derivatives = executed["derivatives"]
    add("fixed-path-jvp-vjp", "duality-error", derivatives["duality_error"])
    add(
        "fixed-path-jvp-vjp",
        "finite-difference-error",
        _difference(derivatives["jvp"], derivatives["finite_difference"]),
    )
    add(
        "fixed-path-jvp-vjp",
        "inventory-derivative-error",
        abs(float(derivatives["inventory_derivative"]) - 0.5 / FARADAY),
    )
    reference_records = []
    try:
        references = _load_references(spec, campaign_directory)
        fine_matrices = []
        for label, engine in zip(("paper", "pybamm"), REFERENCE_ENGINES, strict=True):
            payload, manifest_id = references[engine]
            coarse, medium, fine = [
                _reference_matrix(level) for level in payload["levels"]
            ]
            coarse_error, fine_error = (
                _difference(coarse, medium),
                _difference(medium, fine),
            )
            add(
                "references",
                f"{label}-contraction-excess",
                max(0.0, fine_error - 0.9 * coarse_error),
            )
            add("references", f"{label}-self-convergence", fine_error)
            add("references", f"{label}-discrepancy", _difference(normalized, fine))
            fine_matrices.append(fine)
            reference_records.append(
                {"engine_id": engine, "manifest_id": manifest_id, "observations": payload}
            )
        add("references", "cross-reference-agreement", _difference(*fine_matrices))
    except (OSError, KeyError, TypeError, ValueError) as error:
        for metric in CASES[-1].metrics:
            metrics[metric_key(CASES[-1].case_id, metric.name)] = {
                "value": None,
                "unavailable_reason": f"reference-inconclusive:{type(error).__name__}:{error}",
            }
    return {
        "metrics": metrics,
        "provenance": provenance,
        "native_observations": {
            name: np.asarray(result.outputs.values).tolist()
            for name, result in executed.items()
            if name != "derivatives"
        },
        "derivative_observations": {
            name: np.asarray(value).tolist() for name, value in derivatives.items()
        },
        "references": reference_records,
    }


def campaign_entry():
    return CampaignEntry(
        KIND,
        MARQUIS_2019_SPME_CANDIDATE,
        MARQUIS_2019_SPME_SUPPORT,
        CASES,
        2,
        ("commercial_use", "redistribution", "export"),
        prepare,
        raw_output,
        raw_fields=(
            "metrics",
            "provenance",
            "native_observations",
            "derivative_observations",
            "references",
        ),
    )
