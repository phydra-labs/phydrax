#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ....atomistic import (
    AtomisticSystemPlan,
    AtomisticUnitSystem,
    FreeSpaceRPYMobilityPlan,
    materialize_mobility,
)
from ....atomistic.driven_flow import (
    analyze_laos,
    EvolvingFlowCellPlan,
    LAOSAnalysisPlan,
    PlanarKraynikReineltPlan,
)
from ....discretization import PeriodicCell
from ..entanglement import EntanglementEstimatorPlan, estimate_entanglement
from ..reptation import (
    ChainLengthScalingPlan,
    doi_edwards_linear_rheology,
    DoiEdwardsTubePlan,
    fit_chain_length_scaling,
    glamm_step,
    GLAMMPlan,
)
from ._support import (
    decide_polymer_production_regime,
    PolymerProductionRegime,
)


class PolymerProductionSmokeResult(StrictModule):
    gate_passed: Array
    metrics: Array
    successful: Array
    gate_names: tuple[str, ...] = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def run_polymer_production_smoke() -> PolymerProductionSmokeResult:
    entanglement = estimate_entanglement(
        EntanglementEstimatorPlan(
            0.85,
            1.0,
            block_count=4,
            minimum_frames=8,
            maximum_relative_standard_error=1.0e-10,
        ),
        [20, 20],
        jnp.full((8, 2), 40.0),
        jnp.full((8, 2), 20.0),
    )
    tube = doi_edwards_linear_rheology(
        DoiEdwardsTubePlan(100.0, 1.0, odd_mode_count=16),
        [0.0, 1.0, 10.0],
        [0.01, 0.1, 1.0],
    )
    scaling = fit_chain_length_scaling(
        ChainLengthScalingPlan(exponent_interval=(2.9, 3.1)),
        [10.0, 20.0, 40.0],
        [1.0e3, 8.0e3, 64.0e3],
    )
    glamm = GLAMMPlan(3, 1.0e-3, 1.0, 10.0, 1.0, contour_diffusivity=0.0)
    glamm_result = glamm_step(glamm, glamm.initialize(), jnp.zeros((3, 3)))

    system = AtomisticSystemPlan(
        [0, 1],
        [0, 0],
        [1.0, 1.0],
        AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0],
        element_mask=[False, False],
    ).prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    rpy = FreeSpaceRPYMobilityPlan(1.0, 1.0, maximum_particles=2).prepare(system, [0, 1])
    mobility = materialize_mobility(rpy, positions, maximum_dofs=6)
    mobility_symmetry = jnp.max(jnp.abs(mobility - mobility.T))
    mobility_minimum = jnp.min(jnp.linalg.eigvalsh(mobility))

    cell = PeriodicCell(jnp.eye(3) * 5.0)
    flow = EvolvingFlowCellPlan(cell)
    flow_result = flow.step(flow.initialize(), jnp.zeros((3, 3)), 1.0e-3)
    kr = PlanarKraynikReineltPlan(0.2, 25.0, 5.0)
    period = 2.0 * jnp.pi
    time = jnp.linspace(0.0, 3.0 * period, 601)
    strain = jnp.sin(time)
    stress = 2.0 * jnp.sin(time) + 0.5 * jnp.cos(time)
    laos = analyze_laos(
        LAOSAnalysisPlan(
            1.0,
            1.0,
            harmonic_count=3,
            discard_cycles=1,
            minimum_analysis_cycles=2,
            closure_tolerance=1.0e-8,
        ),
        time,
        strain,
        stress,
    )
    regime = decide_polymer_production_regime(
        PolymerProductionRegime(
            "hard-sphere-colloid",
            "none",
            "none",
            "free-space-rpy",
            "equilibrium",
        )
    )

    names = (
        "entanglement-estimator",
        "doi-edwards-spectrum",
        "reptation-scaling",
        "glamm-zero-flow",
        "free-space-rpy-spd",
        "evolving-cell-zero-flow",
        "planar-kr-certification",
        "laos-harmonic-recovery",
        "support-matrix-admission",
    )
    gates = jnp.asarray(
        [
            entanglement.successful,
            tube.successful,
            scaling.successful,
            glamm_result.successful,
            (mobility_symmetry <= 1.0e-12) & (mobility_minimum > 0.0),
            flow_result.successful,
            kr.generalized.certification_residual <= 1.0e-10,
            laos.successful,
            regime.supported,
        ]
    )
    metrics = jnp.asarray(
        [
            entanglement.coil_entanglement_length,
            tube.tube_survival[-1],
            scaling.exponent,
            jnp.max(jnp.abs(glamm_result.cauchy_stress)),
            mobility_minimum,
            flow_result.condition_number,
            kr.generalized.certification_residual,
            laos.total_harmonic_distortion,
            float(regime.supported),
        ]
    )
    successful = jnp.all(gates) & jnp.all(jnp.isfinite(metrics))
    evidence_id = canonical_fingerprint(
        {
            "kind": "polymer-production-smoke-evidence",
            "gates": names,
            "passed": np.asarray(gates),
            "metrics": array_tree_fingerprint(np.asarray(metrics)),
        }
    )
    return PolymerProductionSmokeResult(gates, metrics, successful, names, evidence_id)


__all__ = ["PolymerProductionSmokeResult", "run_polymer_production_smoke"]
