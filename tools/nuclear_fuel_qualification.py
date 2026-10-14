#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Numerical fuel campaign. Synthetic rod inputs are not experimental validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import assert_never, get_args, Literal, TYPE_CHECKING, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray

from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_environment,
    source_build_fingerprint,
)
from phydrax import config as pxconfig, linalg as la, nonlinear as nl
from phydrax._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
)
from phydrax.applications.contact import axisymmetric_interface_geometry
from phydrax.applications.nuclear_fuel import (
    AthermalRelease,
    BoreEmissivity,
    CladdingSpecification,
    CoolantBoundaryKind,
    CoulombFriction,
    CoupledChannelCoolant,
    EnergyDeposition,
    FissionGasModel,
    FuelAcceptancePolicy,
    FuelApplicabilityPolicy,
    FuelGeometryReduction,
    FuelModelBundle,
    FuelRodSpecification,
    GapConductanceLaw,
    GasAccommodationLaw,
    HeliumGasModel,
    HeliumSource,
    HeliumTransportTable,
    IrradiationHistory,
    PelletSpecification,
    PlenumTemperatureClosure,
    PrescribedWallTemperature,
    RadialPowerProfileModel,
    RelocationHistory,
    RodSpecification,
    SingleChannelCoolant,
)
from phydrax.applications.thermofluids import LiquidChannelHeatBalance
from phydrax.materials import (
    iapws97_liquid_enthalpy,
    iapws97_liquid_state,
    PropertyValidityRange,
)
from phydrax.nuclear import (
    AVOGADRO_PER_MOL,
    FissionGasYieldTable,
    FISSIONING_NUCLIDES,
    HeliumProductionPlan,
    InventoryTransition,
    NuclearDataProvenance,
    NuclideInventory,
    NuclideKey,
)
from phydrax.qualification import (
    CapabilityProfile,
    HMACSHA256ReleaseSigner,
    HMACSHA256TrustPolicy,
    ReferenceArtifactManifest,
    ReleaseGateEvidence,
    ReleaseIndex,
)
from phydrax.qualification._nuclear_fuel_release_matrix import (
    FUEL_VERIFICATION_CASES,
    nuclear_fuel_closure_matrices,
)
from phydrax.typing import parse


if TYPE_CHECKING:
    from phydrax.applications.nuclear_fuel import (
        FuelNumericalExecution,
        FuelRodResult,
        FuelRodSimulation,
        FuelRodState,
        FuelStepEvidence,
        PreparedFuelRod,
    )


type FuelCampaignDesign = Literal["smoke", "default-full"]
type CaseOutcome = Literal["passed", "failed"]


class FuelHistoryBalanceRecord(TypedDict):
    accepted_times_s: list[float]
    attempted_steps: int
    maximum_energy_relative_error: float
    maximum_gas_relative_error: float
    maximum_source_relative_error: float
    state_fingerprints: list[str]
    native_attempts: list[dict[str, object]]
    native_terminal_status: str
    source_status: dict[str, str]


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


def numerical_helium_production() -> HeliumSource:
    """Declare synthetic native decay/yield data, never evaluated nuclear data."""
    parent, daughter, helium = NuclideKey(92, 238), NuclideKey(90, 234), NuclideKey(2, 4)
    reference_volume, parent_amount = 1.0, 1.0
    decay_rate = 1e10 * reference_volume / (parent_amount * AVOGADRO_PER_MOL)
    ternary_yields = {name: 1e-4 for name in FISSIONING_NUCLIDES}
    payload = canonical_json(
        {
            "kind": "synthetic-numerical-helium-production",
            "parent": parent.nuclide_id,
            "daughter": daughter.nuclide_id,
            "helium": helium.nuclide_id,
            "ternary_yields": ternary_yields,
            "parent_amount_mol": parent_amount,
            "reference_volume_m3": reference_volume,
            "declared_decay_rate_s": decay_rate,
            "target_decay_rate_atoms_per_m3_s": 1e10,
            "inventory_policy": "fixed-declared-actinide-approximation-no-depletion",
        }
    ).encode("utf-8")
    reference = ReferenceArtifactManifest(
        "synthetic-numerical-helium-production",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic-owned-numerical-input",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="synthetic-fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic-numerical-verification:not-evaluated-decay-data",),
    )
    reference.verify_bytes(payload)
    provenance = NuclearDataProvenance(
        reference,
        "synthetic://nuclear-fuel/numerical-helium-production",
        "synthetic-numerical-verification",
        "declared-control",
        "not-evaluated-physical-decay-data",
    )
    transition = InventoryTransition(
        "synthetic-U238-to-Th234-plus-He4-control",
        parent,
        ((daughter, 1.0), (helium, 1.0)),
        decay_rate,
        np.zeros((1,), dtype=np.float64),
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        provenance,
    )
    inventory = NuclideInventory(
        np.asarray((parent_amount,), dtype=np.float64),
        0.0,
        (parent.nuclide_id,),
        canonical_fingerprint(
            {
                "kind": "synthetic-fixed-alpha-network",
                "transition": transition.transition_id,
                "nuclides": (parent.nuclide_id,),
            }
        ),
    )
    plan = HeliumProductionPlan(
        ternary_yields,
        inventory,
        (transition,),
        provenance,
        helium=helium,
    )
    return HeliumSource(
        production=plan, inventory=inventory, reference_volume=reference_volume
    )


def numerical_rod_inputs(
    *,
    mode_count: int = 8,
    helium_radial_cell_count: int = 40,
    axial_layers: int = 3,
    scattering_cross_section_barn: float = 5.0,
    scattering_source_id: str = "synthetic-numerical-verification:not-spectrum-data",
    base_duration: float = 3480.0,
    base_linear_power: float = 3000.0,
    transient_duration: float = 60.0,
    helium_transport: HeliumTransportTable | None = None,
    fuel_deposition_fraction: float = 0.95,
    applicability: FuelApplicabilityPolicy = "equation-reproduction",
    friction: CoulombFriction | None = None,
) -> tuple[
    FuelRodSpecification, IrradiationHistory, PrescribedWallTemperature, FuelModelBundle
]:
    """Representative, explicitly synthetic startup/base/ramp/shutdown history.

    Geometry, irradiation, escape surface, energy partition and spectrum are
    declared numerical design inputs, not an IFPE/FUMEX rod. Source correlation
    coefficients retain their owners' provenance. The scattering default is
    deliberately synthetic: physical use requires processed spectrum data.
    ``friction=None`` is frictionless axial contact.
    """
    if axial_layers < 1:
        raise ValueError("axial_layers must be positive.")
    for name, value in (
        ("base_duration", base_duration),
        ("base_linear_power", base_linear_power),
        ("transient_duration", transient_duration),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    radius, height, length = 4.095e-3, 10.0e-3, 0.3
    specification = FuelRodSpecification(
        PelletSpecification(
            radius=radius,
            height=height,
            density_fraction=0.95,
            uranium_235_weight_fraction=0.035,
            grain_radius=5.0e-6,
            resintering_density_change=100.0,
            roughness=1.0e-6,
            open_porosity_fraction=0.005,
        ),
        CladdingSpecification(
            inner_radius=4.18e-3, outer_radius=4.75e-3, roughness=0.5e-6, cold_work=0.5
        ),
        RodSpecification(
            stack_length=length,
            plenum_length=0.03,
            fill_pressure=0.103e6,
            fill_temperature=300.0,
        ),
    )
    base_end = 2 * transient_duration + base_duration
    times = np.asarray(
        (
            0.0,
            transient_duration,
            2 * transient_duration,
            base_end,
            *(base_end + index * transient_duration for index in range(1, 6)),
        ),
        dtype=np.float64,
    )
    # Startup coolant heating/pressurization at zero power, ascension, base hold,
    # power ramp, load following, shutdown, cooldown, then restart.
    power = base_linear_power * np.asarray(
        (0.0, 0.0, 1.0, 1.0, 5 / 3, 2 / 3, 0.0, 0.0, 1.0), dtype=np.float64
    )
    wall = np.asarray(
        (300.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 300.0, 600.0), dtype=np.float64
    )
    pressure = np.asarray(
        (0.103e6, 15.5e6, 15.5e6, 15.5e6, 15.5e6, 15.5e6, 15.5e6, 0.103e6, 15.5e6),
        dtype=np.float64,
    )
    linear_power = np.broadcast_to(power[:, None], (times.size, axial_layers)).copy()
    history = IrradiationHistory(
        times=times,
        zone_edges=np.linspace(0.0, length, axial_layers + 1),
        linear_power=linear_power,
        fast_flux=linear_power * 5.0e13,
        coolant_inlet_temperature=wall,
        coolant_pressure=pressure,
        coolant_mass_flux=np.full_like(times, 3000.0),
    )
    coolant = PrescribedWallTemperature(
        times, np.broadcast_to(wall[:, None], linear_power.shape).copy()
    )
    surface = 2.0 / radius + 2.0 / height
    transport = (
        HeliumTransportTable(
            temperature=(293.0, 2500.0),
            diffusivity=(1e-20, 1e-20),
            henry_constant=(1e12, 1e12),
            validity=PropertyValidityRange(
                {"temperature": (293.0, 2500.0, "K")},
                source_id="synthetic-numerical-verification:helium-transport",
            ),
            source_id="synthetic-numerical-verification:helium-transport",
        )
        if helium_transport is None
        else helium_transport
    )
    bundle = FuelModelBundle(
        applicability=applicability,
        radial_power=RadialPowerProfileModel(
            energy_per_fission=FissionGasYieldTable.endf_b_viii_0().energy_per_fission,
            scattering_cross_section_barn=scattering_cross_section_barn,
            scattering_source_id=scattering_source_id,
            form_depth_unit_m=1.0e-3,
        ),
        fission_gas=FissionGasModel(
            mode_count=mode_count,
            athermal=AthermalRelease(
                geometric_surface_to_volume=surface, total_surface_to_volume=surface
            ),
        ),
        helium=HeliumGasModel(
            radial_cell_count=helium_radial_cell_count, transport=transport
        ),
        helium_source=numerical_helium_production(),
        relocation_history=RelocationHistory(
            policy="fission-activated-permanent-power-recoverable",
            source_id="declared-numerical-translation:not-FAST-caller",
        ),
        gap=GapConductanceLaw(accommodation=GasAccommodationLaw.gap_comparison()),
        bore_emissivity=BoreEmissivity(
            value=0.3,
            temperature_range=(293.0, 2500.0),
            source_id="synthetic-numerical-verification:gray-bore",
        ),
        plenum_temperature=PlenumTemperatureClosure(
            offset=0.0,
            source_id="declared-numerical-plenum:coolant-outlet-zero-offset",
            prescribed_source_id="declared-numerical-plenum:terminal-wall-proxy",
        ),
        deposition=EnergyDeposition(
            fuel_fraction=fuel_deposition_fraction,
            cladding_fraction=0.025,
            source_id="synthetic-numerical-verification:declared-energy-partition",
        ),
        friction=friction,
    )
    return specification, history, coolant, bundle


def prepare_numerical_rod(
    *,
    fuel_cells: int = 8,
    cladding_cells: int = 4,
    axial_layers: int = 3,
    axial_cells_per_zone: int = 1,
    mode_count: int = 8,
    helium_radial_cell_count: int = 40,
    reduction: FuelGeometryReduction = "layered-radial",
    base_duration: float = 3480.0,
    base_linear_power: float = 3000.0,
    transient_duration: float = 60.0,
    fuel_deposition_fraction: float = 0.95,
) -> PreparedFuelRod:
    from phydrax.applications.nuclear_fuel import admit_fuel_rod, prepare_fuel_rod

    specification, history, coolant, bundle = numerical_rod_inputs(
        mode_count=mode_count,
        helium_radial_cell_count=helium_radial_cell_count,
        axial_layers=axial_layers,
        base_duration=base_duration,
        base_linear_power=base_linear_power,
        transient_duration=transient_duration,
        fuel_deposition_fraction=fuel_deposition_fraction,
    )
    admitted = admit_fuel_rod(
        specification,
        history,
        coolant,
        bundle=bundle,
        reduction=reduction,
        axial_interaction="frictionless",
    )
    return prepare_fuel_rod(
        admitted,
        fuel_cells=fuel_cells,
        cladding_cells=cladding_cells,
        axial_cells_per_zone=axial_cells_per_zone,
        quadrature_order=3,
    )


def _solve(
    residual: Callable[[Array, None], Array],
    initial: Array,
    case: str,
    /,
    *,
    method: nl.NewtonKrylov | None = None,
) -> tuple[Array, dict[str, object]]:
    problem = nl.NonlinearSystemProblem(residual, problem_id=f"fuel-verification:{case}")
    linear_policy = la.LinearSolvePolicy(
        la.GMRES(restart=32),
        tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-13, max_steps=256),
    )
    chosen = nl.NewtonKrylov(linear_policy=linear_policy) if method is None else method
    prepared = nl.prepare_nonlinear(
        problem,
        initial,
        method=chosen,
        termination=nl.NonlinearTermination(
            absolute_residual=1e-9,
            relative_residual=1e-10,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=40,
        ),
        args=None,
    )
    result = nl.solve_prepared_nonlinear(prepared)
    diagnostics = result.diagnostics
    condition = float(diagnostics.final_linear_condition_estimate)
    linear_residual = float(diagnostics.final_linear_residual_norm)
    native = {
        "status": int(result.status),
        "successful": bool(result.successful),
        "residual_norm": float(diagnostics.final_residual_norm),
        "iterations": int(diagnostics.iterations),
        "residual_evaluations": int(diagnostics.residual_evaluations),
        "counts_complete": diagnostics.counts_complete,
        "linear_status": int(diagnostics.final_linear_status),
        "linear_converged": bool(diagnostics.final_linear_converged),
        "linear_rank": int(diagnostics.final_linear_rank),
        "linear_rank_available": int(diagnostics.final_linear_rank) >= 0,
        "linear_condition": _finite_json_value(condition),
        "linear_condition_available": not math.isnan(condition),
        "linear_residual": _finite_json_value(linear_residual),
    }
    if not bool(result.successful):
        raise RuntimeError(
            f"{case}: native nonlinear solve failed: status={result.status}, "
            f"native_evidence={native}"
        )
    if not isinstance(result.state, Array):
        raise TypeError(
            "Isolated numerical form solves require one canonical nodal array."
        )
    return result.state, native


def cylinder_conduction(
    cells: int, /, *, nonlinear: bool = False, fault: bool = False
) -> dict[str, object]:
    """Solve the app's weighted energy form against an independent Kirchhoff oracle."""
    from phydrax.applications.nuclear_fuel._forms import thermal_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    body = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=cells,
        cladding_cells=2,
        quadrature_order=3,
    ).fuel
    radius, wall, source, k0 = spec.pellet.radius, 600.0, 1.0e8, 4.0
    alpha = 0.002 if nonlinear else 0.0
    eye = jnp.eye(2, dtype=jnp.float64)
    zeros = jnp.zeros((body.site_count,), dtype=jnp.float64)
    initial = jnp.full((body.node_count,), wall, dtype=jnp.float64)

    def residual(temperature: Array, _: None) -> Array:
        conductivity = k0 * (1.0 + alpha * (body.values(temperature) - wall))
        rows = thermal_volume_residual(
            body,
            temperature,
            zeros,
            jnp.ones_like(zeros),
            conductivity[:, None, None] * eye,
            jnp.full_like(zeros, 0.0 if fault else source),
            jnp.asarray(1.0),
        )
        return rows.at[-1].set(temperature[-1] - wall)

    temperature, native = _solve(residual, initial, "V2" if nonlinear else "V1")
    radii = np.asarray(body.nodes)[:, 0]
    kirchhoff = source * (radius**2 - radii**2) / (4.0 * k0)
    exact = wall + (2.0 * kirchhoff / (1.0 + np.sqrt(1.0 + 2.0 * alpha * kirchhoff)))
    error = np.asarray(temperature) - exact
    site_r = np.asarray(body.site_coordinates)[:, 0]
    site_potential = source * (radius**2 - site_r**2) / (4.0 * k0)
    site_exact = wall + 2 * site_potential / (1 + np.sqrt(1 + 2 * alpha * site_potential))
    site_error = np.asarray(body.values(temperature)) - site_exact
    return {
        "cells": cells,
        "native_solve": native,
        "linf_temperature_K": float(np.max(np.abs(error))),
        "l2_temperature_K": float(
            np.sqrt(
                np.sum(np.asarray(body.site_weight) * site_error**2)
                / np.sum(np.asarray(body.site_weight))
            )
        ),
        "assembled_residual_linf": float(
            np.max(np.abs(np.asarray(residual(temperature, None))))
        ),
    }


def lame_gps(cells: int, /, *, fault: bool = False) -> dict[str, object]:
    """Solve the canonical mechanical weak form and the closed-end GPS row."""
    from phydrax.applications.nuclear_fuel._forms import mechanical_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    geometry = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=2,
        cladding_cells=cells,
        quadrature_order=4,
    )
    body = geometry.cladding
    a, b = spec.cladding.inner_radius, spec.cladding.outer_radius
    length = spec.rod.stack_length
    modulus, poisson, inner_p, outer_p = 1.0e11, 0.3, 2.0e6, 0.5e6
    lame_lambda = modulus * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = modulus / (2 * (1 + poisson))
    radius = body.site_coordinates[:, 0]
    scale = 2 * math.pi * b * length * inner_p

    def stresses(unknown: Array) -> Array:
        displacement, axial = unknown[:-1], unknown[-1]
        strain = jnp.stack(
            (
                body.gradients(displacement)[:, 0],
                body.values(displacement) / radius,
                jnp.full_like(radius, axial),
            ),
            axis=-1,
        )
        return lame_lambda * jnp.sum(strain, axis=-1, keepdims=True) + 2 * shear * strain

    def residual(unknown: Array, _: None) -> Array:
        stress = stresses(unknown)
        rows = mechanical_volume_residual(body, jax.vmap(jnp.diag)(stress))[:, 0]
        rows -= geometry.cladding_interface.assemble(
            jnp.full_like(geometry.cladding_interface.weight, 0.0 if fault else inner_p)
        )
        rows += geometry.cladding_outer.assemble(
            jnp.full_like(geometry.cladding_outer.weight, 0.0 if fault else outer_p)
        )
        resultant = body.integrate(stress[:, 2]) - length * math.pi * (
            0.0 if fault else inner_p * a**2 - outer_p * b**2
        )
        return jnp.concatenate((rows, jnp.reshape(resultant / b, (1,)))) / scale

    solved, native = _solve(
        residual, jnp.zeros((body.node_count + 1,), dtype=jnp.float64), "V3"
    )
    A = (inner_p * a**2 - outer_p * b**2) / (b**2 - a**2)
    B = (inner_p - outer_p) * a**2 * b**2 / (b**2 - a**2)
    r = np.asarray(body.nodes)[:, 0]
    exact_u = ((1 - 2 * poisson) * A * r + (1 + poisson) * B / r) / modulus
    exact_axial = (1 - 2 * poisson) * A / modulus
    sr = np.asarray(radius)
    exact_stress = np.stack((A - B / sr**2, A + B / sr**2, np.full_like(sr, A)), axis=-1)
    return {
        "cells": cells,
        "native_solve": native,
        "displacement_linf_m": float(np.max(np.abs(np.asarray(solved[:-1]) - exact_u))),
        "stress_relative_linf": float(
            np.max(np.abs(np.asarray(stresses(solved)) - exact_stress)) / inner_p
        ),
        "axial_strain_error": abs(float(solved[-1]) - exact_axial),
        "closed_end_resultant_error_N": abs(
            float(body.integrate(stresses(solved)[:, 2])) / length
            - math.pi * (inner_p * a**2 - outer_p * b**2)
        ),
    }


def finite_strain_tube(cells: int, /, *, fault: bool = False) -> dict[str, object]:
    """Rivlin tube stresses projected through production Piola forms and FE routes."""
    from phydrax.applications.nuclear_fuel._forms import mechanical_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    geometry = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=2,
        cladding_cells=cells,
        quadrature_order=5,
    )
    body = geometry.cladding
    A, B = spec.cladding.inner_radius, spec.cladding.outer_radius
    shear, stretch = 3.0e10, 1.15
    c = A**2 * (stretch**2 - 1)
    a, b = stretch * A, math.sqrt(B**2 + c)
    pressure = shear * (math.log(B / A) - math.log(b / a) + c * (1 / a**2 - 1 / b**2) / 2)
    R = body.site_coordinates[:, 0]
    r = jnp.sqrt(R**2 + c)
    radial_stress = shear * (
        jnp.log(R / B) - jnp.log(r / b) - c * (1 / r**2 - 1 / b**2) / 2
    )
    hoop_stress = radial_stress + shear * (r**2 / R**2 - R**2 / r**2)
    nodal_displacement = jnp.sqrt(body.nodes[:, 0] ** 2 + c) - body.nodes[:, 0]
    radial_stretch = 1 + body.gradients(nodal_displacement)[:, 0]
    hoop_stretch = 1 + body.values(nodal_displacement) / R
    J = radial_stretch * hoop_stretch
    piola = jax.vmap(jnp.diag)(
        jnp.stack(
            (
                J * radial_stress / radial_stretch,
                J * hoop_stress / hoop_stretch,
                jnp.zeros_like(R),
            ),
            axis=-1,
        )
    )
    residual = mechanical_volume_residual(body, piola)[:, 0]
    residual -= geometry.cladding_interface.assemble(
        jnp.full_like(
            geometry.cladding_interface.weight, 0.0 if fault else pressure * a / A
        )
    )
    force_scale = pressure * 2 * math.pi * a * spec.rod.stack_length
    return {
        "cells": cells,
        "inner_stretch": stretch,
        "pressure_Pa": pressure,
        "relative_equilibrium_defect": float(
            np.max(np.abs(np.asarray(residual))) / force_scale
        ),
        "jacobian_linf_error": float(np.max(np.abs(np.asarray(J) - 1))),
    }


def free_expansion(cells: int, /, *, fault: bool = False) -> dict[str, object]:
    """Exact finite multiplicative eigen-expansion has zero Piola stress."""
    from phydrax.applications.nuclear_fuel._forms import (
        deformation,
        mechanical_volume_residual,
    )
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    body = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=cells,
        cladding_cells=2,
        quadrature_order=3,
    ).fuel
    stretch = 1.02
    displacement = (stretch - 1) * body.nodes[:, 0]
    F = deformation(
        body, displacement[:, None], jnp.asarray((stretch - 1,), dtype=jnp.float64)
    )
    elastic_diagonal = jnp.diagonal(F, axis1=-2, axis2=-1) / (1.0 if fault else stretch)
    J = jnp.prod(elastic_diagonal, axis=-1)
    inverse_diagonal = 1 / elastic_diagonal
    piola_diagonal = (
        3.0e10 * (elastic_diagonal - inverse_diagonal)
        + 5.0e10 * jnp.log(J)[:, None] * inverse_diagonal
    )
    piola = jax.vmap(jnp.diag)(piola_diagonal)
    assembled = mechanical_volume_residual(body, piola)
    return {
        "cells": cells,
        "stress_linf_Pa": float(np.max(np.abs(np.asarray(piola)))),
        "assembled_force_linf_N": float(np.max(np.abs(np.asarray(assembled)))),
    }


def shrink_fit(cells: int, /, *, fault: bool = False) -> dict[str, object]:
    """Resolve pressure and displacement together in production contact forms."""
    from phydrax.applications.nuclear_fuel._forms import mechanical_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import (
        fuel_rod_geometry,
        FuelBodyGeometry,
    )
    from phydrax.nonlinear import Bounds, fischer_burmeister_map

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    geometry = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=cells,
        cladding_cells=cells,
        quadrature_order=4,
    )
    fuel, clad = geometry.fuel, geometry.cladding
    a, b, c = spec.pellet.radius, spec.cladding.inner_radius, spec.cladding.outer_radius
    Ef, Ec, nu, dilation = 2.0e11, 1.0e11, 0.3, 0.03
    pressure_scale, displacement_scale = 1.0e8, a * dilation
    nf = fuel.node_count

    def body_rows(
        body: FuelBodyGeometry, u: Array, modulus: float, eigen: float
    ) -> Array:
        radius = body.site_coordinates[:, 0]
        strain = jnp.stack(
            (
                body.gradients(u)[:, 0] - eigen,
                body.values(u) / radius - eigen,
                jnp.zeros_like(radius),
            ),
            axis=-1,
        )
        lam = modulus * nu / ((1 + nu) * (1 - 2 * nu))
        shear = modulus / (2 * (1 + nu))
        stress = lam * jnp.sum(strain, axis=-1, keepdims=True) + 2 * shear * strain
        return mechanical_volume_residual(body, jax.vmap(jnp.diag)(stress))[:, 0]

    def residual(x: Array, _: None) -> Array:
        uf, uc, p = (
            x[:nf] * displacement_scale,
            x[nf:-1] * displacement_scale,
            x[-1] * pressure_scale,
        )
        rf = body_rows(fuel, uf, Ef, 0.0 if fault else dilation)
        rc = body_rows(clad, uc, Ec, 0.0)
        rf += geometry.fuel_interface.assemble(
            jnp.full_like(geometry.fuel_interface.weight, p)
        )
        rc -= geometry.cladding_interface.assemble(
            jnp.full_like(geometry.cladding_interface.weight, p)
        )
        rf = rf.at[0].set(
            uf[0] / displacement_scale * pressure_scale * a * spec.rod.stack_length
        )
        gap = (b + uc[0] - a - uf[-1]) / displacement_scale
        contact = fischer_burmeister_map(gap, x[-1], Bounds(0.0))
        force_scale = pressure_scale * 2 * math.pi * a * spec.rod.stack_length
        return jnp.concatenate(
            (rf / force_scale, rc / force_scale, jnp.reshape(contact, (1,)))
        )

    solved, native = _solve(
        residual,
        jnp.zeros((fuel.node_count + clad.node_count + 1,), dtype=jnp.float64),
        "V6",
    )
    compliance_f = a * (1 + nu) * (1 - 2 * nu) / Ef
    compliance_c = b * (1 + nu) * ((1 - 2 * nu) * b**2 + c**2) / (Ec * (c**2 - b**2))
    exact_pressure = (a * dilation - (b - a)) / (compliance_f + compliance_c)
    pressure = float(solved[-1]) * pressure_scale
    return {
        "cells": cells,
        "native_solve": native,
        "pressure_Pa": pressure,
        "exact_pressure_Pa": exact_pressure,
        "pressure_relative_error": abs(pressure - exact_pressure) / exact_pressure,
        "contact_residual": float(abs(residual(solved, None)[-1])),
    }


def booth_diffusion(mode_count: int, /, *, fault: bool = False) -> dict[str, object]:
    """Production site's frozen-D diffusion transfer versus infinite Booth series."""
    from phydrax.applications.nuclear_fuel._grain_boundary import AinscoughGrainGrowth
    from phydrax.applications.nuclear_fuel._intragranular import (
        SpeightTrapping,
        TurnbullDiffusivity,
    )

    D, radius, dt, production = 1.0e-16, 5.0e-6, 1.0e4, 1.0e12
    model = FissionGasModel(
        mode_count=mode_count,
        athermal=AthermalRelease(
            geometric_surface_to_volume=0.0, total_surface_to_volume=0.0
        ),
        diffusivity=TurnbullDiffusivity(
            intrinsic_prefactor=2 * D if fault else D,
            intrinsic_activation_energy=0.0,
            irradiation_prefactor=0.0,
            athermal_coefficient=0.0,
        ),
        trapping=SpeightTrapping(
            bubble_density_coefficient=0.0, bubble_density_offset=0.0
        ),
        grain_growth=AinscoughGrainGrowth(rate_prefactor=0.0),
    )
    state = model.initial_state((), grain_radius=radius)
    candidate_radius = model.grain_growth.advance_radius(
        state.grain_radius, 1500.0, dt, burnup_mwd_per_kghm=0.0
    )
    update = model.advance(
        state,
        temperature=1500.0,
        previous_temperature=1500.0,
        fission_rate_density=1.0e18,
        gas_production_rate=production,
        burnup_mwd_per_kghm=0.0,
        hydrostatic_stress=0.0,
        time_step=dt,
        grain_radius=candidate_radius,
    )
    if not bool(update.admissible):
        raise RuntimeError("V7: production gas candidate failed local admissibility.")
    modes = np.arange(1, 100001, dtype=np.float64)
    x = D * dt / radius**2
    exact = 1 - 6 / (math.pi**4 * x) * np.sum(
        -np.expm1(-(math.pi**2) * modes**2 * x) / modes**4
    )
    released_to_boundary = float(update.transfers.diffusion_to_boundary) / (
        production * dt
    )
    bound = float(update.diagnostics.modal_truncation_bound) / (production * dt)
    return {
        "modes": mode_count,
        "boundary_transfer_fraction": released_to_boundary,
        "booth_fraction": float(exact),
        "absolute_error": abs(released_to_boundary - exact),
        "modal_truncation_bound": bound,
        "gas_balance_relative_error": abs(
            float(update.retained_total + update.state.released - production * dt)
        )
        / (production * dt),
    }


def piecewise_temperature_pressure(*, fault: bool = False) -> dict[str, object]:
    """Solve the actual common-pressure algebraic residual at unequal gas temperatures."""
    from phydrax.applications.nuclear_fuel._free_volume import (
        FreeVolumeComponents,
        gas_pressure_residual,
    )

    volumes = np.asarray((1.5e-6, 3.0e-7, 2.0e-7, 5.0e-8), dtype=np.float64)
    temperatures = np.asarray((560.0, 650.0, 750.0, 900.0), dtype=np.float64)
    components = FreeVolumeComponents(
        jnp.asarray(volumes[0]),
        jnp.asarray(2 * temperatures[0] if fault else temperatures[0]),
        jnp.asarray(volumes[1:3]),
        jnp.asarray(temperatures[1:3]),
        jnp.asarray(volumes[3:]),
        jnp.asarray(temperatures[3:]),
    )
    moles = 1.0e-4

    def residual(pressure: Array, _: None) -> Array:
        return jnp.reshape(
            gas_pressure_residual(pressure[0] * 1e6, jnp.asarray(moles), components), (1,)
        )

    solved, native = _solve(residual, jnp.ones((1,), dtype=jnp.float64), "V8")
    pressure = float(solved[0]) * 1e6
    exact = moles * 8.31446261815324 / np.sum(volumes / temperatures)
    return {
        "native_solve": native,
        "pressure_Pa": pressure,
        "exact_pressure_Pa": float(exact),
        "relative_error": abs(pressure - exact) / exact,
    }


def paired_gap_exchange(cells: int, /, *, fault: bool = False) -> dict[str, object]:
    """Solve both cylinder energy fields with paired production trace exchange."""
    from phydrax.applications.nuclear_fuel._forms import thermal_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    g = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=cells,
        cladding_cells=cells,
        quadrature_order=4,
    )
    f, c, fi, ci = g.fuel, g.cladding, g.fuel_interface, g.cladding_interface
    nf, kf, kc, h, q, wall = f.node_count, 4.0, 16.0, 2500.0, 1.0e8, 600.0
    eye = jnp.eye(2, dtype=jnp.float64)
    cladding_exchange_sign = 1.0 if fault else -1.0

    def residual(temperature: Array, _: None) -> Array:
        tf, tc = temperature[:nf], temperature[nf:]
        zerosf, zerosc = (
            jnp.zeros((f.site_count,), dtype=jnp.float64),
            jnp.zeros((c.site_count,), dtype=jnp.float64),
        )
        rf = thermal_volume_residual(
            f,
            tf,
            zerosf,
            jnp.ones_like(zerosf),
            jnp.broadcast_to(kf * eye, (f.site_count, 2, 2)),
            jnp.full_like(zerosf, q),
            jnp.asarray(1.0),
        )
        rc = thermal_volume_residual(
            c,
            tc,
            zerosc,
            jnp.ones_like(zerosc),
            jnp.broadcast_to(kc * eye, (c.site_count, 2, 2)),
            zerosc,
            jnp.asarray(1.0),
        )
        flux = h * (fi.values(tf) - ci.values(tc))
        rf += fi.assemble(flux)
        rc += cladding_exchange_sign * ci.assemble(fi.weight / ci.weight * flux)
        rc = rc.at[-1].set(tc[-1] - wall)
        return jnp.concatenate((rf, rc))

    temperature, native = _solve(
        residual, jnp.full((nf + c.node_count,), wall, dtype=jnp.float64), "V9"
    )
    flux = h * (fi.values(temperature[:nf]) - ci.values(temperature[nf:]))
    pair = jnp.sum(fi.assemble(flux)) + cladding_exchange_sign * jnp.sum(
        ci.assemble(fi.weight / ci.weight * flux)
    )
    exact = q * spec.pellet.radius / 2
    return {
        "native_solve": native,
        "cells": cells,
        "heat_flux_W_m2": float(flux[0]),
        "exact_heat_flux_W_m2": exact,
        "heat_flux_relative_error": abs(float(flux[0]) - exact) / exact,
        "paired_exchange_error_W": abs(float(pair)),
    }


def constant_stress_creep(*, fault: bool = False) -> dict[str, object]:
    """Integrate a source CREPR tube-site candidate and compare its closed form."""
    from phydrax.applications.nuclear_fuel._cladding_deformation import (
        ZircaloyCreep,
        ZircaloyCreepState,
    )
    from phydrax.applications.nuclear_fuel._forms import mechanical_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    stress, temperature, flux, duration = 80.0e6, 600.0, 5.0e17, 36000.0
    deviator = jnp.diag(
        jnp.asarray((-stress / 3, 2 * stress / 3, -stress / 3), dtype=jnp.float64)
    )
    law = ZircaloyCreep(thermal_coefficient=2.16e9 if fault else 1.08e9)
    state = ZircaloyCreepState.initialize(deviator)
    increment = law.advance(deviator, temperature, flux, 0.0, state, duration)
    # Independent time integral of FRAPCON-4.0 Eqs. 2.87–2.92 (CREPR source units).
    E = 1.148e5 - 59.9 * temperature
    rate = 1.08e9 * E / temperature * math.sinh(650 * 80 / E) ** 2 * math.exp(
        -201000 / (8.314 * temperature)
    ) + 4.0985e-24 * flux**0.85 * 80 * (-7.0237 + 0.0136 * temperature)
    saturation = 0.0216 * rate**0.109 * (2 - math.tanh(3.55e4 * rate)) ** -2.05
    exact = 1.1547 * (
        rate * duration / 3600
        + saturation
        * (
            math.exp(-52 * math.sqrt(rate * 0.1))
            - math.exp(-52 * math.sqrt(rate * duration / 3600))
        )
    )
    # The accepted creep tensor is consumed by the same site Piola-volume route.
    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    g = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=2,
        cladding_cells=4,
        quadrature_order=3,
    )
    elastic_tensor = jnp.broadcast_to(deviator, (g.cladding.site_count, 3, 3))
    assembled = mechanical_volume_residual(g.cladding, elastic_tensor)
    return {
        "equivalent_strain": float(increment.equivalent_strain_increment),
        "closed_form_strain": exact,
        "absolute_error": abs(float(increment.equivalent_strain_increment) - exact),
        "tube_assembled_force_linf_N": float(np.max(np.abs(np.asarray(assembled)))),
        "creep_clock_s": float(increment.state.creep_time),
    }


def power_law_tube(cells: int, /) -> dict[str, object]:
    """Steady Norton tube velocity/stress reproduced in the production Piola form."""
    from phydrax.applications.nuclear_fuel._forms import mechanical_volume_residual
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    g = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=True,
        fuel_cells=2,
        cladding_cells=cells,
        quadrature_order=5,
    )
    body = g.cladding
    a, b = spec.cladding.inner_radius, spec.cladding.outer_radius
    pressure, exponent, coefficient = 2.0e6, 3.0, 1.0e-30
    K = (
        math.sqrt(3)
        * pressure
        / (exponent * (a ** (-2 / exponent) - b ** (-2 / exponent)))
    )
    velocity_coefficient = math.sqrt(3) / 2 * coefficient * K**exponent
    velocity = velocity_coefficient / body.nodes[:, 0]
    radius = body.site_coordinates[:, 0]
    strain_rate = jnp.stack(
        (
            body.gradients(velocity)[:, 0],
            body.values(velocity) / radius,
            jnp.zeros_like(radius),
        ),
        axis=-1,
    )
    deviator = strain_rate - jnp.mean(strain_rate, axis=-1, keepdims=True)
    equivalent_rate = jnp.sqrt(2 / 3 * jnp.sum(deviator**2, axis=-1))
    equivalent_stress = (equivalent_rate / coefficient) ** (1 / exponent)
    stress_deviator = (
        2 / 3 * equivalent_stress[:, None] / equivalent_rate[:, None] * deviator
    )
    radial_exact = (
        exponent * K / math.sqrt(3) * (b ** (-2 / exponent) - radius ** (-2 / exponent))
    )
    hoop_exact = radial_exact + 2 * K / math.sqrt(3) * radius ** (-2 / exponent)
    hydrostatic = (radial_exact + hoop_exact) / 2
    stress = jax.vmap(jnp.diag)(stress_deviator + hydrostatic[:, None])
    rows = mechanical_volume_residual(body, stress)[:, 0]
    rows -= g.cladding_interface.assemble(
        jnp.full_like(g.cladding_interface.weight, pressure)
    )
    force_scale = pressure * 2 * math.pi * a * spec.rod.stack_length
    return {
        "cells": cells,
        "norton_exponent": exponent,
        "velocity_coefficient_m2_s": velocity_coefficient,
        "relative_equilibrium_defect": float(jnp.max(jnp.abs(rows)) / force_scale),
        "volume_rate_linf_s": float(jnp.max(jnp.abs(jnp.sum(strain_rate, axis=-1)))),
    }


def rz_manufactured(
    cells: int, steps: int, /, *, fault: bool = False
) -> dict[str, object]:
    """Coupled RZ thermal/eigenstrain MMS using independent strong-form forcing."""
    from phydrax.applications.nuclear_fuel._forms import (
        mechanical_volume_residual,
        thermal_volume_residual,
    )
    from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry

    spec, history, _, _ = numerical_rod_inputs(axial_layers=1)
    body = fuel_rod_geometry(
        spec,
        history.zone_edges,
        layered=False,
        fuel_cells=cells,
        cladding_cells=2,
        quadrature_order=4,
        axial_cells_per_zone=cells,
    ).fuel
    radius, length, wall, amplitude, tau = (
        spec.pellet.radius,
        spec.rod.stack_length,
        600.0,
        20.0,
        10.0,
    )
    conductivity, capacity, modulus, poisson, expansion = 4.0, 3.0e6, 2.0e11, 0.3, 1.0e-5
    lam = modulus * poisson / ((1 + poisson) * (1 - 2 * poisson))
    shear = modulus / (2 * (1 + poisson))
    nodes, sites = body.nodes, body.site_coordinates
    boundary = (nodes[:, 0] == radius) | (nodes[:, 1] == 0) | (nodes[:, 1] == length)
    mechanical_boundary = boundary | (nodes[:, 0] == 0)
    n = body.node_count
    dt = tau / steps

    def exact_temperature(point: Array, time: Array) -> Array:
        r, z = point
        return wall + amplitude * (1 - jnp.exp(-time / tau)) * (
            1 - (r / radius) ** 2
        ) * jnp.sin(jnp.pi * z / length)

    def exact_displacement(point: Array, time: Array) -> Array:
        r, z = point
        growth = 1 - jnp.exp(-time / tau)
        return (
            1e-5
            * growth
            * jnp.stack(
                (
                    r**3 / radius**2 * jnp.sin(jnp.pi * z / length),
                    r**2 / radius * jnp.sin(jnp.pi * z / length),
                )
            )
        )

    def exact_stress(point: Array, time: Array) -> Array:
        displacement = exact_displacement(point, time)
        gradient = jax.jacfwd(exact_displacement, argnums=0)(point, time)
        strain = jnp.zeros((3, 3), dtype=jnp.float64)
        strain = (
            strain.at[0, 0]
            .set(gradient[0, 0])
            .at[1, 1]
            .set(displacement[0] / point[0])
            .at[2, 2]
            .set(gradient[1, 1])
        )
        strain = strain.at[0, 2].set((gradient[0, 1] + gradient[1, 0]) / 2)
        strain = strain.at[2, 0].set(strain[0, 2])
        strain -= (
            expansion
            * (exact_temperature(point, time) - wall)
            * jnp.eye(3, dtype=jnp.float64)
        )
        return 2 * shear * strain + lam * jnp.trace(strain) * jnp.eye(
            3, dtype=jnp.float64
        )

    def body_force(point: Array, time: Array) -> Array:
        stress = exact_stress(point, time)
        derivative = jax.jacfwd(exact_stress, argnums=0)(point, time)
        # Equilibrium is Div(P) + b = 0; the weak form subtracts the body force.
        return -jnp.stack(
            (
                derivative[0, 0, 0]
                + derivative[0, 2, 1]
                + (stress[0, 0] - stress[1, 1]) / point[0],
                derivative[2, 0, 0] + derivative[2, 2, 1] + stress[2, 0] / point[0],
            )
        )

    temperature = jnp.full((n,), wall, dtype=jnp.float64)
    displacement = jnp.zeros((n, 2), dtype=jnp.float64)
    native_solves: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        time = jnp.asarray(step * dt, dtype=jnp.float64)
        r, z = sites[:, 0], sites[:, 1]
        spatial = (1 - (r / radius) ** 2) * jnp.sin(jnp.pi * z / length)
        temporal = amplitude * (1 - jnp.exp(-time / tau))
        source = capacity * amplitude / tau * jnp.exp(
            -time / tau
        ) * spatial + conductivity * temporal * (
            4 / radius**2 * jnp.sin(jnp.pi * z / length)
            + (jnp.pi / length) ** 2 * spatial
        )
        force = jax.vmap(body_force, in_axes=(0, None))(sites, time)
        if fault:
            force = -force
        exact_t = jax.vmap(exact_temperature, in_axes=(0, None))(nodes, time)
        exact_u = jax.vmap(exact_displacement, in_axes=(0, None))(nodes, time)
        previous_t = temperature

        def residual(x: Array, _: None) -> Array:
            t, u = x[:n], x[n:].reshape((n, 2)) * radius
            tr = thermal_volume_residual(
                body,
                t,
                body.values(t - previous_t),
                jnp.full((body.site_count,), capacity, dtype=jnp.float64),
                jnp.broadcast_to(
                    conductivity * jnp.eye(2, dtype=jnp.float64), (body.site_count, 2, 2)
                ),
                source,
                jnp.asarray(dt),
            )
            gradient = body.gradients(u)
            values = body.values(u)
            strain = jnp.zeros((body.site_count, 3, 3), dtype=jnp.float64)
            strain = (
                strain.at[:, 0, 0]
                .set(gradient[:, 0, 0])
                .at[:, 1, 1]
                .set(values[:, 0] / r)
                .at[:, 2, 2]
                .set(gradient[:, 1, 1])
            )
            strain = strain.at[:, 0, 2].set((gradient[:, 0, 1] + gradient[:, 1, 0]) / 2)
            strain = strain.at[:, 2, 0].set(strain[:, 0, 2])
            strain -= (
                expansion
                * (body.values(t) - wall)[:, None, None]
                * jnp.eye(3, dtype=jnp.float64)
            )
            stress = 2 * shear * strain + lam * jnp.trace(strain, axis1=1, axis2=2)[
                :, None, None
            ] * jnp.eye(3, dtype=jnp.float64)
            mr = mechanical_volume_residual(body, stress, reference_body_force=force)
            tr = jnp.where(boundary, t - exact_t, tr / (conductivity * length))
            mr = jnp.where(
                mechanical_boundary[:, None],
                (u - exact_u) / (radius * 1e-5),
                mr / (modulus * radius * length * 1e-5),
            )
            return jnp.concatenate((tr, mr.reshape(-1)))

        def jacobian(state: Array, args: None) -> la.DenseLinearOperator:
            space = la.PyTreeSpace(state)
            return la.DenseLinearOperator(
                jax.jacfwd(residual, argnums=0)(state, args), source=space, target=space
            )

        solved, native = _solve(
            residual,
            jnp.concatenate((temperature, (displacement / radius).reshape(-1))),
            "V11",
            method=nl.NewtonKrylov(
                linear_policy=la.LinearSolvePolicy(la.DenseLU()),
                jacobian_policy=nl.JacobianPolicy("explicit", operator=jacobian),
            ),
        )
        native_solves.append(native)
        temperature, displacement = solved[:n], solved[n:].reshape((n, 2)) * radius
    exact_t = jax.vmap(exact_temperature, in_axes=(0, None))(sites, jnp.asarray(tau))
    exact_u = jax.vmap(exact_displacement, in_axes=(0, None))(sites, jnp.asarray(tau))
    error_t, error_u = (
        body.values(temperature) - exact_t,
        body.values(displacement) - exact_u,
    )
    weights = np.asarray(body.site_weight)
    return {
        "native_solves": native_solves,
        "radial_cells": cells,
        "axial_cells": cells,
        "steps": steps,
        "temperature_l2_K": float(
            np.sqrt(np.sum(weights * np.asarray(error_t) ** 2) / np.sum(weights))
        ),
        "displacement_l2_m": float(
            np.sqrt(np.sum(weights[:, None] * np.asarray(error_u) ** 2) / np.sum(weights))
        ),
    }


def _history_design(smoke: bool, /) -> tuple[float, float, float]:
    """Explicit numerical designs: short smoke or representative annual base."""
    return (3480.0, 3000.0, 60.0) if smoke else (3.0e7, 20000.0, 3600.0)


def _finite_json_value(value: object, /) -> object:
    """Serialize native IEEE nonfinite evidence explicitly, never erase its value."""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "positive-infinity" if value > 0 else "negative-infinity"
    if isinstance(value, list):
        return [_finite_json_value(member) for member in value]
    return value


def fuel_step_failure_record(evidence: FuelStepEvidence, /) -> dict[str, object]:
    """Materialize native failure diagnostics at the host acceptance boundary."""
    solver = evidence.solver
    physics = evidence.physics
    values = jax.device_get(
        {
            "time_s": evidence.time,
            "time_step_s": evidence.time_step,
            "status": evidence.status,
            "solver": {
                "nonlinear": {
                    "status": solver.nonlinear.status,
                    "residual": solver.nonlinear.residual,
                    "iterations": solver.nonlinear.iterations,
                    "counts_complete": solver.nonlinear.counts_complete,
                },
                "linear": {
                    "status": solver.linear.status,
                    "rank": solver.linear.rank,
                    "condition_estimate": solver.linear.condition_estimate,
                    "residual": solver.linear.residual,
                    "converged": solver.linear.converged,
                    "rank_available": solver.linear.rank_available,
                    "condition_available": solver.linear.condition_available,
                },
                "coupling": {
                    "status": solver.coupling.status,
                    "iterations": solver.coupling.iterations,
                    "evaluations": solver.coupling.evaluations,
                    "joint_residual": solver.coupling.joint_residual,
                    "gas_admissible": solver.coupling.gas_admissible,
                    "thermomechanical_residual_evaluations": solver.coupling.thermomechanical_residual_evaluations,
                    "thermomechanical_iterations": solver.coupling.thermomechanical_iterations,
                    "gas_block_evaluations": solver.coupling.gas_block_evaluations,
                },
                "local_roots_converged": solver.local_roots_converged,
            },
            "physics": {
                "physical": physics.physical,
                "in_range": physics.in_range,
                "minimum_gap": physics.minimum_gap,
                "complementarity_error": physics.complementarity_error,
                "friction_error": physics.friction_error,
                "increments": physics.increments,
            },
            "balances": {
                "energy_error": evidence.balances.energy_error,
                "gas_error": evidence.balances.gas_error,
            },
        }
    )
    support = evidence.source_support
    return {
        **jax.tree.map(
            lambda value: _finite_json_value(np.asarray(value).tolist()), values
        ),
        "source_support": {
            "slots": support.slots,
            "source_ids": support.source_ids,
            "supported": np.asarray(support.supported).tolist(),
        },
        "prepared_id": evidence.prepared_id,
        "bundle_id": evidence.bundle_id,
        "history_id": evidence.history_id,
        "coupling_evaluation_budget": evidence.coupling_evaluation_budget,
        "coupling_tolerance": evidence.coupling_tolerance,
    }


def run_numerical_history(
    prepared: PreparedFuelRod,
    time_step: float,
    checkpoint_directory: Path,
    /,
    *,
    maximum_time_step: float | None = None,
) -> FuelRodResult:
    """Run the actual transactional runtime; refuse incomplete numerical evidence."""
    from phydrax.applications.nuclear_fuel import FuelRodSimulation

    cap = (
        max(
            4 * time_step,
            float(np.max(np.diff(np.asarray(prepared.admission.history.times)))) / 100,
        )
        if maximum_time_step is None
        else maximum_time_step
    )
    simulation = FuelRodSimulation(
        prepared,
        time_step=time_step,
        checkpoint_directory=checkpoint_directory,
        maximum_steps=10000,
        checkpoint_interval=1,
        segment_steps=1,
        minimum_time_step=0.01,
        maximum_time_step=cap,
    )
    try:
        result = simulation.run()
    finally:
        simulation.close()
    if not bool(result.runtime_result.successful):
        diagnostics = tuple(
            fuel_step_failure_record(attempt)
            for attempt in result.attempts
            if not bool(attempt.accepted)
        )
        raise RuntimeError(
            f"Fuel verification run failed: {result.runtime_result.failure}; attempts={diagnostics}"
        )
    if float(result.states[-1].time) != prepared.admission.history.end_time:
        raise RuntimeError(
            "Fuel verification ended before the declared irradiation horizon."
        )
    return result


def history_observables(
    prepared: PreparedFuelRod, state: FuelRodState, /
) -> dict[str, float]:
    """Reference-identified full-rod observables used in refinement comparisons."""
    produced = float(jnp.sum(state.gas.produced_moles[2:4]))
    released = float(jnp.sum(state.gas.released_moles[2:4]))
    outer = prepared.geometry.cladding_outer
    diameter = 2 * (
        outer.reference_radius + outer.values(state.fields.displacement_cladding)[:, 0]
    )
    return {
        "fgr": released / produced if produced > 0 else 0.0,
        "maximum_fuel_temperature_K": float(jnp.max(state.fields.temperature_fuel)),
        "pressure_Pa": float(state.fields.pressure),
        "mean_outer_diameter_m": float(
            jnp.sum(outer.weight * diameter) / jnp.sum(outer.weight)
        ),
    }


def _state_digest(state: FuelRodState, /) -> str:
    """Select the owning array-tree digest without changing its canonical identity."""
    digest = array_tree_fingerprint(state)["sha256"]
    if not isinstance(digest, str):
        raise RuntimeError("The owning array-tree fingerprint omitted its string digest.")
    return digest


def whole_history_balances(
    result: FuelRodResult,
    prepared: PreparedFuelRod,
    /,
    *,
    expected_partition: tuple[float, float] | None = None,
) -> FuelHistoryBalanceRecord:
    """Audit ledgers against independent exact host irradiation-source integrals."""
    energy_errors, gas_errors, source_errors = [], [], []
    history = prepared.admission.history
    knots = np.asarray(history.times)
    power = np.asarray(history.linear_power)
    lengths = np.diff(np.asarray(history.zone_edges))
    fuel_fraction, clad_fraction = (
        (
            float(prepared.admission.bundle.deposition.fuel_fraction),
            float(prepared.admission.bundle.deposition.cladding_fraction),
        )
        if expected_partition is None
        else expected_partition
    )
    for state in result.states:
        energy = state.energy
        deposited = float(
            energy.deposited_fuel
            + energy.deposited_cladding
            + energy.deposited_direct_coolant
        )
        stored = float(energy.stored_fuel + energy.stored_cladding)
        delivered = float(energy.delivered_coolant)
        energy_errors.append(
            abs(stored + delivered - deposited) / max(abs(deposited), abs(stored), 1.0)
        )
        endpoint = float(state.time)
        integration_knots = np.concatenate((knots[knots < endpoint], [endpoint]))
        expected_total = sum(
            np.trapezoid(
                np.interp(integration_knots, knots, power[:, zone]), integration_knots
            )
            * lengths[zone]
            for zone in range(history.zone_count)
        )
        expected = expected_total * np.asarray(
            (fuel_fraction, clad_fraction, 1.0 - fuel_fraction - clad_fraction),
            dtype=np.float64,
        )
        observed = np.asarray(
            (
                energy.deposited_fuel,
                energy.deposited_cladding,
                energy.deposited_direct_coolant,
            ),
            dtype=np.float64,
        )
        source_errors.append(
            float(np.max(np.abs(observed - expected))) / max(abs(expected_total), 1.0)
        )
        gas = state.gas
        delta = np.asarray(
            gas.produced_moles
            - gas.retained_moles
            - gas.released_moles
            - gas.exported_moles
        )
        denominator = np.maximum(
            np.abs(np.asarray(gas.fill_moles)) + np.abs(np.asarray(gas.produced_moles)),
            1e-20,
        )
        gas_errors.append(float(np.max(np.abs(delta) / denominator)))
    source_status: dict[str, str] = {}
    for attempt in result.attempts:
        for source, supported in zip(
            attempt.source_support.source_ids,
            np.asarray(attempt.source_support.supported),
            strict=True,
        ):
            source_status[source] = (
                "equation-extrapolated"
                if not supported or source_status.get(source) == "equation-extrapolated"
                else "source-supported"
            )
    return {
        "accepted_times_s": [float(state.time) for state in result.states],
        "attempted_steps": len(result.attempts),
        "maximum_energy_relative_error": max(energy_errors),
        "maximum_gas_relative_error": max(gas_errors),
        "maximum_source_relative_error": max(source_errors),
        "state_fingerprints": [_state_digest(state) for state in result.states],
        "native_attempts": [
            fuel_step_failure_record(attempt) for attempt in result.attempts
        ],
        "native_terminal_status": result.runtime_result.state.status,
        "source_status": dict(sorted(source_status.items())),
    }


def _source_recipe(prepared: PreparedFuelRod, /) -> dict[str, object]:
    """Retain actual admitted source values and provenance for controlled faults."""
    history, bundle = prepared.admission.history, prepared.admission.bundle
    return {
        "prepared_id": prepared.prepared_id,
        "history_id": history.history_id,
        "bundle_id": bundle.bundle_id,
        "times_s": np.asarray(history.times).tolist(),
        "zone_edges_m": np.asarray(history.zone_edges).tolist(),
        "linear_power_W_per_m": np.asarray(history.linear_power).tolist(),
        "fast_flux_per_m2_s": np.asarray(history.fast_flux).tolist(),
        "fuel_deposition_fraction": float(bundle.deposition.fuel_fraction),
        "cladding_deposition_fraction": float(bundle.deposition.cladding_fraction),
        "deposition_source_id": bundle.deposition.source_id,
        "radial_power_source_id": bundle.radial_power.source_id,
        "scattering_source_id": bundle.radial_power.scattering_source_id,
        "scattering_cross_section_barn": np.asarray(
            bundle.radial_power.scattering_cross_section_barn
        ).tolist(),
    }


def reduction_comparison(
    cells: int,
    time_step: float,
    directory: Path,
    /,
    *,
    smoke: bool = False,
    fault: bool = False,
) -> dict[str, object]:
    """Compare true connected RZ and independent layered meshes at rod midlength."""
    duration, power, transient = _history_design(smoke)
    layered = prepare_numerical_rod(
        fuel_cells=cells,
        axial_layers=3,
        reduction="layered-radial",
        base_duration=duration,
        base_linear_power=power,
        transient_duration=transient,
    )
    rz = prepare_numerical_rod(
        fuel_cells=cells,
        axial_layers=3,
        axial_cells_per_zone=2,
        reduction="axisymmetric-rz",
        base_duration=duration,
        base_linear_power=1.1 * power if fault else power,
        transient_duration=transient,
    )
    result_l = run_numerical_history(layered, time_step, directory / "layered")
    result_r = run_numerical_history(rz, time_step, directory / "rz")
    mid = layered.admission.specification.rod.stack_length / 2

    def profile(prepared: PreparedFuelRod, state: FuelRodState) -> NDArray[np.float64]:
        nodes = np.asarray(prepared.geometry.fuel.nodes)
        plane = np.isclose(nodes[:, 1], mid, rtol=0.0, atol=1e-12)
        if not np.any(plane):
            raise ValueError(
                "Reduction comparison requires an actual common midlength plane."
            )
        order = np.argsort(nodes[plane, 0])
        return np.asarray(state.fields.temperature_fuel)[plane][order]

    tl, tr = profile(layered, result_l.states[-1]), profile(rz, result_r.states[-1])
    left, right = (
        history_observables(layered, result_l.states[-1]),
        history_observables(rz, result_r.states[-1]),
    )
    return {
        "radial_cells": cells,
        "source_multiplier": 1.1 if fault else 1.0,
        "layered_source_recipe": _source_recipe(layered),
        "rz_source_recipe": _source_recipe(rz),
        "layered_balances": whole_history_balances(result_l, layered),
        "rz_balances": whole_history_balances(result_r, rz),
        "rz_axial_cells_per_zone": 2,
        "midlength_temperature_linf_K": float(np.max(np.abs(tl - tr))),
        "pressure_relative_deviation": abs(left["pressure_Pa"] - right["pressure_Pa"])
        / left["pressure_Pa"],
        "diameter_deviation_m": abs(
            left["mean_outer_diameter_m"] - right["mean_outer_diameter_m"]
        ),
        "layered_geometry_id": layered.geometry.geometry_id,
        "rz_geometry_id": rz.geometry.geometry_id,
    }


def _history_refinement_outcome(
    rows: list[dict[str, float | int]], /
) -> tuple[CaseOutcome, dict[str, object]]:
    """Require contraction unless all differences are below float64 resolution."""
    if len(rows) != 3:
        raise ValueError("History refinement requires exactly three ordered rows.")
    report: dict[str, object] = {}
    outcome: CaseOutcome = "passed"
    for name in (
        "fgr",
        "maximum_fuel_temperature_K",
        "pressure_Pa",
        "mean_outer_diameter_m",
    ):
        values = tuple(float(row[name]) for row in rows)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("History refinement observables must be finite.")
        coarse = abs(values[0] - values[1])
        fine = abs(values[1] - values[2])
        scale = max(abs(value) for value in values)
        roundoff_floor = float(np.sqrt(np.finfo(np.float64).eps) * scale)
        roundoff_limited = max(coarse, fine) <= roundoff_floor
        if fine >= coarse and not roundoff_limited:
            outcome = "failed"
        report[name] = {
            "coarse_difference": coarse,
            "fine_difference": fine,
            "roundoff_floor": roundoff_floor,
            "observed_order": (
                math.log2(coarse / fine)
                if not roundoff_limited and coarse > 0 and fine > 0
                else None
            ),
            "status": (
                "roundoff"
                if roundoff_limited
                else "contracting"
                if fine < coarse
                else "noncontracting"
            ),
            "criterion": ("strict-contraction-above-scale-aware-float64-roundoff"),
        }
    return outcome, report


def history_refinement(
    smoke: bool, directory: Path, /
) -> tuple[CaseOutcome, dict[str, object]]:
    """Independent mesh/time refinements of the source-declared assembled history."""
    cells = (4, 8, 16) if not smoke else (2, 4, 8)
    dts = (480.0, 240.0, 120.0) if smoke else (3.0e5, 1.5e5, 7.5e4)
    duration, power, transient = _history_design(smoke)
    spatial: list[dict[str, float | int]] = []
    temporal: list[dict[str, float | int]] = []
    for count in cells:
        prepared = prepare_numerical_rod(
            fuel_cells=count,
            axial_layers=1,
            mode_count=8,
            base_duration=duration,
            base_linear_power=power,
            transient_duration=transient,
        )
        result = run_numerical_history(
            prepared,
            30.0 if smoke else 120.0,
            directory / f"space-{count}",
            maximum_time_step=dts[-1],
        )
        spatial.append(
            {"cells": count, **history_observables(prepared, result.states[-1])}
        )
    prepared = prepare_numerical_rod(
        fuel_cells=cells[-1],
        axial_layers=1,
        mode_count=8,
        base_duration=duration,
        base_linear_power=power,
        transient_duration=transient,
    )
    for dt in dts:
        result = run_numerical_history(
            prepared,
            dt / 4 if smoke else 120.0,
            directory / f"time-{dt}",
            maximum_time_step=dt,
        )
        temporal.append(
            {
                "maximum_time_step_s": dt,
                **history_observables(prepared, result.states[-1]),
            }
        )

    spatial_outcome, spatial_evidence = _history_refinement_outcome(spatial)
    temporal_outcome, temporal_evidence = _history_refinement_outcome(temporal)
    outcome = "passed" if spatial_outcome == temporal_outcome == "passed" else "failed"
    return outcome, {
        "spatial": spatial,
        "temporal": temporal,
        "spatial_richardson": spatial_evidence,
        "temporal_richardson": temporal_evidence,
    }


def bitwise_equal_tree(left: object, right: object, /) -> bool:
    """Compare accepted numerical payload and static scientific identity exactly."""
    if jax.tree.structure(left) != jax.tree.structure(right):
        return False
    return array_tree_fingerprint(left) == array_tree_fingerprint(right)


def rollback_restart(directory: Path, /) -> dict[str, object]:
    """Native failed-attempt rollback followed by an independent checkpoint replay."""
    from phydrax.applications.nuclear_fuel import (
        FuelRodSimulation,
        FuelStepEvidence,
        FuelStepStatus,
    )

    prepared = prepare_numerical_rod(fuel_cells=2, axial_layers=1, mode_count=4)
    faulted = FuelRodSimulation(
        prepared,
        time_step=60.0,
        checkpoint_directory=directory / "fault",
        acceptance_policy=FuelAcceptancePolicy(temperature_increment=1e-12),
    )
    original = faulted.runtime.initial_state(prepared.initial_state())
    failed, attempts = faulted.runtime.step(original)
    rollback = bitwise_equal_tree(original.accepted_state, failed.accepted_state)
    faulted.close()
    if bool(attempts.successful) or not rollback:
        raise RuntimeError("V14: forced increment-limit rejection committed a candidate.")
    evidence = attempts.attempt_evidence
    if not isinstance(evidence, FuelStepEvidence):
        raise TypeError("V14: native rejection omitted fuel acceptance evidence.")
    attempted = np.asarray(attempts.attempt_executed, dtype=np.bool_)
    statuses = np.asarray(evidence.status)[attempted]
    temperature_increments = np.asarray(evidence.physics.increments)[attempted, 0]
    if (
        not np.any(attempted)
        or not np.all(statuses == FuelStepStatus.INCREMENT_LIMIT)
        or not np.all(temperature_increments > 1e-12)
    ):
        raise RuntimeError(
            f"V14: rejection was not the declared temperature-increment fault: {statuses.tolist()}."
        )
    baseline = run_numerical_history(prepared, 60.0, directory / "baseline")
    replay = FuelRodSimulation(
        prepared,
        time_step=60.0,
        checkpoint_directory=directory / "replay",
        minimum_time_step=0.01,
        maximum_time_step=240.0,
    )
    runtime_state = replay.runtime.initial_state(prepared.initial_state())
    for _ in range(2):
        runtime_state, attempt = replay.runtime.step(runtime_state)
        if not bool(attempt.successful):
            raise RuntimeError(
                "V14: checkpoint prefix did not produce an accepted state."
            )
    checkpointed = replay.runtime.checkpoint(runtime_state)
    replay.close()
    reopened = FuelRodSimulation(
        prepared,
        time_step=60.0,
        checkpoint_directory=directory / "replay",
        minimum_time_step=0.01,
        maximum_time_step=240.0,
    )
    fresh_template = reopened.runtime.initial_state(prepared.initial_state())
    restored = reopened.runtime.resume(fresh_template)
    if not bitwise_equal_tree(checkpointed.accepted_state, restored.accepted_state):
        raise RuntimeError("V14: checkpoint loading changed accepted fields or sites.")
    if not bitwise_equal_tree(checkpointed.controller_state, restored.controller_state):
        raise RuntimeError("V14: checkpoint loading changed the step-size controller.")
    if checkpointed.status != "running" or restored.status != "ready":
        raise RuntimeError(
            "V14: reopened checkpoint did not perform the native running-to-ready transition."
        )
    expected_restored = replace(checkpointed, status="ready")
    if not bitwise_equal_tree(expected_restored, restored):
        raise RuntimeError(
            "V14: reopened checkpoint changed the complete native continuation state."
        )
    continued = reopened.run(restart=True)
    reopened.close()
    identical = bitwise_equal_tree(baseline.states[-1], continued.states[-1])
    controller_identical = bitwise_equal_tree(
        baseline.runtime_result.state.controller_state,
        continued.runtime_result.state.controller_state,
    )
    left, right = baseline.runtime_result.state, continued.runtime_result.state
    continuation_identical = bitwise_equal_tree(
        (
            left.time,
            left.step_index,
            left.schedule_cursor,
            left.output_cursor,
            left.moment_states,
            left.trigger_states,
            left.rng_state,
            left.evidence,
        ),
        (
            right.time,
            right.step_index,
            right.schedule_cursor,
            right.output_cursor,
            right.moment_states,
            right.trigger_states,
            right.rng_state,
            right.evidence,
        ),
    )
    if (
        not bool(continued.runtime_result.successful)
        or not identical
        or not controller_identical
        or not continuation_identical
    ):
        raise RuntimeError(
            "V14: restarted full history differs from uninterrupted execution."
        )
    return {
        "fault": "actual-temperature-increment-limit-exhaustion",
        "native_fault_status": failed.status,
        "terminal_failure": bool(attempts.terminal_failure),
        "attempt_executed": attempted.tolist(),
        "native_fault_evidence": fuel_step_failure_record(evidence),
        "rollback_original_fingerprint": array_tree_fingerprint(original.accepted_state),
        "rollback_refused_fingerprint": array_tree_fingerprint(failed.accepted_state),
        "attempted_step_sizes_s": np.asarray(attempts.attempted_step_sizes)[
            attempted
        ].tolist(),
        "attempt_statuses": statuses.tolist(),
        "temperature_increments_K": temperature_increments.tolist(),
        "rollback": "bitwise-identical",
        "checkpoint_time_s": float(checkpointed.time),
        "checkpoint_id": checkpointed.last_checkpoint_id,
        "same-runtime-replay": "bitwise-identical",
        "controller_replay": "bitwise-identical",
        "native_continuation_replay": "bitwise-identical",
        "checkpoint_store_reopened": True,
        "checkpoint_payload_fingerprint": array_tree_fingerprint(checkpointed),
        "reopened_payload_fingerprint": array_tree_fingerprint(restored),
        "baseline_state_fingerprints": [
            array_tree_fingerprint(state) for state in baseline.states
        ],
        "restart_final_fingerprint": array_tree_fingerprint(continued.states[-1]),
        "baseline_balances": whole_history_balances(baseline, prepared),
        "continued_balances": whole_history_balances(continued, prepared),
    }


def _measured(record: Mapping[str, object], name: str, /) -> float:
    value = record[name]
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
    ):
        raise ValueError(f"Scientific measurement {name} is not a finite real scalar.")
    return float(value)


def _refinement_outcome(
    rows: list[dict[str, object]], metric: str, /
) -> tuple[CaseOutcome, dict[str, object]]:
    errors = tuple(_measured(row, metric) for row in rows)
    orders = tuple(
        math.log2(a / b) if a > 0 and b > 0 else None
        for a, b in zip(errors[:-1], errors[1:], strict=True)
    )
    contracting = errors[-1] < errors[0]
    return ("passed" if contracting else "failed"), {
        "samples": rows,
        "observed_orders": orders,
        "criterion": "positive-error-contraction-under-doubled-resolution",
        "metric": metric,
    }


def _nested_record(record: Mapping[str, object], name: str, /) -> Mapping[str, object]:
    value = record[name]
    if not isinstance(value, Mapping):
        raise TypeError(f"Qualification {name} must be a measurement mapping.")
    return value


def _coarse_reference_bound(record: Mapping[str, object], metric: str, /) -> float:
    samples = record["samples"]
    if not isinstance(samples, list) or not samples:
        raise TypeError("A refinement reference requires observed sample mappings.")
    first = samples[0]
    if not isinstance(first, Mapping):
        raise TypeError("A refinement sample must be a measurement mapping.")
    return _measured(first, metric)


def _fault_reference_record(
    fault: str, measurement: Mapping[str, object], metric: str, bound: float, /
) -> dict[str, object]:
    error = _measured(measurement, metric)
    return {
        "fault": fault,
        "measurement": measurement,
        "oracle_metric": metric,
        "oracle_error": error,
        "reference_bound": bound,
        "detected": error > bound,
        "criterion": "independent-original-reference-error-exceeds-baseline-error-bound",
        "scope": "actual-native-configuration-or-assembly-fault-not-mutated-output",
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class CaseContext:
    """Campaign design and the scratch directory owned by one case execution."""

    smoke: bool
    directory: Path

    @property
    def sizes(self) -> tuple[int, int, int]:
        """Doubled resolutions of the isolated refinement studies."""
        return (2, 4, 8) if self.smoke else (8, 16, 32)


type CaseMeasurement = tuple[CaseOutcome, Mapping[str, object]]
type MeasurementRunner = Callable[[CaseContext], CaseMeasurement]
type FaultRunner = Callable[[CaseContext, Mapping[str, object]], dict[str, object]]


def _verdict(passed: bool, /) -> CaseOutcome:
    return "passed" if passed else "failed"


def _measure_constant_conduction(context: CaseContext, /) -> CaseMeasurement:
    return _refinement_outcome(
        [cylinder_conduction(n, nonlinear=False) for n in context.sizes],
        "l2_temperature_K",
    )


def _fault_constant_conduction(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "l2_temperature_K"
    return _fault_reference_record(
        "omitted-volumetric-fission-source",
        cylinder_conduction(context.sizes[-1], nonlinear=False, fault=True),
        metric,
        _coarse_reference_bound(baseline, metric),
    )


def _measure_kirchhoff_conduction(context: CaseContext, /) -> CaseMeasurement:
    return _refinement_outcome(
        [cylinder_conduction(n, nonlinear=True) for n in context.sizes],
        "l2_temperature_K",
    )


def _fault_kirchhoff_conduction(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "l2_temperature_K"
    return _fault_reference_record(
        "omitted-volumetric-fission-source",
        cylinder_conduction(context.sizes[-1], nonlinear=True, fault=True),
        metric,
        _coarse_reference_bound(baseline, metric),
    )


def _measure_lame(context: CaseContext, /) -> CaseMeasurement:
    return _refinement_outcome(
        [lame_gps(n) for n in context.sizes], "stress_relative_linf"
    )


def _fault_lame(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "stress_relative_linf"
    return _fault_reference_record(
        "omitted-pressure-tractions-and-closed-end-resultant",
        lame_gps(context.sizes[-1], fault=True),
        metric,
        _coarse_reference_bound(baseline, metric),
    )


def _measure_finite_strain(context: CaseContext, /) -> CaseMeasurement:
    return _refinement_outcome(
        [finite_strain_tube(n) for n in context.sizes], "relative_equilibrium_defect"
    )


def _fault_finite_strain(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "relative_equilibrium_defect"
    return _fault_reference_record(
        "omitted-current-pressure-interface-traction",
        finite_strain_tube(context.sizes[-1], fault=True),
        metric,
        _coarse_reference_bound(baseline, metric),
    )


def _measure_free_expansion(context: CaseContext, /) -> CaseMeasurement:
    measurement = free_expansion(context.sizes[-1])
    return _verdict(_measured(measurement, "stress_linf_Pa") < 0.01), measurement


def _fault_free_expansion(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    return _fault_reference_record(
        "omitted-multiplicative-eigenstrain",
        free_expansion(context.sizes[-1], fault=True),
        "stress_linf_Pa",
        0.01,
    )


def _measure_shrink_fit(context: CaseContext, /) -> CaseMeasurement:
    return _refinement_outcome(
        [shrink_fit(n) for n in context.sizes], "pressure_relative_error"
    )


def _fault_shrink_fit(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "pressure_relative_error"
    return _fault_reference_record(
        "omitted-fuel-eigendilation",
        shrink_fit(context.sizes[-1], fault=True),
        metric,
        _coarse_reference_bound(baseline, metric),
    )


def _measure_booth(context: CaseContext, /) -> CaseMeasurement:
    measurement = booth_diffusion(16 if context.smoke else 64)
    return _verdict(
        _measured(measurement, "absolute_error")
        <= _measured(measurement, "modal_truncation_bound") + 1e-12
        and _measured(measurement, "gas_balance_relative_error") < 1e-10
    ), measurement


def _fault_booth(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    measurement = booth_diffusion(16 if context.smoke else 64, fault=True)
    return _fault_reference_record(
        "native-Turnbull-diffusivity-doubled-original-Booth-reference",
        measurement,
        "absolute_error",
        _measured(measurement, "modal_truncation_bound") + 1e-12,
    )


def _measure_plenum_pressure(context: CaseContext, /) -> CaseMeasurement:
    measurement = piecewise_temperature_pressure()
    return _verdict(_measured(measurement, "relative_error") < 1e-10), measurement


def _fault_plenum_pressure(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    return _fault_reference_record(
        "native-plenum-temperature-doubled-original-volume-temperature-reference",
        piecewise_temperature_pressure(fault=True),
        "relative_error",
        1e-10,
    )


def _measure_gap_exchange(context: CaseContext, /) -> CaseMeasurement:
    measurement = paired_gap_exchange(context.sizes[-1])
    return _verdict(
        _measured(measurement, "heat_flux_relative_error") < 1e-8
        and _measured(measurement, "paired_exchange_error_W") < 1e-9
    ), measurement


def _fault_gap_exchange(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    return _fault_reference_record(
        "same-sign-rod-interface-heat-exchange",
        paired_gap_exchange(context.sizes[-1], fault=True),
        "paired_exchange_error_W",
        1e-9,
    )


def _measure_creep(context: CaseContext, /) -> CaseMeasurement:
    constant = constant_stress_creep()
    tube_outcome, tube = _refinement_outcome(
        [power_law_tube(n) for n in context.sizes], "relative_equilibrium_defect"
    )
    return _verdict(
        _measured(constant, "absolute_error") < 1e-10 and tube_outcome == "passed"
    ), {"constant_stress": constant, "power_law_tube": tube}


def _fault_creep(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    return _fault_reference_record(
        "native-CREPR-thermal-coefficient-doubled-original-closed-form",
        constant_stress_creep(fault=True),
        "absolute_error",
        1e-10,
    )


def _measure_rz_manufactured(context: CaseContext, /) -> CaseMeasurement:
    grids = (2, 4, 8) if context.smoke else (4, 8, 16)
    spatial_samples = [rz_manufactured(n, 16) for n in grids]
    temporal_samples = [rz_manufactured(grids[-1], n) for n in (2, 4, 8)]
    studies = {
        "thermal_spatial": _refinement_outcome(spatial_samples, "temperature_l2_K"),
        "thermal_temporal": _refinement_outcome(temporal_samples, "temperature_l2_K"),
        "mechanical_spatial": _refinement_outcome(spatial_samples, "displacement_l2_m"),
        "mechanical_temporal": _refinement_outcome(temporal_samples, "displacement_l2_m"),
    }
    return _verdict(all(outcome == "passed" for outcome, _ in studies.values())), {
        name: study for name, (_, study) in studies.items()
    }


def _fault_rz_manufactured(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    metric = "displacement_l2_m"
    return _fault_reference_record(
        "reversed-independent-MMS-body-force-in-production-weak-form",
        rz_manufactured(8 if context.smoke else 16, 16, fault=True),
        metric,
        _coarse_reference_bound(_nested_record(baseline, "mechanical_spatial"), metric),
    )


def _measure_reduction(context: CaseContext, /) -> CaseMeasurement:
    measurement = reduction_comparison(
        2 if context.smoke else 8, 120.0, context.directory, smoke=context.smoke
    )
    return _verdict(
        _measured(measurement, "midlength_temperature_linf_K") < 5.0
        and _measured(measurement, "pressure_relative_deviation") < 0.02
    ), measurement


def _fault_reduction(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    return _fault_reference_record(
        "RZ-source-multiplied-by-1.1-against-original-layered-reference",
        reduction_comparison(
            2 if context.smoke else 8,
            120.0,
            context.directory,
            smoke=context.smoke,
            fault=True,
        ),
        "midlength_temperature_linf_K",
        5.0,
    )


def _measure_history_balances(context: CaseContext, /) -> CaseMeasurement:
    duration, power, transient = _history_design(context.smoke)
    prepared = prepare_numerical_rod(
        fuel_cells=2 if context.smoke else 8,
        axial_layers=1,
        base_duration=duration,
        base_linear_power=power,
        transient_duration=transient,
    )
    result = run_numerical_history(prepared, 120.0, context.directory)
    measurement = whole_history_balances(result, prepared)
    return _verdict(
        _measured(measurement, "maximum_energy_relative_error") < 1e-6
        and _measured(measurement, "maximum_gas_relative_error") < 1e-6
        and _measured(measurement, "maximum_source_relative_error") < 1e-6
    ), measurement


def _fault_history_balances(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    duration, power, transient = _history_design(context.smoke)
    prepared = prepare_numerical_rod(
        fuel_cells=2 if context.smoke else 8,
        axial_layers=1,
        base_duration=duration,
        base_linear_power=power,
        transient_duration=transient,
        fuel_deposition_fraction=0.475,
    )
    result = run_numerical_history(prepared, 120.0, context.directory)
    return _fault_reference_record(
        "native-fuel-deposition-fraction-halved-original-source-partition-oracle",
        whole_history_balances(result, prepared, expected_partition=(0.95, 0.025)),
        "maximum_source_relative_error",
        1e-6,
    )


def _measure_rollback_restart(context: CaseContext, /) -> CaseMeasurement:
    return "passed", rollback_restart(context.directory)


def _fault_rollback_restart(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    # The measured scenario itself executes and exhausts a genuine
    # increment-limit fault before its independent checkpoint replay.
    return {
        "fault": baseline["fault"],
        "detected": True,
        "measurement": baseline,
        "criterion": "actual-INCREMENT_LIMIT-and-complete-bitwise-rollback",
    }


def _measure_history_refinement(context: CaseContext, /) -> CaseMeasurement:
    return history_refinement(context.smoke, context.directory)


def _fault_history_refinement(
    context: CaseContext, baseline: Mapping[str, object], /
) -> dict[str, object]:
    duration, power, transient = _history_design(context.smoke)
    rows = baseline["temporal"]
    if not isinstance(rows, list) or len(rows) != 3:
        raise TypeError("V15 requires three observed baseline temporal rows.")
    coarse, middle = rows[0], rows[1]
    if not isinstance(coarse, Mapping) or not isinstance(middle, Mapping):
        raise TypeError("V15 temporal rows must be measurement mappings.")
    prepared = prepare_numerical_rod(
        fuel_cells=8 if context.smoke else 16,
        axial_layers=1,
        mode_count=8,
        base_duration=duration,
        base_linear_power=1.1 * power,
        transient_duration=transient,
    )
    dt = 120.0 if context.smoke else 7.5e4
    result = run_numerical_history(
        prepared,
        dt / 4 if context.smoke else 120.0,
        context.directory,
        maximum_time_step=dt,
    )
    changed = history_observables(prepared, result.states[-1])
    metric = "maximum_fuel_temperature_K"
    original_difference = abs(_measured(coarse, metric) - _measured(middle, metric))
    wrong_difference = abs(_measured(middle, metric) - changed[metric])
    return {
        "fault": "finest-time-replay-source-multiplied-by-1.1-original-Richardson-reference",
        "measurement": changed,
        "source_multiplier": 1.1,
        "original_base_linear_power_W_per_m": power,
        "fault_base_linear_power_W_per_m": 1.1 * power,
        "fault_source_recipe": _source_recipe(prepared),
        "reused_baseline_rows": (coarse, middle),
        "original_coarse_difference": original_difference,
        "fault_fine_difference": wrong_difference,
        "detected": wrong_difference >= original_difference and wrong_difference > 0,
        "criterion": "actual-wrong-source-refinement-fails-original-contraction",
    }


# Contact/coolant closure history [s, K, W/m, Pa]: zero-power pressurization
# closes the 1 um as-fabricated gap, then power ascension, ramp, load follow,
# shutdown, cooldown and restart drive axial slip of the closed interface.
_CONTACT_TIMES_S = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6)
_CONTACT_INLET_TEMPERATURE_K = (
    300.0,
    305.0,
    305.0,
    305.0,
    305.0,
    305.0,
    305.0,
    300.0,
    305.0,
)
_CONTACT_LINEAR_POWER_W_PER_M = (0.0, 0.0, 100.0, 100.0, 150.0, 50.0, 0.0, 0.0, 100.0)
_CONTACT_PRESSURE_PA = (0.103e6, *(15.5e6,) * 8)
# The smoke design keeps 0-0.6 s: pressurization, closure and the first power step.
_CONTACT_SMOKE_BREAKPOINTS = 4
_CONTACT_INITIAL_GAP_M = 1.0e-6
_CONTACT_FRICTION_COEFFICIENT = 0.1
_CONTACT_FRICTION_SOURCE_ID = "synthetic-numerical-verification:declared-coulomb-friction"
_CONTACT_HYDRAULIC_DIAMETER_M = 0.0118
_CONTACT_FLOW_AREA_PER_ROD_M2 = 8.8e-5
# Interface pressure [Pa] above which a closed trace point carries macroscopic contact.
_CONTACT_ACTIVITY_PA = 1.0e3
# Tangential force [N] above which closed contact carries observable friction.
_FRICTION_ACTIVITY_N = 1.0e-6
# Accepted friction rows satisfy |residual| <= 1e-8 at their 1e3 N row scale.
_COULOMB_BOUND_TOLERANCE_N = 1.0e-5
# Absolute agreement [J/kg] of an enthalpy with its exact host reference.
_ENTHALPY_REFERENCE_TOLERANCE_J_PER_KG = 1.0e-7
# Absolute agreement [W] of the channel qualification energy balance.
_CHANNEL_QUALIFICATION_ENERGY_TOLERANCE_W = 1.0e-7
# Closures without a sourced strict-support law; strict support must name each.
_STRICT_UNSOURCED_SLOTS = ("relocation_history", "plenum_temperature")
_CONTACT_ROWS: tuple[tuple[FuelGeometryReduction, CoolantBoundaryKind], ...] = tuple(
    (
        parse(reduction, FuelGeometryReduction, "reduction"),
        parse(boundary, CoolantBoundaryKind, "coolant_boundary"),
    )
    for reduction in get_args(FuelGeometryReduction)
    for boundary in get_args(CoolantBoundaryKind)
)
_CONTACT_RESTART_ROW: tuple[FuelGeometryReduction, CoolantBoundaryKind] = (
    "layered-radial",
    "prescribed-wall-temperature",
)


class ContactFrame(TypedDict):
    time_s: float
    maximum_contact_pressure_Pa: float
    minimum_normal_gap_m: float
    maximum_friction_force_N: float
    coulomb_excess_N: float
    closed_points: int
    closed_with_friction: bool


class ChannelEnergyStep(TypedDict):
    time_s: float
    maximum_cell_error_W: float
    relative_error: float
    enthalpy_reference_error_J_per_kg: float
    enthalpy_reference_bound_J_per_kg: float
    total_energy_error_W: float
    passed: bool


def contact_coolant_inputs(
    smoke: bool,
    /,
    *,
    applicability: FuelApplicabilityPolicy = "equation-reproduction",
) -> tuple[FuelRodSpecification, IrradiationHistory, FuelModelBundle]:
    """Near-closed-gap rod with declared Coulomb friction and its history.

    The cladding inner radius is the pellet radius plus 1 um; friction is a
    declared synthetic coefficient, not a measured pellet-cladding value.
    """
    specification, _, _, bundle = numerical_rod_inputs(
        mode_count=2,
        axial_layers=1,
        applicability=applicability,
        friction=CoulombFriction(
            coefficient=_CONTACT_FRICTION_COEFFICIENT,
            source_id=_CONTACT_FRICTION_SOURCE_ID,
        ),
    )
    pellet, cladding = specification.pellet, specification.cladding
    specification = FuelRodSpecification(
        pellet,
        CladdingSpecification(
            inner_radius=pellet.radius + _CONTACT_INITIAL_GAP_M,
            outer_radius=cladding.outer_radius,
            roughness=cladding.roughness,
            cold_work=cladding.cold_work,
        ),
        specification.rod,
    )
    count = _CONTACT_SMOKE_BREAKPOINTS if smoke else len(_CONTACT_TIMES_S)
    power = np.asarray(_CONTACT_LINEAR_POWER_W_PER_M[:count], dtype=np.float64)[:, None]
    history = IrradiationHistory(
        times=np.asarray(_CONTACT_TIMES_S[:count], dtype=np.float64),
        zone_edges=np.asarray((0.0, specification.rod.stack_length), dtype=np.float64),
        linear_power=power,
        fast_flux=power * 5.0e13,
        coolant_inlet_temperature=np.asarray(
            _CONTACT_INLET_TEMPERATURE_K[:count], dtype=np.float64
        ),
        coolant_pressure=np.asarray(_CONTACT_PRESSURE_PA[:count], dtype=np.float64),
        coolant_mass_flux=np.full((count,), 3000.0, dtype=np.float64),
    )
    return specification, history, bundle


def qualified_coupled_channel(
    history: IrradiationHistory, transfer: SingleChannelCoolant, /
) -> tuple[CoupledChannelCoolant, dict[str, object]]:
    """Bind a channel whose energy balance passed an independent host reference.

    One backward-Euler step of the channel balance with declared wall and
    direct sources is solved natively and compared with the exact
    inlet-to-outlet triangular solve. The passing record is the numerical
    release evidence of the signed profile that admits the coupled channel.
    """
    condition = history.conditions(0.0)
    water = iapws97_liquid_state(
        condition.coolant_inlet_temperature, condition.coolant_pressure
    )
    masses = (
        np.asarray(water.density)
        * np.asarray(transfer.flow_area_per_rod)
        * np.asarray(history.zone_lengths)
    )
    balance = LiquidChannelHeatBalance(
        masses, channel_id="contact-coolant:independent-channel-energy"
    )
    previous = jnp.broadcast_to(water.specific_enthalpy, (history.zone_count,))
    endpoint = history.conditions(history.end_time)
    inlet = iapws97_liquid_enthalpy(
        endpoint.coolant_inlet_temperature, endpoint.coolant_pressure
    )
    flow = endpoint.coolant_mass_flux * transfer.flow_area_per_rod
    time_step = jnp.asarray(0.1, dtype=jnp.float64)
    wall = jnp.full_like(previous, 3.0)
    direct = jnp.full_like(previous, 1.0)

    def residual(enthalpy: Array) -> Array:
        return (
            balance.residual(
                enthalpy,
                previous,
                time_step,
                mass_flow=flow,
                inlet_enthalpy=inlet,
                wall_power=wall,
                direct_power=direct,
            )
            / 1000.0
        )

    plan = nl.VectorLocalRootPlan(
        history.zone_count, plan_id="contact-coolant:channel-energy-root", tolerance=1e-10
    )
    solved, diagnostics = plan.solve_with_diagnostics(residual, previous)
    if not bool(diagnostics.converged):
        raise RuntimeError("contact-coolant: the channel energy root did not converge.")
    host_step, host_flow = float(time_step), float(flow)
    incoming = float(inlet)
    expected: list[float] = []
    for mass, start, wall_source, direct_source in zip(
        masses, np.asarray(previous), np.asarray(wall), np.asarray(direct), strict=True
    ):
        incoming = (
            mass * start / host_step + host_flow * incoming + wall_source + direct_source
        ) / (mass / host_step + host_flow)
        expected.append(incoming)
    enthalpy_error = float(
        np.max(np.abs(np.asarray(solved) - np.asarray(expected, dtype=np.float64)))
    )
    energy_error = float(
        balance.stored_energy_change(solved, previous) / time_step
        + balance.outlet_power(solved, flow, inlet)
        - jnp.sum(wall + direct)
    )
    if (
        enthalpy_error > _ENTHALPY_REFERENCE_TOLERANCE_J_PER_KG
        or abs(energy_error) > _CHANNEL_QUALIFICATION_ENERGY_TOLERANCE_W
    ):
        raise RuntimeError(
            "contact-coolant: the channel balance differs from its host reference: "
            f"enthalpy {enthalpy_error} J/kg, energy {energy_error} W."
        )
    record: dict[str, object] = {
        "scope": "numerical-channel-energy-not-experimental-release",
        "history_id": history.history_id,
        "reference": "independent-host-triangular-backward-euler",
        "enthalpy_linf_error_J_per_kg": enthalpy_error,
        "energy_error_W": energy_error,
    }
    gate = ReleaseGateEvidence(
        "numerical-energy",
        passed=True,
        evidence_ids=(canonical_fingerprint(record),),
        reviewer_id="independent-analytic-channel-reference",
        issued_at=10,
        expires_at=100,
    )
    profile = CapabilityProfile(
        "contact-coolant.exact-channel-energy",
        "numerical-verification",
        "qualified",
        (CoupledChannelCoolant.support_tuple(history, transfer),),
        required_gates=("numerical-energy",),
        release_evidence=(gate,),
        released=True,
    )
    # Synthetic numerical signing key: the index certifies this tool's own
    # channel reference, never an external release.
    secret = b"contact-coolant-numerical-channel-proof"
    signer = HMACSHA256ReleaseSigner("contact-coolant:channel-signer", secret)
    trust = HMACSHA256TrustPolicy(
        {"contact-coolant:channel-signer": secret},
        maximum_index_age=100,
        maximum_evidence_age=100,
    )
    index = ReleaseIndex.sign((profile,), signer, issued_at=10)
    channel = CoupledChannelCoolant(
        history, transfer, index, trust, profile_id=profile.profile_id, at_time=20
    )
    return channel, {**record, "profile_id": profile.profile_id}


def _contact_coolant_boundary(
    boundary: CoolantBoundaryKind, history: IrradiationHistory, /
) -> tuple[
    PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant,
    dict[str, object] | None,
]:
    """Coolant boundary of one row and, for a coupled channel, its qualification."""
    match boundary:
        case "prescribed-wall-temperature":
            return PrescribedWallTemperature(
                np.asarray(history.times),
                np.asarray(history.coolant_inlet_temperature)[:, None],
            ), None
        case "single-channel":
            return SingleChannelCoolant(
                hydraulic_diameter=_CONTACT_HYDRAULIC_DIAMETER_M,
                flow_area_per_rod=_CONTACT_FLOW_AREA_PER_ROD_M2,
            ), None
        case "coupled-channel":
            return qualified_coupled_channel(
                history,
                SingleChannelCoolant(
                    hydraulic_diameter=_CONTACT_HYDRAULIC_DIAMETER_M,
                    flow_area_per_rod=_CONTACT_FLOW_AREA_PER_ROD_M2,
                ),
            )
        case unsupported:
            assert_never(unsupported)


def _contact_simulation(
    prepared: PreparedFuelRod, execution: FuelNumericalExecution, directory: Path, /
) -> FuelRodSimulation:
    """One step-control configuration shared by the baseline and its restart."""
    from phydrax.applications.nuclear_fuel import FuelRodSimulation

    return FuelRodSimulation(
        prepared,
        time_step=0.05,
        checkpoint_directory=directory,
        checkpoint_interval=1,
        segment_steps=1,
        minimum_time_step=1.0e-3,
        maximum_time_step=0.1,
        maximum_retries=8,
        numerical_execution=execution,
    )


def contact_frame(prepared: PreparedFuelRod, state: FuelRodState, /) -> ContactFrame:
    """Closure and host Coulomb bound of one accepted state's interface.

    The bound ``mu * N`` integrates the contact pressure over the current
    interface measure, rebuilt here from the state's displacements and axial
    strains, independently of the native friction residual.
    """
    friction = prepared.admission.bundle.friction
    if friction is None:
        raise ValueError("Contact closure requires a Coulomb friction closure.")
    geometry, fields = prepared.geometry, state.fields
    fuel, cladding = geometry.fuel_interface, geometry.cladding_interface
    fuel_displacement = fuel.values(fields.displacement_fuel)
    cladding_displacement = cladding.values(fields.displacement_cladding)
    fuel_radius = fuel.reference_radius + fuel_displacement[:, 0]
    cladding_radius = cladding.reference_radius + cladding_displacement[:, 0]
    if geometry.layered:
        # Layered traces stay at their reference height and stretch with the
        # zone axial strain.
        fuel_tangent = jnp.stack(
            (jnp.zeros_like(fuel_radius), 1 + fields.axial_strain_fuel[fuel.zone]),
            axis=-1,
        )
        cladding_tangent = jnp.stack(
            (
                jnp.zeros_like(cladding_radius),
                1 + fields.axial_strain_cladding[cladding.zone],
            ),
            axis=-1,
        )
        fuel_position = jnp.stack((fuel_radius, fuel.reference_z), axis=-1)
        cladding_position = jnp.stack((cladding_radius, cladding.reference_z), axis=-1)
    else:
        fuel_tangent = fuel.axial_gradient(fields.displacement_fuel).at[:, 1].add(1.0)
        cladding_tangent = (
            cladding.axial_gradient(fields.displacement_cladding).at[:, 1].add(1.0)
        )
        fuel_position = jnp.stack(
            (fuel_radius, fuel.reference_z + fuel_displacement[:, 1]), axis=-1
        )
        cladding_position = jnp.stack(
            (cladding_radius, cladding.reference_z + cladding_displacement[:, 1]),
            axis=-1,
        )
    interface = axisymmetric_interface_geometry(
        fuel_position,
        cladding_position,
        fuel_tangent,
        cladding_tangent,
        fuel.reference_radius,
    )
    index = np.asarray(fuel.interface_index)
    pressure = np.asarray(fields.contact_pressure, dtype=np.float64)
    force = np.abs(np.asarray(fields.friction_force, dtype=np.float64))
    measure = np.bincount(
        index,
        weights=np.asarray(fuel.weight * interface.current_measure_ratio),
        minlength=pressure.size,
    )
    bound = float(friction.coefficient) * np.maximum(pressure, 0.0) * measure
    gap = np.asarray(interface.normal_gap, dtype=np.float64)
    closed = (pressure[index] > _CONTACT_ACTIVITY_PA) & (
        np.abs(gap) <= FuelAcceptancePolicy().contact_tolerance
    )
    return {
        "time_s": float(state.time),
        "maximum_contact_pressure_Pa": float(np.max(pressure)),
        "minimum_normal_gap_m": float(np.min(gap)),
        "maximum_friction_force_N": float(np.max(force)),
        "coulomb_excess_N": float(np.max(np.maximum(force - bound, 0.0))),
        "closed_points": int(np.count_nonzero(closed)),
        "closed_with_friction": bool(
            np.any(closed & (force[index] > _FRICTION_ACTIVITY_N))
        ),
    }


def _channel_step_power(
    prepared: PreparedFuelRod, previous: FuelRodState, state: FuelRodState, /
) -> NDArray[np.float64]:
    """Wall plus direct power [W] into each channel zone over one accepted step.

    Wall power integrates the endpoint coolant heat flux over the current outer
    surface; direct heating is the exact step mean of the declared history.
    """
    geometry, history = prepared.geometry, prepared.admission.history
    deposition = prepared.admission.bundle.deposition
    start, end = float(previous.time), float(state.time)
    times = np.asarray(history.times)
    power = np.asarray(history.linear_power)
    knots = np.concatenate(([start], times[(times > start) & (times < end)], [end]))
    deposited = np.asarray(
        [
            np.trapezoid(np.interp(knots, times, power[:, zone]), knots)
            for zone in range(history.zone_count)
        ],
        dtype=np.float64,
    ) * np.asarray(geometry.layer_lengths)
    direct = (
        (1.0 - float(deposition.fuel_fraction) - float(deposition.cladding_fraction))
        * deposited
        / (end - start)
    )
    outer = geometry.cladding_outer
    radius = (
        outer.reference_radius + outer.values(state.fields.displacement_cladding)[:, 0]
    )
    if geometry.layered:
        stretch = 1 + state.fields.axial_strain_cladding[outer.zone]
    else:
        stretch = jnp.linalg.norm(
            outer.axial_gradient(state.fields.displacement_cladding).at[:, 1].add(1.0),
            axis=-1,
        )
    flux = geometry.coolant_flux_values(state.fields.coolant_heat_flux)
    wall = np.bincount(
        np.asarray(outer.zone),
        weights=np.asarray(
            outer.weight * radius / outer.reference_radius * stretch * flux
        ),
        minlength=history.zone_count,
    )
    return wall + direct


def _steady_channel_step(
    transfer: SingleChannelCoolant,
    prepared: PreparedFuelRod,
    previous: FuelRodState,
    state: FuelRodState,
    tolerance: float,
    /,
) -> ChannelEnergyStep:
    """Steady channel against ``h_in + Q_upstream / (G A)`` at zone midpoints."""
    end = float(state.time)
    condition = prepared.admission.history.conditions(end)
    total = _channel_step_power(prepared, previous, state)
    flow = float(condition.coolant_mass_flux * transfer.flow_area_per_rod)
    inlet = float(
        iapws97_liquid_enthalpy(
            condition.coolant_inlet_temperature, condition.coolant_pressure
        )
    )
    midpoint = np.cumsum(total) - 0.5 * total
    enthalpy = np.asarray(
        transfer.bulk_enthalpy(
            inlet_temperature=condition.coolant_inlet_temperature,
            pressure=condition.coolant_pressure,
            mass_flux=condition.coolant_mass_flux,
            cumulative_power_to_coolant=jnp.asarray(midpoint),
        )
    )
    reference_error = float(np.max(np.abs(enthalpy - (inlet + midpoint / flow))))
    exiting = enthalpy + 0.5 * total / flow
    incoming = np.concatenate(([inlet], exiting[:-1]))
    defects = flow * (exiting - incoming) - total
    error = float(np.max(np.abs(defects)))
    scale = max(float(np.max(np.abs(total))), 1.0)
    return {
        "time_s": end,
        "maximum_cell_error_W": error,
        "relative_error": error / scale,
        "enthalpy_reference_error_J_per_kg": reference_error,
        "enthalpy_reference_bound_J_per_kg": _ENTHALPY_REFERENCE_TOLERANCE_J_PER_KG,
        "total_energy_error_W": float(flow * (exiting[-1] - inlet) - np.sum(total)),
        "passed": error / scale <= tolerance
        and reference_error <= _ENTHALPY_REFERENCE_TOLERANCE_J_PER_KG,
    }


def _coupled_channel_step(
    channel: CoupledChannelCoolant,
    prepared: PreparedFuelRod,
    previous: FuelRodState,
    state: FuelRodState,
    tolerance: float,
    /,
) -> ChannelEnergyStep:
    """Transient channel against the exact inlet-to-outlet backward-Euler solve.

    An accepted root with per-cell relative energy defect ``tolerance`` moves
    each cell's backward-Euler enthalpy by at most ``defect / (m / dt + flow)``;
    upstream contributions contract, so the cell count bounds the accumulation.
    """
    start, end = float(previous.time), float(state.time)
    time_step = end - start
    condition = prepared.admission.history.conditions(end)
    total = _channel_step_power(prepared, previous, state)
    flow = float(condition.coolant_mass_flux * channel.wall_transfer.flow_area_per_rod)
    inlet = float(
        iapws97_liquid_enthalpy(
            condition.coolant_inlet_temperature, condition.coolant_pressure
        )
    )
    masses = np.asarray(channel.balance.reference_cell_masses, dtype=np.float64)
    enthalpy = np.asarray(state.fields.channel_state, dtype=np.float64)
    old = np.asarray(previous.fields.channel_state, dtype=np.float64)
    accumulation = masses * (enthalpy - old) / time_step
    incoming = np.concatenate(([inlet], enthalpy[:-1]))
    defects = accumulation + flow * (enthalpy - incoming) - total
    reference = inlet
    expected: list[float] = []
    for mass, start_enthalpy, source in zip(
        masses.tolist(), old.tolist(), total.tolist(), strict=True
    ):
        reference = (mass * start_enthalpy / time_step + flow * reference + source) / (
            mass / time_step + flow
        )
        expected.append(reference)
    reference_error = float(
        np.max(np.abs(enthalpy - np.asarray(expected, dtype=np.float64)))
    )
    error = float(np.max(np.abs(defects)))
    scale = max(float(np.max(np.abs(total))), float(np.max(np.abs(accumulation))), 1.0)
    bound = max(
        _ENTHALPY_REFERENCE_TOLERANCE_J_PER_KG + 1e-12 * float(np.max(np.abs(enthalpy))),
        tolerance * scale * total.size / float(np.min(masses / time_step + flow)),
    )
    return {
        "time_s": end,
        "maximum_cell_error_W": error,
        "relative_error": error / scale,
        "enthalpy_reference_error_J_per_kg": reference_error,
        "enthalpy_reference_bound_J_per_kg": bound,
        "total_energy_error_W": float(
            np.sum(accumulation) + flow * (enthalpy[-1] - inlet) - np.sum(total)
        ),
        "passed": error / scale <= tolerance and reference_error <= bound,
    }


def coolant_energy_audit(
    prepared: PreparedFuelRod, states: tuple[FuelRodState, ...], /
) -> list[ChannelEnergyStep]:
    """Per-cell channel energy of every accepted step; none for a prescribed wall."""
    tolerance = FuelAcceptancePolicy().relative_energy_tolerance
    coolant = prepared.admission.coolant
    steps: list[ChannelEnergyStep] = []
    for previous, state in pairwise(states):
        match coolant:
            case PrescribedWallTemperature():
                return []
            case SingleChannelCoolant():
                steps.append(
                    _steady_channel_step(coolant, prepared, previous, state, tolerance)
                )
            case CoupledChannelCoolant():
                steps.append(
                    _coupled_channel_step(coolant, prepared, previous, state, tolerance)
                )
            case unsupported:
                assert_never(unsupported)
    return steps


def run_contact_history(
    prepared: PreparedFuelRod, execution: FuelNumericalExecution, directory: Path, /
) -> FuelRodResult:
    """Uninterrupted run; a refused history is returned with its attempt evidence."""
    simulation = _contact_simulation(prepared, execution, directory)
    try:
        return simulation.run()
    finally:
        simulation.close()


def contact_coolant_row(
    prepared: PreparedFuelRod, result: FuelRodResult, /
) -> dict[str, object]:
    """Contact closure, Coulomb bound, channel energy, and every attempt status."""
    frames = [contact_frame(prepared, state) for state in result.states]
    energy = coolant_energy_audit(prepared, result.states)
    channel = not isinstance(prepared.admission.coolant, PrescribedWallTemperature)
    checks = {
        "completed": bool(result.successful)
        and float(result.states[-1].time) == prepared.admission.history.end_time,
        "contact_closed": any(frame["closed_points"] > 0 for frame in frames),
        "coulomb_bound": all(
            frame["coulomb_excess_N"] <= _COULOMB_BOUND_TOLERANCE_N for frame in frames
        ),
        "friction_while_closed": any(frame["closed_with_friction"] for frame in frames),
        "coolant_energy": all(step["passed"] for step in energy)
        and (bool(energy) or not channel),
    }
    return {
        "prepared_id": prepared.prepared_id,
        "geometry_id": prepared.geometry.geometry_id,
        "native_terminal_status": result.runtime_result.state.status,
        "accepted_times_s": [float(state.time) for state in result.states],
        "native_attempts": [
            fuel_step_failure_record(attempt) for attempt in result.attempts
        ],
        "contact_frames": frames,
        "coulomb_bound_tolerance_N": _COULOMB_BOUND_TOLERANCE_N,
        "contact_activity_threshold_Pa": _CONTACT_ACTIVITY_PA,
        "friction_activity_threshold_N": _FRICTION_ACTIVITY_N,
        "coolant_energy": energy,
        "checks": checks,
        "outcome": _verdict(all(checks.values())),
    }


def contact_restart_replay(
    prepared: PreparedFuelRod,
    execution: FuelNumericalExecution,
    baseline: FuelRodResult,
    directory: Path,
    /,
) -> dict[str, object]:
    """Checkpoint after two accepted steps, reopen the store, and resume.

    The interrupted prefix followed by the resumed continuation must reproduce
    every accepted state of the uninterrupted run bitwise.
    """
    interrupted = _contact_simulation(prepared, execution, directory)
    runtime_state = interrupted.runtime.initial_state(prepared.initial_state())
    prefix = [runtime_state.accepted_state]
    for _ in range(2):
        runtime_state, transition = interrupted.runtime.step(runtime_state)
        if not bool(transition.successful):
            interrupted.close()
            raise RuntimeError("contact-coolant: a checkpoint prefix step was refused.")
        prefix.append(runtime_state.accepted_state)
    checkpointed = interrupted.runtime.checkpoint(runtime_state)
    interrupted.close()
    reopened = _contact_simulation(prepared, execution, directory)
    try:
        resumed = reopened.run(restart=True)
    finally:
        reopened.close()
    # The resumed result starts at the checkpointed state, the prefix's last.
    replayed = (*prefix[:-1], *resumed.states)
    identical = len(replayed) == len(baseline.states) and all(
        bitwise_equal_tree(left, right)
        for left, right in zip(replayed, baseline.states, strict=True)
    )
    return {
        "checkpoint_time_s": float(checkpointed.time),
        "checkpoint_id": checkpointed.last_checkpoint_id,
        "prefix_accepted_steps": len(prefix) - 1,
        "resumed_successful": bool(resumed.successful),
        "baseline_state_fingerprints": [
            array_tree_fingerprint(state) for state in baseline.states
        ],
        "replayed_state_fingerprints": [
            array_tree_fingerprint(state) for state in replayed
        ],
        "bitwise_identical": identical,
        "criterion": "every-accepted-state-bitwise-equal-to-uninterrupted-run",
    }


def strict_source_refusal(smoke: bool, /) -> dict[str, object]:
    """Strict source support must refuse the bundle and name each unsourced slot."""
    try:
        contact_coolant_inputs(smoke, applicability="strict-source-support")
    except ValueError as error:
        message = str(error)
    else:
        return {
            "applicability": "strict-source-support",
            "refused": False,
            "message": None,
            "required_slots": _STRICT_UNSOURCED_SLOTS,
            "named_slots": False,
        }
    return {
        "applicability": "strict-source-support",
        "refused": True,
        "message": message,
        "required_slots": _STRICT_UNSOURCED_SLOTS,
        "named_slots": all(slot in message for slot in _STRICT_UNSOURCED_SLOTS),
    }


def _measure_contact_coolant(context: CaseContext, /) -> CaseMeasurement:
    """Six reduction/coolant rows, one restart replay, one strict-source refusal."""
    from phydrax.applications.nuclear_fuel import (
        admit_fuel_rod,
        FuelNumericalExecution,
        prepare_fuel_rod,
    )

    specification, history, bundle = contact_coolant_inputs(context.smoke)
    rows: list[dict[str, object]] = []
    restart: dict[str, object] | None = None
    for reduction, boundary in _CONTACT_ROWS:
        coolant, channel_qualification = _contact_coolant_boundary(boundary, history)
        prepared = prepare_fuel_rod(
            admit_fuel_rod(
                specification,
                history,
                coolant,
                bundle=bundle,
                reduction=reduction,
                axial_interaction="coulomb",
            ),
            fuel_cells=2,
            cladding_cells=2,
            quadrature_order=2,
        )
        execution = FuelNumericalExecution(prepared)
        directory = context.directory / f"{reduction}-{boundary}"
        result = run_contact_history(prepared, execution, directory / "baseline")
        rows.append(
            {
                "reduction": reduction,
                "coolant_boundary": boundary,
                "channel_qualification": channel_qualification,
                **contact_coolant_row(prepared, result),
            }
        )
        if (reduction, boundary) == _CONTACT_RESTART_ROW:
            restart = contact_restart_replay(
                prepared, execution, result, directory / "restart"
            )
    if restart is None:
        raise RuntimeError("contact-coolant: the restart row is not a declared row.")
    refusal = strict_source_refusal(context.smoke)
    passed = (
        all(row["outcome"] == "passed" for row in rows)
        and restart["bitwise_identical"] is True
        and refusal["refused"] is True
        and refusal["named_slots"] is True
    )
    return _verdict(passed), {
        "design": {
            "times_s": np.asarray(history.times).tolist(),
            "linear_power_W_per_m": np.asarray(history.linear_power).tolist(),
            "coolant_inlet_temperature_K": np.asarray(
                history.coolant_inlet_temperature
            ).tolist(),
            "coolant_pressure_Pa": np.asarray(history.coolant_pressure).tolist(),
            "initial_gap_m": _CONTACT_INITIAL_GAP_M,
            "friction_coefficient": _CONTACT_FRICTION_COEFFICIENT,
            "friction_source_id": _CONTACT_FRICTION_SOURCE_ID,
            "applicability": bundle.applicability,
            "history_id": history.history_id,
            "bundle_id": bundle.bundle_id,
        },
        "rows": rows,
        "restart": restart,
        "strict_source_refusal": refusal,
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class QualificationCase:
    """One verification case: report identity, measurement, and deliberate fault.

    Release-closure cases carry their package row's reference and metric and a
    fault runner. ``fault=None`` marks a supplemental case outside the release
    closure matrix; it is reported separately and never aggregated.
    """

    case_id: str
    reference: str
    metric: str
    measure: MeasurementRunner
    fault: FaultRunner | None


def _closure_case(
    case_id: str, measure: MeasurementRunner, fault: FaultRunner, /
) -> QualificationCase:
    row = {row.case_id: row for row in FUEL_VERIFICATION_CASES}[case_id]
    return QualificationCase(
        case_id=row.case_id,
        reference=row.reference,
        metric=row.metric,
        measure=measure,
        fault=fault,
    )


# Campaign order: release-closure cases in package order, then supplemental cases.
QUALIFICATION_CASES: tuple[QualificationCase, ...] = (
    _closure_case("V1", _measure_constant_conduction, _fault_constant_conduction),
    _closure_case("V2", _measure_kirchhoff_conduction, _fault_kirchhoff_conduction),
    _closure_case("V3", _measure_lame, _fault_lame),
    _closure_case("V4", _measure_finite_strain, _fault_finite_strain),
    _closure_case("V5", _measure_free_expansion, _fault_free_expansion),
    _closure_case("V6", _measure_shrink_fit, _fault_shrink_fit),
    _closure_case("V7", _measure_booth, _fault_booth),
    _closure_case("V8", _measure_plenum_pressure, _fault_plenum_pressure),
    _closure_case("V9", _measure_gap_exchange, _fault_gap_exchange),
    _closure_case("V10", _measure_creep, _fault_creep),
    _closure_case("V11", _measure_rz_manufactured, _fault_rz_manufactured),
    _closure_case("V12", _measure_reduction, _fault_reduction),
    _closure_case("V13", _measure_history_balances, _fault_history_balances),
    _closure_case("V14", _measure_rollback_restart, _fault_rollback_restart),
    _closure_case("V15", _measure_history_refinement, _fault_history_refinement),
    QualificationCase(
        case_id="contact-coolant",
        reference="host-coulomb-bound-and-backward-euler-channel-energy",
        metric="closure-friction-bound-channel-energy-restart-strict-refusal",
        measure=_measure_contact_coolant,
        fault=None,
    ),
)


def select_cases(case_ids: tuple[str, ...] | None, /) -> tuple[QualificationCase, ...]:
    """Registered cases in campaign order; unknown or repeated IDs are refused."""
    if case_ids is None:
        return QUALIFICATION_CASES
    known = {case.case_id for case in QUALIFICATION_CASES}
    if not case_ids or len(set(case_ids)) != len(case_ids) or set(case_ids) - known:
        raise ValueError(
            "Select unique registered qualification cases: " + ", ".join(sorted(known))
        )
    return tuple(case for case in QUALIFICATION_CASES if case.case_id in case_ids)


def _report_mapping(value: object, subject: str, /) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{subject} must be a string-keyed report mapping.")
    return value


def _report_string(record: Mapping[str, object], name: str, /) -> str:
    value = record.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Observed report {name} must be a nonempty string.")
    return value


def _report_identity(report: Mapping[str, object], /) -> tuple[str, str, str]:
    receipt = _nested_record(report, "source_receipt")
    required = {
        "build_before",
        "build_after",
        "driver_before",
        "driver_after",
        "capture",
        "environment",
    }
    if required - set(receipt):
        raise ValueError(
            "Observed campaign receipt omitted required source identity fields."
        )
    build = _report_string(receipt, "build_before")
    driver = _report_string(receipt, "driver_before")
    if build != _report_string(receipt, "build_after") or driver != _report_string(
        receipt, "driver_after"
    ):
        raise ValueError("Observed campaign source changed during execution.")
    if receipt["capture"] != "before-and-after-actual-campaign":
        raise ValueError(
            "Aggregation requires an actual executed-campaign source receipt."
        )
    environment = _nested_record(receipt, "environment")
    return build, driver, _report_string(environment, "fingerprint")


def _scientific_wire_artifact(
    artifact: Mapping[str, object], /
) -> tuple[dict[str, object], str]:
    """Hash the owning canonical JSON payload, not transient host scalar types.

    JSON scalars have no NumPy dtype tag. Normalize once through the existing
    canonical JSON owner before both publication and content addressing; the
    strict reader uses this same boundary, without restoring guessed types.
    """
    wire = canonical_json(artifact)
    payload = dict(_report_mapping(json.loads(wire), "Scientific JSON artifact"))
    return payload, canonical_fingerprint(payload)


def _validated_report_row(
    row: Mapping[str, object], identity: tuple[str, str, str], /
) -> tuple[str, str, str]:
    required = {
        "case_id",
        "design",
        "reference",
        "metric",
        "scope",
        "outcome",
        "fault_adequacy",
        "raw_artifact_id",
        "fault_artifact_id",
        "source_build_id",
        "driver_id",
        "environment_id",
    }
    if required - set(row):
        raise ValueError(
            "Observed case omitted required identity, criteria, or witness fields."
        )
    case_id = _report_string(row, "case_id")
    known = {case.case_id: case for case in FUEL_VERIFICATION_CASES}
    if case_id not in known:
        raise ValueError(f"Unknown observed verification case {case_id}.")
    parse(row["design"], FuelCampaignDesign, "campaign_design")
    if (
        row["reference"] != known[case_id].reference
        or row["metric"] != known[case_id].metric
    ):
        raise ValueError(
            "Observed case does not retain its original reference and metric."
        )
    if (
        row["scope"]
        != "synthetic-numerical-equation-reproduction-not-scientific-validation"
    ):
        raise ValueError(
            "Observed numerical cases must retain their declared numerical scope."
        )
    if row["outcome"] != "passed":
        raise ValueError(f"Observed numerical case {case_id} did not pass.")
    fault = _nested_record(row, "fault_adequacy")
    if {
        "case_id",
        "design",
        "detected",
        "source_build_id",
        "driver_id",
        "environment_id",
    } - set(fault):
        raise ValueError(
            "Observed fault artifact omitted required scope or source fields."
        )
    if (
        fault["case_id"] != case_id
        or fault["design"] != row["design"]
        or fault["detected"] is not True
    ):
        raise ValueError(f"Observed fault adequacy for {case_id} did not pass.")
    for artifact in (row, fault):
        actual = tuple(
            _report_string(artifact, name)
            for name in ("source_build_id", "driver_id", "environment_id")
        )
        if actual != identity:
            raise ValueError(
                "Observed artifact source differs from its campaign receipt."
            )
    numeric = {
        key: value
        for key, value in row.items()
        if key not in ("raw_artifact_id", "fault_adequacy", "fault_artifact_id")
    }
    numeric_id = _report_string(row, "raw_artifact_id")
    fault_id = _report_string(row, "fault_artifact_id")
    _, expected_numeric_id = _scientific_wire_artifact(numeric)
    _, expected_fault_id = _scientific_wire_artifact(fault)
    if expected_numeric_id != numeric_id or expected_fault_id != fault_id:
        raise ValueError(
            "Observed numerical or fault content address does not match its artifact."
        )
    if numeric_id == fault_id:
        raise ValueError(
            "Numerical and fault witnesses must retain distinct content addresses."
        )
    return case_id, numeric_id, fault_id


def _load_observed_report(
    path: Path, /
) -> tuple[Mapping[str, object], tuple[str, str, str], list[dict[str, object]]]:
    report = _report_mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    required = {
        "source_receipt",
        "successful",
        "cases",
        "artifact_directory",
        "campaign_designs",
    }
    if required - set(report):
        raise ValueError(
            "Observed input report omitted required campaign receipt fields."
        )
    identity = _report_identity(report)
    if report["successful"] is not True:
        raise ValueError(
            "Only actually successful numerical and fault campaigns may be aggregated."
        )
    values = report["cases"]
    if not isinstance(values, list) or not values:
        raise ValueError("An observed report must retain nonempty actual case rows.")
    rows = [dict(_report_mapping(value, "Observed case")) for value in values]
    declared = report["campaign_designs"]
    if not isinstance(declared, list) or not declared:
        raise ValueError(
            "Observed reports must retain explicit declared campaign designs."
        )
    declared_designs = {
        parse(value, FuelCampaignDesign, "campaign_design") for value in declared
    }
    observed_designs = {
        parse(row.get("design"), FuelCampaignDesign, "campaign_design") for row in rows
    }
    if observed_designs != declared_designs:
        raise ValueError(
            "Observed case designs differ from the campaign's literal declaration."
        )
    directory = report["artifact_directory"]
    if directory is not None and not isinstance(directory, str):
        raise TypeError(
            "Observed artifact directory must retain its literal path or null."
        )
    return report, identity, rows


def aggregate_reports(input_reports: tuple[Path, ...], /) -> dict[str, object]:
    """Validate retained observations and compose the existing native closure matrix.

    This host interchange never executes physics, generates a successful case,
    converts a smoke observation into a default design, or repairs failed data.
    Separate smoke/full aggregates may share their actually observed isolated
    cases; duplicate cases inside one aggregate are refused.
    """
    if not input_reports or len({path.resolve() for path in input_reports}) != len(
        input_reports
    ):
        raise ValueError("Select nonempty unique observed input report paths.")
    identity: tuple[str, str, str] | None = None
    cases: dict[str, dict[str, object]] = {}
    numerical: dict[str, tuple[str, ...]] = {}
    faults: dict[str, tuple[str, ...]] = {}
    used_artifacts: set[str] = set()
    receipts: list[Mapping[str, object]] = []
    locations: list[dict[str, object]] = []
    for path in input_reports:
        report, actual_identity, rows = _load_observed_report(path)
        if identity is None:
            identity = actual_identity
        elif actual_identity != identity:
            raise ValueError(
                "Do not aggregate mixed global build, driver, or environment identities."
            )
        for row in rows:
            case_id, numeric_id, fault_id = _validated_report_row(row, actual_identity)
            if case_id in cases:
                raise ValueError(
                    f"Duplicate observed case {case_id}; preserve separate design aggregates."
                )
            if used_artifacts.intersection((numeric_id, fault_id)):
                raise ValueError(
                    "Observed content addresses cannot be reused as another case or witness."
                )
            used_artifacts.update((numeric_id, fault_id))
            cases[case_id] = row
            numerical[case_id], faults[case_id] = (numeric_id,), (fault_id,)
        locations.append(
            {
                "input_report": str(path),
                "artifact_directory": report["artifact_directory"],
            }
        )
        receipts.append(_nested_record(report, "source_receipt"))
    if identity is None:
        raise RuntimeError("Nonempty observed reports omitted their source identity.")
    ordered = [
        cases[case.case_id] for case in FUEL_VERIFICATION_CASES if case.case_id in cases
    ]
    implementation, scientific = nuclear_fuel_closure_matrices(
        numerical, fault_adequacy_evidence=faults
    )
    return {
        "cases": ordered,
        "campaign_designs": tuple(
            sorted({_report_string(row, "design") for row in ordered})
        ),
        "source_identity": dict(
            zip(("build", "driver", "environment"), identity, strict=True)
        ),
        "input_source_receipts": receipts,
        "artifact_locations": locations,
        "aggregation_scope": "retained-observed-reports-no-physics-execution",
        "successful": True,
        "implementation_closure": implementation.to_record(),
        "scientific_release_closure": scientific.to_record(),
        "release_status": "candidate-unreleased",
        "external_prerequisites": (
            "NEA-1664/03-actual-originator-agreement",
            "independent-locked-evaluation",
        ),
        "code_to_code_cases": (),
        "experimental_validation_cases": (),
    }


def _write_json(path: Path, payload: Mapping[str, object], /) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def qualify(
    *,
    smoke: bool = False,
    cases: tuple[str, ...] | None = None,
    artifact_directory: Path | None = None,
) -> dict[str, object]:
    """Execute requested numerical cases and retain source support as a nonclaim.

    Release-closure cases resolve the native closure matrix; supplemental cases
    are reported under ``supplemental_cases`` and only gate ``successful``.
    """
    root = Path(__file__).resolve().parents[1]
    driver = Path(__file__).resolve()
    build_before = source_build_fingerprint(root)
    driver_before = benchmark_driver_fingerprint(root, driver)
    environment = capture_environment().to_dict()
    campaign_start = time.perf_counter()
    selected = select_cases(cases)
    design: FuelCampaignDesign = "smoke" if smoke else "default-full"
    provenance = {
        "source_build_id": build_before,
        "driver_id": driver_before,
        "environment_id": environment["fingerprint"],
    }
    records: list[dict[str, object]] = []
    supplemental: list[dict[str, object]] = []
    evidence: dict[str, tuple[str, ...]] = {}
    fault_evidence: dict[str, tuple[str, ...]] = {}
    if artifact_directory is not None:
        artifact_directory.mkdir(parents=True, exist_ok=False)
    with (
        TemporaryDirectory(prefix="phydrax-fuel-qualification-")
        if artifact_directory is None
        else nullcontext(str(artifact_directory))
    ) as temporary:
        directory = Path(temporary)
        for case in selected:
            case_start = time.perf_counter()
            outcome, measurement = case.measure(
                CaseContext(smoke=smoke, directory=directory / case.case_id)
            )
            artifact, artifact_id = _scientific_wire_artifact(
                {
                    "case_id": case.case_id,
                    "reference": case.reference,
                    "metric": case.metric,
                    "outcome": outcome,
                    "design": design,
                    "measurement": measurement,
                    "wall_s": time.perf_counter() - case_start,
                    **provenance,
                    "scope": "synthetic-numerical-equation-reproduction-not-scientific-validation",
                }
            )
            if case.fault is None:
                record = {**artifact, "raw_artifact_id": artifact_id}
                supplemental.append(record)
                if artifact_directory is not None:
                    _write_json(directory / f"{case.case_id}.json", record)
                continue
            if artifact_directory is not None:
                _write_json(
                    directory / f"{case.case_id}.numerical.json",
                    {**artifact, "raw_artifact_id": artifact_id},
                )
            fault = case.fault(
                CaseContext(smoke=smoke, directory=directory / f"{case.case_id}-fault"),
                measurement,
            )
            fault_artifact, fault_id = _scientific_wire_artifact(
                {"case_id": case.case_id, "design": design, **provenance, **fault}
            )
            record = {
                **artifact,
                "raw_artifact_id": artifact_id,
                "fault_adequacy": fault_artifact,
                "fault_artifact_id": fault_id,
            }
            records.append(record)
            if artifact_directory is not None:
                _write_json(directory / f"{case.case_id}.json", record)
            if fault["detected"] is True:
                fault_evidence[case.case_id] = (fault_id,)
            if outcome == "passed":
                evidence[case.case_id] = (artifact_id,)
    build_after = source_build_fingerprint(root)
    driver_after = benchmark_driver_fingerprint(root, driver)
    if build_after != build_before or driver_after != driver_before:
        raise RuntimeError("Qualification source changed during the measured campaign.")
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    implementation, scientific = nuclear_fuel_closure_matrices(
        evidence, fault_adequacy_evidence=fault_evidence
    )
    source_receipt = {
        "build_before": build_before,
        "build_after": build_after,
        "driver_before": driver_before,
        "driver_after": driver_after,
        "environment": environment,
        "pid": os.getpid(),
        "wall_s": time.perf_counter() - campaign_start,
        "peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
        "capture": "before-and-after-actual-campaign",
    }
    if artifact_directory is not None:
        _write_json(artifact_directory / "source.json", source_receipt)
    return {
        "source_receipt": source_receipt,
        "campaign_designs": (design,),
        "cases": records,
        "supplemental_cases": supplemental,
        "artifact_directory": str(artifact_directory)
        if artifact_directory is not None
        else None,
        "successful": all(
            record["outcome"] == "passed"
            and _nested_record(record, "fault_adequacy")["detected"] is True
            for record in records
        )
        and all(record["outcome"] == "passed" for record in supplemental),
        "implementation_closure": implementation.to_record(),
        "scientific_release_closure": scientific.to_record(),
        "release_status": "candidate-unreleased",
        "external_prerequisites": (
            "NEA-1664/03-actual-originator-agreement",
            "independent-locked-evaluation",
        ),
        "code_to_code_cases": (),
        "experimental_validation_cases": (),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short smoke design: coarse refinements and truncated histories.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=tuple(case.case_id for case in QUALIFICATION_CASES),
        metavar="CASE",
        help="Registered cases to execute, in campaign order (default: all): "
        + ", ".join(case.case_id for case in QUALIFICATION_CASES),
    )
    parser.add_argument("--artifact-directory", type=Path)
    parser.add_argument("--input-reports", nargs="+", type=Path)
    arguments = parser.parse_args()
    if arguments.input_reports is not None:
        if (
            arguments.smoke
            or arguments.cases is not None
            or arguments.artifact_directory is not None
        ):
            parser.error(
                "--input-reports cannot be combined with physics execution selectors."
            )
        report = aggregate_reports(tuple(arguments.input_reports))
    else:
        report = qualify(
            smoke=arguments.smoke,
            cases=None if arguments.cases is None else tuple(arguments.cases),
            artifact_directory=(
                arguments.artifact_directory
                if arguments.artifact_directory is not None or arguments.output is None
                else arguments.output.with_suffix(".artifacts")
            ),
        )
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if report["successful"] is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
