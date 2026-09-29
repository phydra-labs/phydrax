#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Qualify numerical interoperability by its scenario matrix.

A scenario is identified by its coordinates: capability, interaction, concrete
providers, geometry, execution, and derivative route. It belongs to one family
of the numerical-interoperability plan and names existing pytest nodes as
positive evidence and as negative boundaries (a refusal or a visible failure
that must hold). A node reference is ``path::test`` (every parametrization) or
``path::test[id]``.

The runner collects the referenced test files once, runs the resolved nodes
in-process (single device) or in a fresh interpreter whose startup environment
forces host devices (multi-device execution), and binds each scenario into one
causal record chain: support tuple, zero-unqualified-node criterion, campaign
start, raw observation, campaign observation, and qualification evidence.
Records use a logical clock and content addresses; measured node durations and
session wall times are reported beside them and never content-addressed.

A scenario passes only when every resolved node passed. A failed node or a
reference that no longer resolves fails it. A skipped node, a collection
failure (a failed collector covering the reference, or an aborted collection
that may never have reached it), a fresh-interpreter route that missed its
``--route-timeout`` deadline, or an unavailable optional provider leaves it
inconclusive, never passed. Scaling scenarios reference rows of the separately
recorded ``benchmarks/numerical_interoperability.py`` campaign; an unregistered
row fails the scenario.

Usage, from the repository root::

    uv run --extra tests python -m tools.numerical_interoperability_qualification --list
    uv run --extra tests python -m tools.numerical_interoperability_qualification \\
        --family derivatives --workers 8 --output report.json

``--scenario`` and ``--family`` are repeatable and select their union; without
either every scenario runs. Unselected scenarios make the report outcome
``inconclusive``. The report is printed when ``--output`` is omitted. The exit
status is nonzero when any selected scenario failed or was inconclusive.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax

from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment, source_build_fingerprint
from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationCriterion,
    QualificationEvidence,
    QualificationRuntimeIdentity,
    SupportTuple,
    validate_qualification_causality,
)
from tools._pytest_outcomes import (
    collect_pytest_nodes,
    NodeOutcome,
    PytestCollection,
    PytestRun,
    run_pytest,
    run_pytest_subprocess,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPABILITY = "numerical-interoperability.qualification-scenario"
APPROVAL_ID = "numerical-interoperability-plan"
REVIEWER_ID = "numerical-interoperability-runner"
BENCHMARK_DRIVER = "benchmarks/numerical_interoperability.py"

# Logical clock: the records carry causal order, never wall time.
CRITERION_TICK = 1
STARTED_TICK = 2
OBSERVED_TICK = 3
ISSUED_TICK = 4
EXPIRES_TICK = 5

FAMILIES: Mapping[str, str] = {
    "semantic-attachments": "Semantic attachments",
    "queries-and-sides": "Queries and sides",
    "spatial-transmission": "Spatial transmission",
    "exterior-operator-2d": "2-D exterior operator",
    "sem-vem-bem": "SEM-VEM-BEM",
    "method-substitution": "Method substitution",
    "derivatives": "Derivatives",
    "inventory-transfer": "Inventory transfer",
    "temporal-coupling": "Temporal coupling",
    "block-dae-views": "Block/DAE views",
    "learning": "Learning",
    "measurement-inverse": "Measurement/inverse",
    "control-uq-rom": "Control/UQ/ROM",
    "lifecycle": "Lifecycle",
    "junctions-embedded": "Junctions/embedded",
    "external": "External",
    "execution-wrappers": "Execution wrappers",
    "scaling": "Scaling",
}
# Scaling scenarios qualify resource admission; their performance is benchmark evidence.
_EVIDENCE_KIND = {"scaling": "operational"}

_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_REFERENCE = re.compile(r"tests/[A-Za-z0-9_/]+\.py::test_[A-Za-z0-9_]+(?:\[[^\]]+\])?")

type ScenarioRole = Literal["positive", "negative"]


@dataclass(frozen=True)
class OptionalProvider:
    """An optional runtime a scenario's evidence needs; absence is inconclusive."""

    name: str
    kind: Literal["python-module", "executable"]

    def available(self) -> bool:
        if self.kind == "python-module":
            return importlib.util.find_spec(self.name) is not None
        return shutil.which(self.name) is not None


@dataclass(frozen=True)
class Scenario:
    """One capability x interaction x providers x geometry x execution x derivative."""

    family: str
    capability: str
    interaction: str
    providers: tuple[str, ...]
    geometry: str
    execution: str
    derivative: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]
    optional_providers: tuple[OptionalProvider, ...] = ()
    host_devices: int = 1
    benchmark_rows: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown scenario family {self.family!r}.")
        tokens = (
            self.capability,
            self.interaction,
            *self.providers,
            self.geometry,
            self.execution,
            self.derivative,
            *self.benchmark_rows,
        )
        invalid = [token for token in tokens if _TOKEN.fullmatch(token) is None]
        if invalid or not self.providers:
            raise ValueError(f"Scenario coordinates must be kebab-case: {invalid}.")
        if len(set(self.providers)) != len(self.providers):
            raise ValueError("Scenario providers must be unique.")
        if not self.positive or not self.negative:
            raise ValueError(
                "A scenario needs positive evidence and a negative boundary."
            )
        references = (*self.positive, *self.negative)
        malformed = [ref for ref in references if _REFERENCE.fullmatch(ref) is None]
        if malformed:
            raise ValueError(f"Malformed pytest node references: {malformed}.")
        if len(set(references)) != len(references):
            raise ValueError("A node reference may appear once per scenario.")
        if isinstance(self.host_devices, bool) or self.host_devices < 1:
            raise ValueError("host_devices must be a positive integer.")
        forced = f"forced-host-devices-{self.host_devices}"
        if (self.host_devices > 1) != (self.execution == forced):
            raise ValueError(
                "Multi-device scenarios declare execution "
                "'forced-host-devices-<count>' matching host_devices."
            )

    @property
    def scenario_id(self) -> str:
        return "/".join(
            (
                self.capability,
                self.interaction,
                "+".join(self.providers),
                self.geometry,
                self.execution,
                self.derivative,
            )
        )

    @property
    def references(self) -> tuple[tuple[ScenarioRole, str], ...]:
        positive = tuple(("positive", reference) for reference in self.positive)
        negative = tuple(("negative", reference) for reference in self.negative)
        return (*positive, *negative)

    def coordinates(self) -> dict[str, str]:
        return {
            "capability": self.capability,
            "interaction": self.interaction,
            "providers": "+".join(self.providers),
            "geometry": self.geometry,
            "execution": self.execution,
            "derivative": self.derivative,
        }


_BINDINGS = "tests/unit/meshing/test_interface_bindings.py::"
_OWNERSHIP = "tests/unit/solver/coupling/test_constraint_ownership.py::"
_CONTRACTS = "tests/unit/solver/coupling/test_spatial_contracts.py::"
_QUERIES = "tests/unit/discretization/test_prepared_field_queries.py::"
_SIDES = "tests/unit/discretization/test_side_actions.py::"
_FLUX = "tests/unit/equations/test_finite_element_pointwise_flux.py::"
_STABILITY = "tests/unit/discretization/test_trace_stability.py::"
_TRACE_SPACES = "tests/unit/discretization/test_boundary_trace_spaces.py::"
_TRANSMISSION = "tests/unit/solver/coupling/test_transmission.py::"
_NITSCHE = "tests/unit/solver/coupling/test_nitsche.py::"
_LAWS = "tests/unit/solver/coupling/test_interface_laws.py::"
_GALERKIN = "tests/unit/operators/test_scalar_galerkin2d.py::"
_FLAGSHIP = "tests/integration/test_sem_vem_bem_transmission.py::"
_FEM_BEM_COMPONENTS = "tests/unit/solver/coupling/test_fem_bem_components.py::"
_HANDOFF = "tests/unit/solver/test_pic_field_handoff.py::"
_INVERSE = "tests/integration/test_coupled_inverse_workflow.py::"
_PARAMETERS = "tests/unit/solver/coupling/test_parameter_observation_bindings.py::"
_SOLVE_DERIVATIVE = "tests/unit/linalg/test_linear_solve_derivative_contract.py::"
_VEM_COMPILER = "tests/unit/equations/test_virtual_element_compiler.py::"
_CONDENSATION = "tests/unit/solver/coupling/test_condensation.py::"
_FEM_BEM_SCALAR = "tests/unit/solver/test_fem_bem_scalar.py::"
_FEM_BEM_VECTOR = "tests/unit/solver/test_fem_bem_vector.py::"
_FUNCTIONALS = "tests/unit/solver/coupling/test_measurement_functionals.py::"
_PARTITIONED = "tests/unit/solver/test_partitioned_coupling.py::"
_WAVEFORM = "tests/unit/solver/test_partitioned_coupling_waveform.py::"
_MIXED_TIME = "tests/integration/test_mixed_method_time_coupling.py::"
_PARTICIPANTS = "tests/unit/solver/coupling/test_method_participants.py::"
_CONTINUATION = "tests/unit/solver/test_dae_continuation.py::"
_TRANSIENT = "tests/unit/solver/coupling/test_transient.py::"
_NAMED_BLOCKS = "tests/unit/linalg/test_named_blocks.py::"
_DAE_ADAPTER = "tests/unit/solver/test_dae_coordinate_adapter.py::"
_LEARNED = "tests/integration/test_coupled_learned_components.py::"
_LEARNED_LAW = "tests/integration/test_coupled_learned_interface_law.py::"
_HYBRID = "tests/integration/test_hybrid_numerical_field.py::"
_COMPARISON = "tests/unit/measurement/test_measurement_comparison.py::"
_LIKELIHOOD = "tests/unit/uq/test_measurement_likelihood.py::"
_POSTERIOR = "tests/unit/uq/test_posterior_terms.py::"
_CONTROL = "tests/integration/test_coupled_control.py::"
_STATE_SPACE = "tests/integration/test_coupled_state_space.py::"
_ROM = "tests/integration/test_coupled_rom.py::"
_EXAMPLE = (
    "tests/integration/test_coupled_examples.py::"
    "test_coupled_example_passes_its_own_reference_checks"
)
_REBIND = "tests/integration/test_adaptive_fe_fv_rebind.py::"
_RESTART = "tests/integration/test_coupled_restart.py::"
_COMPOSITION = "tests/unit/lifecycle/test_composition_rebind.py::"
_TRAINING_KERNEL = "tests/unit/test_training_kernel.py::"
_DISTRIBUTION = "tests/unit/meshing/test_distribution.py::"
_WORKSET = "tests/unit/test_execution_workset.py::"
_FOAM = "tests/integration/test_foam_composition_rebind.py::"
_FMI_COUPLING = "tests/interchange/test_fmi_composition.py::"
_FMI = "tests/interchange/test_fmi.py::"
_PIC_CAPABILITIES = "tests/unit/solver/test_distributed_pic_capabilities.py::"
_DISTRIBUTED_PIC = "tests/unit/solver/test_distributed_pic.py::"
_LANES = "tests/unit/solver/coupling/test_execution_contracts.py::"

_MISSING_WITNESS = (
    _BINDINGS + "test_interface_declarations_without_exact_witness_are_refused"
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "semantic-attachments",
        "interface-binding",
        "two-sided-mesh-interface",
        ("brep-projection", "mesh-assembly"),
        "brep-plate-two-rectangles-triangulated",
        "single-device",
        "none",
        positive=(
            _BINDINGS
            + "test_two_sided_mesh_binding_orders_sides_by_authoritative_normal",
            _BINDINGS + "test_ownership_redistribution_keeps_the_bound_interface",
            _BINDINGS + "test_analytic_paired_support_remains_its_own_authority",
        ),
        negative=(
            _MISSING_WITNESS + "[other-source-revision]",
            _MISSING_WITNESS + "[shape-without-witness]",
            _MISSING_WITNESS + "[reversed-normals]",
            _MISSING_WITNESS + "[same-side]",
            _MISSING_WITNESS + "[uncertified-witness]",
            _MISSING_WITNESS + "[foreign-patch]",
            _MISSING_WITNESS + "[undeclared-geometry]",
            _BINDINGS
            + "test_moved_endpoint_invalidates_binding_but_not_interface_identity",
        ),
    ),
    Scenario(
        "semantic-attachments",
        "coupled-field-binding",
        "component-field-identity",
        ("finite-element", "virtual-element"),
        "unit-square-pair-cut-x1",
        "single-device",
        "none",
        positive=(
            _CONTRACTS + "test_plan_orders_components_laws_and_bindings_canonically",
            _CONTRACTS + "test_prepared_layout_and_block_operator_identities",
        ),
        negative=(
            _OWNERSHIP + "test_binding_field_identities_must_match_the_components",
            _OWNERSHIP + "test_transmission_sides_must_follow_the_binding_orientation",
            _CONTRACTS + "test_plan_refuses_duplicate_and_colliding_identities",
            _CONTRACTS + "test_plan_refuses_undeclared_and_unused_bindings",
            _CONTRACTS
            + "test_plan_refuses_parameter_and_observation_bindings_of_unknown_components",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "value-transpose-adjoint",
        ("finite-element-lagrange",),
        "triangles-and-quadrilaterals-2d",
        "single-device",
        "none",
        positive=(
            _QUERIES
            + "test_prepared_query_reuses_its_route_across_coefficient_refreshes",
            _QUERIES
            + "test_query_transpose_is_the_coordinate_dual_and_adjoint_uses_declared_pairings",
            _QUERIES + "test_one_sided_queries_bind_the_requested_trace_side",
            _SIDES + "test_facet_rule_is_exact_through_its_declared_degree",
            _SIDES
            + "test_trace_action_separates_pullback_measure_load_and_hilbert_adjoint",
            _SIDES + "test_contracted_routes_trace_vector_components_against_the_normal",
            _SIDES
            + "test_fe_exterior_value_trace_is_exact_with_physical_measure_and_normals",
            _SIDES + "test_fe_interior_traces_share_sites_continuous_h1_and_jumping_l2",
            _SIDES + "test_fe_trace_load_pairing_and_mass_hilbert_adjoint",
            _SIDES
            + "test_fe_vector_normal_and_tangential_traces_contract_the_outward_normal",
        ),
        negative=(
            _QUERIES + "test_masked_query_admits_valid_points_and_keeps_every_status",
            _QUERIES + "test_query_refuses_a_refreshed_geometry_revision",
            _SIDES + "test_facet_rule_refuses_unsupported_families_and_evaluates_points",
            _SIDES + "test_side_records_refuse_inconsistent_declarations",
            _SIDES + "test_fe_side_traces_and_reactions_refuse_unsupported_requests",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "pointwise-conormal-flux",
        "constitutive-flux-trace",
        ("finite-element-lagrange",),
        "triangles-2d",
        "single-device",
        "none",
        positive=(
            _FLUX
            + "test_tensor_flux_equals_the_analytic_conormal_flux_at_the_trace_sites",
            _FLUX + "test_owner_and_neighbor_fluxes_of_an_interior_facet_cancel",
            _FLUX
            + "test_flux_transpose_loads_rows_whose_pairing_is_the_analytic_flux_moment",
            _FLUX + "test_p1_constants_equal_the_analytic_trace_inverse_bound",
            _FLUX + "test_p2_constants_scale_with_diffusivity_over_mesh_size",
            _SIDES
            + "test_fe_reaction_flux_matches_the_conormal_integral_on_dirichlet_rows",
        ),
        negative=(
            _FLUX + "test_forms_without_a_declared_flux_law_are_refused",
            _FLUX
            + "test_callable_diffusivity_flux_publishes_no_degree_and_is_not_certified",
            _STABILITY + "test_energy_kernel_larger_than_the_constants_is_refused",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "projected-versus-exact-trace",
        ("virtual-element",),
        "polygons-2d",
        "single-device",
        "none",
        positive=(
            _QUERIES
            + "test_virtual_element_projected_query_reuses_its_route_across_refreshes",
            _SIDES
            + "test_virtual_element_value_trace_is_exact_and_distinct_from_the_projection",
            _SIDES
            + "test_virtual_element_moment_traces_are_outward_traces_of_linear_fields",
            _SIDES
            + "test_virtual_element_reaction_flux_is_the_edge_integral_of_the_physical_flux",
            _SIDES
            + "test_virtual_element_projected_flux_uses_the_side_cell_tensor_diffusivity",
            _SIDES + "test_virtual_element_boundary_impositions_follow_the_form_order",
        ),
        negative=(
            _QUERIES + "test_virtual_element_channels_are_labeled_and_family_specific",
            _SIDES + "test_virtual_element_fluxes_refuse_foreign_or_undefined_sides",
            _SIDES + "test_polygon_side_traces_refuse_undefined_traces_and_sides",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "value-transpose-adjoint",
        ("explicit-polygon-h1",),
        "polygons-2d",
        "single-device",
        "none",
        positive=(
            _QUERIES + "test_explicit_polygon_h1_query_is_exact_for_affine_fields",
            _SIDES
            + "test_explicit_polygon_h1_edge_trace_is_linear_and_pulls_back_through_the_measure",
            _SIDES
            + "test_explicit_polygon_h1_interior_sides_share_sites_and_oppose_normals",
        ),
        negative=(
            _QUERIES
            + "test_explicit_polygon_h1_query_masks_outside_points_and_fan_edge_gradients",
            _SIDES + "test_polygon_side_traces_refuse_undefined_traces_and_sides",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "value-transpose-adjoint",
        ("isogeometric-nurbs",),
        "curved-and-box-patches-2d",
        "single-device",
        "none",
        positive=(
            _QUERIES
            + "test_isogeometric_query_reproduces_physical_linear_fields_on_a_curved_patch",
            _QUERIES
            + "test_isogeometric_query_reproduces_tensor_quadratics_on_a_box_patch",
            _QUERIES
            + "test_isogeometric_query_transpose_and_adjoint_use_declared_pairings",
            _QUERIES + "test_isogeometric_query_binds_sides_on_a_c0_knot_line",
            _SIDES
            + "test_isogeometric_exterior_traces_have_analytic_lengths_normals_and_values",
            _SIDES + "test_isogeometric_vector_traces_contract_the_outward_normal",
            _SIDES
            + "test_isogeometric_interior_knot_faces_share_sites_and_oppose_normals",
        ),
        negative=(
            _QUERIES
            + "test_isogeometric_query_refuses_undeclared_supports_orders_and_revisions",
            _SIDES
            + "test_isogeometric_side_traces_refuse_undefined_sides_and_foreign_domains",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "value-transpose-sbp-adjoint",
        ("finite-difference-sbp", "bspline-reconstruction"),
        "tensor-grids",
        "single-device",
        "none",
        positive=(
            _QUERIES + "test_fd_bspline_query_reproduces_tensor_cubics_across_refreshes",
            _QUERIES + "test_fd_query_transpose_is_dual_and_adjoint_uses_the_sbp_norm",
            _SIDES + "test_fd_sbp_boundary_norm_integrates_face_polynomials_to_its_order",
            _SIDES + "test_fd_trace_load_duality_and_sbp_hilbert_adjoint",
        ),
        negative=(
            _SIDES + "test_fd_side_traces_refuse_rules_sides_periodic_faces_and_fluxes",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "trigonometric-value-and-face-trace",
        ("global-spectral",),
        "periodic-and-bounded-boxes",
        "single-device",
        "none",
        positive=(
            _QUERIES + "test_spectral_query_is_exact_for_trigonometric_polynomial_fields",
            _SIDES
            + "test_spectral_face_traces_are_spectrally_accurate_with_native_measures",
            _SIDES + "test_spectral_rule_traces_pull_loads_back_to_complex_modes",
        ),
        negative=(
            _SIDES
            + "test_spectral_side_traces_refuse_periodic_sine_and_undefined_requests",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query-side-trace",
        "reconstructed-face-state",
        ("finite-volume-k-exact", "finite-volume-muscl", "finite-volume-weno-z"),
        "structured-grids-triangles-tetrahedra",
        "single-device",
        "none",
        positive=(
            _QUERIES
            + "test_fv_triangle_k_exact_query_reproduces_quadratics_and_their_derivatives",
            _SIDES + "test_fv_face_measures_normals_and_sides_follow_the_cell_geometry",
            _SIDES
            + "test_fv_linear_reconstruction_face_states_are_exact_polynomial_traces",
            _SIDES
            + "test_fv_structured_unlimited_muscl_face_state_is_the_linear_face_average",
            _SIDES
            + "test_fv_face_state_routes_are_local_with_exact_transposes_and_volume_adjoints",
            _SIDES + "test_fv_structured_cell_average_load_injects_the_exposed_face_area",
            _SIDES
            + "test_fv_limited_muscl_and_structured_weno_face_states_match_their_owners",
        ),
        negative=(
            _QUERIES
            + "test_fv_triangle_limited_muscl_query_is_nonlinear_with_a_local_linearization",
            _QUERIES
            + "test_nonlinear_query_exposes_a_linearization_instead_of_a_transpose",
            _SIDES + "test_fv_weno_z_face_states_publish_a_local_linearization",
            _SIDES
            + "test_fv_cell_average_and_face_state_descriptors_are_distinct_identities",
            _SIDES
            + "test_fv_side_traces_refuse_undefined_sides_domains_and_reconstructions",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "field-query",
        "masked-query-quadrature-adjoint",
        ("point-cloud-reconstruction",),
        "scattered-points-2d",
        "single-device",
        "none",
        positive=(
            _QUERIES + "test_point_cloud_query_adjoint_uses_the_cloud_quadrature_pairing",
        ),
        negative=(
            _QUERIES
            + "test_point_cloud_masked_query_keeps_partial_support_and_conditioning_evidence",
        ),
    ),
    Scenario(
        "queries-and-sides",
        "boundary-trace-space",
        "cauchy-data-versus-surface-currents",
        (
            "galerkin-2d-p1-dp0",
            "calderon-3d-p1-dp0",
            "rwg",
            "buffa-christiansen",
        ),
        "closed-polygons-and-triangulated-surfaces",
        "single-device",
        "none",
        positive=(
            _TRACE_SPACES
            + "test_polygon_cauchy_traces_pair_by_arc_length_and_divergence_theorem",
            _TRACE_SPACES
            + "test_closed_surface_cauchy_traces_pair_by_area_and_divergence_theorem",
            _TRACE_SPACES
            + "test_closed_surface_cauchy_traces_keep_the_complex_kernel_dtype",
            _TRACE_SPACES
            + "test_buffa_christiansen_dual_pairs_through_its_barycentric_currents",
        ),
        negative=(
            _TRACE_SPACES
            + "test_rwg_current_space_pairs_by_area_and_is_a_distinct_representation",
            _TRACE_SPACES
            + "test_polygon_traversal_reversal_keeps_the_outward_neumann_trace",
            _TRACE_SPACES
            + "test_closed_surface_reversed_winding_keeps_the_outward_neumann_trace",
            _TRACE_SPACES
            + "test_rwg_orientation_reversal_negates_currents_and_moves_revision",
            _TRACE_SPACES + "test_polygon_revision_follows_moved_vertices",
        ),
    ),
    Scenario(
        "spatial-transmission",
        "scalar-transmission",
        "matching-elimination",
        ("finite-element-p2", "virtual-element-k2"),
        "unit-square-pair-cut-x1",
        "single-device",
        "none",
        positive=(
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-fe-p2-matching]",
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-vem-k2-matching]",
            _TRANSMISSION + "test_refinement_converges_at_second_order[fe-fe-matching]",
            _TRANSMISSION + "test_refinement_converges_at_second_order[fe-vem-matching]",
            _CONTRACTS + "test_single_component_plan_reproduces_the_native_solve",
            _OWNERSHIP
            + "test_dirichlet_lifts_of_both_owners_are_composed_once[matching]",
            _OWNERSHIP
            + "test_gauged_floating_problem_matches_the_field_up_to_a_constant",
        ),
        negative=(
            _OWNERSHIP + "test_owner_boundary_law_on_the_interface_is_refused",
            _OWNERSHIP + "test_dirichlet_data_on_the_whole_interface_is_refused",
            _OWNERSHIP + "test_second_law_on_the_same_interface_is_refused",
            _OWNERSHIP + "test_matching_elimination_of_nonmatching_facets_is_refused",
            _OWNERSHIP + "test_elimination_refuses_charts_that_are_not_row_selections",
            _OWNERSHIP + "test_elimination_refuses_strongly_imposed_rows",
            _OWNERSHIP + "test_coupled_kernel_without_gauge_is_refused",
            _OWNERSHIP + "test_native_solve_failure_propagates_without_acceptance",
        ),
    ),
    Scenario(
        "spatial-transmission",
        "scalar-transmission",
        "mortar-multiplier",
        ("finite-element-p2", "virtual-element-k2", "explicit-polygon-h1"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-fe-p2-side-trace]",
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-vem-k2-side-trace]",
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-fe-p2-discontinuous-p1]",
            _TRANSMISSION
            + "test_quadratic_transmission_is_reproduced_exactly[fe-vem-k2-discontinuous-p1]",
            _TRANSMISSION + "test_refinement_converges_at_second_order[fe-fe-side-trace]",
            _TRANSMISSION
            + "test_refinement_converges_at_second_order[fe-vem-side-trace]",
            _TRANSMISSION
            + "test_polygon_region_reproduces_linear_field_across_nonmatching_mortar",
            _CONTRACTS + "test_interface_quadrature_integrates_trace_products_exactly",
            _CONTRACTS + "test_interface_resampling_pullback_is_the_exact_transpose",
            _OWNERSHIP
            + "test_dirichlet_lifts_of_both_owners_are_composed_once[mortar-side-trace]",
        ),
        negative=(
            _OWNERSHIP + "test_rank_deficient_multiplier_space_is_refused",
            _OWNERSHIP + "test_partial_interface_coverage_is_refused",
            _OWNERSHIP + "test_regions_on_the_same_side_of_the_interface_are_refused",
            _OWNERSHIP + "test_reduced_operator_addressed_as_a_full_field_is_refused",
            _OWNERSHIP + "test_law_blocks_that_do_not_square_are_refused",
        ),
    ),
    Scenario(
        "spatial-transmission",
        "scalar-transmission",
        "nitsche-penalty",
        ("finite-element-p1-p2", "virtual-element-k1"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _NITSCHE
            + "test_diffusivity_jump_is_reproduced_exactly_on_a_nonmatching_interface",
            _NITSCHE + "test_quadratic_field_is_reproduced_exactly_by_p2",
            _NITSCHE + "test_nonmatching_refinement_converges_at_optimal_order",
            _NITSCHE + "test_penalty_is_the_factor_times_the_certified_p1_constants",
            _NITSCHE + "test_certified_variants_assemble_coercive_operators",
            _NITSCHE + "test_one_sided_nitsche_couples_a_virtual_element_side_exactly",
            _STABILITY + "test_interval_constant_is_kappa_over_h_and_padding_is_inert",
        ),
        negative=(
            _NITSCHE
            + "test_symmetric_penalty_at_or_below_the_certified_bound_is_refused",
            _NITSCHE + "test_owner_without_a_declared_flux_law_is_refused",
        ),
    ),
    Scenario(
        "spatial-transmission",
        "interface-physical-law",
        "conductance-radiation-port-transfer",
        ("finite-element-p1", "virtual-element-k1"),
        "cut-plates-and-port-interval",
        "single-device",
        "none",
        positive=(
            _LAWS + "test_conductance_flux_converges_to_the_contact_jump",
            _LAWS + "test_gap_radiation_flux_solves_through_newton",
            _LAWS + "test_integral_port_closes_a_field_with_a_resistor",
            _LAWS + "test_field_transfer_exchange_converges_between_nonmatching_meshes",
        ),
        negative=(
            _LAWS + "test_flux_law_refuses_a_misdeclared_affine_flux",
            _LAWS + "test_integral_port_refuses_undeclared_connector_semantics",
            _LAWS + "test_field_transfer_refuses_unbound_or_nonconservative_transfers",
        ),
    ),
    Scenario(
        "exterior-operator-2d",
        "exterior-laplace-galerkin",
        "layer-potentials-and-exterior-relation",
        ("galerkin-2d-p1-dp0",),
        "closed-straight-panel-polygons",
        "single-device",
        "none",
        positive=(
            _GALERKIN + "test_straight_panel_closed_forms_match_hand_derived_integrals",
            _GALERKIN + "test_shared_endpoint_entries_match_mpmath",
            _GALERKIN + "test_separated_and_near_pairs_match_high_order_host_quadrature",
            _GALERKIN + "test_constant_density_double_layer_has_the_declared_jumps",
            _GALERKIN
            + "test_blocked_actions_match_materialization_transposes_and_adjoints",
            _GALERKIN + "test_orientation_reversal_preserves_physical_operators",
            _GALERKIN + "test_green_identities_hold_with_second_order_consistency",
            _GALERKIN
            + "test_bordered_exterior_solve_recovers_decaying_and_shifted_fields",
            _GALERKIN + "test_trace_projection_is_a_declared_l2_projection",
            _GALERKIN + "test_report_publishes_pair_evidence_and_exact_support",
            _CONTRACTS
            + "test_galerkin_boundary_component_publishes_the_exterior_relation",
        ),
        negative=(
            _GALERKIN + "test_curve_refuses_non_simple_or_degenerate_polygons",
            _GALERKIN + "test_curve_requires_an_explicit_canonical_source_identity",
            _GALERKIN + "test_logarithmic_cauchy_data_violate_bounded_compatibility",
            _GALERKIN + "test_exterior_formulation_refuses_undeclared_configurations",
            _GALERKIN + "test_resource_and_quadrature_limits_refuse_before_use",
        ),
    ),
    Scenario(
        "sem-vem-bem",
        "boundary-integral-transmission",
        "mortar-and-bordered-exterior-relation",
        ("spectral-element-gll", "virtual-element", "galerkin-2d-p1-dp0"),
        "square-annulus-and-unbounded-exterior",
        "single-device",
        "none",
        positive=(
            _FLAGSHIP + "test_flagship_is_one_accepted_coupled_solve",
            _FLAGSHIP + "test_hole_dirichlet_data_is_imposed_strongly",
            _FLAGSHIP + "test_fields_and_sensors_match_the_independent_reference",
            _FLAGSHIP + "test_exterior_relation_compatibility_and_far_field",
            _FLAGSHIP + "test_flux_balance_matches_the_owner_reaction",
            _FLAGSHIP + "test_complete_block_transpose_is_exact",
            _FLAGSHIP + "test_h_refinement_converges_at_second_order",
            _FLAGSHIP + "test_boundary_panels_are_the_floor_only_when_coarse",
            _FLAGSHIP + "test_heterogeneous_coefficient_case",
            _FLAGSHIP + "test_matching_elimination_variant",
        ),
        negative=(
            _FLAGSHIP + "test_unsupported_declarations_are_refused_at_preparation",
            _FLAGSHIP + "test_failed_native_solve_is_visible",
            _FLAGSHIP + "test_dense_materialization_excess_is_refused_at_solve",
        ),
    ),
    Scenario(
        "sem-vem-bem",
        "boundary-integral-transmission",
        "far-field-and-gauge",
        ("finite-element", "galerkin-2d-p1-dp0"),
        "square-box-and-unbounded-exterior",
        "single-device",
        "none",
        positive=(
            _TRANSMISSION + "test_gauged_kernel_pair_is_the_coupled_constant_mode",
            _TRANSMISSION
            + "test_decaying_exterior_gates_the_far_field_constant[within-tolerance]",
        ),
        negative=(
            _TRANSMISSION
            + "test_decaying_exterior_gates_the_far_field_constant[beyond-tolerance]",
            _TRANSMISSION + "test_far_field_declaration_refuses_incomplete_modes",
            _TRANSMISSION + "test_kernel_through_law_unknowns_is_refused_without_a_gauge",
            _TRANSMISSION + "test_boundary_law_refuses_inconsistent_declarations",
        ),
    ),
    Scenario(
        "method-substitution",
        "boundary-integral-transmission",
        "volume-method-substitution",
        (
            "spectral-element-gll",
            "finite-element",
            "explicit-polygon-h1",
            "isogeometric-nurbs",
            "virtual-element",
        ),
        "square-box-and-unbounded-exterior",
        "single-device",
        "none",
        positive=(
            _TRANSMISSION + "test_boundary_law_couples_every_volume_method_unchanged",
            _FLAGSHIP + "test_vem_to_fe_substitution_keeps_declarations_and_observation",
        ),
        negative=(
            _TRANSMISSION
            + "test_boundary_law_refuses_inconsistent_declarations[non-trace-volume-component]",
            _NITSCHE + "test_virtual_element_side_with_flux_weight_is_refused",
        ),
    ),
    Scenario(
        "method-substitution",
        "fem-bem-product-component",
        "native-product-in-coupled-plan",
        ("fem-bem-scalar-3d", "fem-bem-elasticity-3d"),
        "tetrahedral-bipyramid",
        "single-device",
        "none",
        positive=(
            _FEM_BEM_COMPONENTS
            + "test_scalar_component_publishes_the_product_blocks_and_identity",
            _FEM_BEM_COMPONENTS
            + "test_scalar_single_component_plan_reproduces_the_product_solve",
            _FEM_BEM_COMPONENTS
            + "test_scalar_published_operator_is_the_product_operator_with_exact_transpose",
            _FEM_BEM_COMPONENTS
            + "test_elasticity_component_publishes_the_product_blocks_and_identity",
            _FEM_BEM_COMPONENTS
            + "test_elasticity_single_component_plan_reproduces_the_product_solve",
            _FEM_BEM_COMPONENTS
            + "test_elasticity_published_operator_is_the_product_operator_with_exact_transpose",
        ),
        negative=(
            _FEM_BEM_COMPONENTS + "test_scalar_nonpositive_conductivity_is_not_accepted",
            _FEM_BEM_COMPONENTS + "test_components_accept_only_their_prepared_product",
            _FEM_BEM_COMPONENTS + "test_components_refuse_another_argument_kind",
        ),
    ),
    Scenario(
        "method-substitution",
        "pic-field-solver-substitution",
        "field-state-handoff",
        ("yee-cochain", "psatd-spectral"),
        "periodic-grid-3d",
        "single-device",
        "none",
        positive=(
            _HANDOFF + "test_handoff_places_yee_components_at_staggered_psatd_positions",
            _HANDOFF + "test_prescribed_magnetic_field_lands_on_its_face_centers",
            _HANDOFF + "test_particles_gather_the_same_fields_after_handoff",
            _HANDOFF + "test_psatd_continuation_keeps_gauss_and_a_continuous_ledger",
            _HANDOFF + "test_plasma_modes_continue_across_the_handoff",
            _HANDOFF + "test_spectral_state_hands_back_to_the_cochain_solver",
        ),
        negative=(
            _HANDOFF + "test_inadmissible_handoffs_are_refused",
            _HANDOFF + "test_state_of_another_solver_is_refused",
            _HANDOFF + "test_constraint_violating_state_is_not_handed_off_successfully",
            _HANDOFF + "test_non_neutral_periodic_state_fails_gauss_in_both_directions",
        ),
    ),
    Scenario(
        "derivatives",
        "coupled-solution-map-derivative",
        "parameter-to-observation",
        ("finite-element-p1", "virtual-element"),
        "two-region-plate",
        "single-device",
        "implicit-adjoint",
        positive=(
            _INVERSE + "test_misfit_derivatives_match_host_central_differences",
            _INVERSE + "test_rhs_only_heat_flux_derivative_is_the_analytic_sensitivity",
            _PARAMETERS
            + "test_derivative_capability_follows_the_solve_differentiation_mode",
            _SOLVE_DERIVATIVE
            + "test_primal_factor_route_derivatives_are_exact_factored_solves",
            _VEM_COMPILER
            + "test_runtime_dirichlet_solve_derivatives_match_central_differences",
            _VEM_COMPILER
            + "test_runtime_boundary_solve_derivatives_match_central_differences",
            _VEM_COMPILER + "test_runtime_objective_gradient_compiles_under_jit",
        ),
        negative=(
            _INVERSE + "test_failed_primal_is_rejected_and_has_no_derivative",
            _INVERSE + "test_failed_derivative_solve_poisons_only_the_derivative",
            _INVERSE + "test_rhs_only_policy_refuses_conductivity_derivatives",
            _PARAMETERS + "test_a_solver_argument_entering_the_operator_is_refused",
            _PARAMETERS + "test_reprepare_parameters_are_fixed_structure",
        ),
    ),
    Scenario(
        "derivatives",
        "condensed-solution-map-derivative",
        "static-condensation",
        ("finite-element-p2",),
        "unit-square-pair-cut-x1",
        "single-device",
        "implicit-adjoint",
        positive=(
            _CONDENSATION
            + "test_condensed_parameter_derivatives_match_the_uncondensed_solution_map",
        ),
        negative=(
            _CONDENSATION + "test_partial_or_unrolled_pivot_derivatives_are_refused",
            _CONDENSATION + "test_stopped_condensation_refuses_argument_derivatives",
        ),
    ),
    Scenario(
        "derivatives",
        "exterior-galerkin-derivative",
        "dirichlet-data-to-cauchy-data",
        ("galerkin-2d-p1-dp0",),
        "closed-straight-panel-polygons",
        "single-device",
        "exact-linear",
        positive=(
            _GALERKIN + "test_density_actions_have_exact_linear_derivatives",
            _GALERKIN
            + "test_bounded_exterior_dirichlet_derivatives_match_finite_differences",
            _GALERKIN
            + "test_decaying_exterior_derivatives_inside_the_compatibility_regime",
        ),
        negative=(
            _GALERKIN
            + "test_galerkin_refuses_geometry_kernel_quadrature_and_target_derivatives",
            _GALERKIN + "test_rejected_exterior_solves_poison_their_derivatives",
            _GALERKIN + "test_none_differentiation_refuses_dirichlet_derivatives",
        ),
    ),
    Scenario(
        "derivatives",
        "fem-bem-owner-derivative",
        "coefficient-and-data-derivative",
        ("fem-bem-scalar-3d", "fem-bem-elasticity-3d"),
        "tetrahedral-bipyramid",
        "single-device",
        "implicit-solution-map-and-rhs-only",
        positive=(
            _FEM_BEM_SCALAR + "test_solution_map_jvp_matches_central_differences",
            _FEM_BEM_SCALAR + "test_solution_map_vjp_is_the_transpose_of_the_jvp",
            _FEM_BEM_SCALAR
            + "test_compiled_conductivity_gradient_rebinds_without_host_synchronization",
            _FEM_BEM_VECTOR + "test_rhs_only_load_jvp_matches_central_finite_differences",
            _FEM_BEM_VECTOR + "test_rhs_only_load_vjp_is_the_transpose_of_the_jvp",
        ),
        negative=(
            _FEM_BEM_SCALAR
            + "test_rhs_only_admits_data_and_refuses_conductivity_derivatives",
            _FEM_BEM_SCALAR + "test_none_policy_refuses_runtime_derivatives",
            _FEM_BEM_SCALAR
            + "test_algorithmic_differentiation_is_refused_at_preparation",
            _FEM_BEM_SCALAR
            + "test_failed_solve_keeps_primal_status_and_poisons_derivatives",
            _FEM_BEM_SCALAR
            + "test_nonpositive_conductivity_is_not_accepted_or_differentiated",
            _FEM_BEM_SCALAR + "test_prepared_structure_derivatives_are_refused",
            _FEM_BEM_VECTOR
            + "test_unqualified_differentiation_modes_are_refused_at_preparation",
            _FEM_BEM_VECTOR + "test_stopped_route_refuses_load_gradients",
            _FEM_BEM_VECTOR + "test_rejected_solve_poisons_load_derivatives",
            _FEM_BEM_VECTOR
            + "test_derivative_capability_refuses_fixed_prepared_structure",
        ),
    ),
    Scenario(
        "inventory-transfer",
        "inventory-functional",
        "conservative-extensive-exchange",
        ("finite-element-p1", "finite-volume-cell-centered"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _FUNCTIONALS
            + "test_nodal_basis_density_inventory_is_the_exact_field_integral",
            _FUNCTIONALS + "test_flux_moment_inventory_sums_each_component_exactly_once",
            _FUNCTIONALS
            + "test_nonmatching_conservative_projection_balances_the_true_functional",
            _PARTITIONED + "test_component_inventories_keep_separate_ledger_rows",
            _WAVEFORM + "test_integrate_delivers_the_exact_window_amount_and_ledger",
            _MIXED_TIME
            + "test_consumed_interface_heat_balances_the_ledger_and_total_energy",
            _PARTICIPANTS
            + "test_gauss_seidel_fixed_point_window_balances_its_spent_window_amount",
        ),
        negative=(
            _FUNCTIONALS + "test_density_and_extensive_storage_refuse_double_weighting",
            _FUNCTIONALS
            + "test_false_conservation_claim_is_refused_by_the_dual_certificate",
            _FUNCTIONALS
            + "test_probability_functional_cannot_measure_a_whole_window_amount",
            _PARAMETERS + "test_densities_contents_and_values_are_not_identified",
        ),
    ),
    Scenario(
        "temporal-coupling",
        "partitioned-time-coupling",
        "waveform-and-window-integral",
        ("finite-element-p1-backward-euler", "finite-volume-ssprk33"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _MIXED_TIME
            + "test_independent_local_integrators_converge_first_order_in_the_window",
            _MIXED_TIME + "test_rejected_window_rolls_back_carried_key_and_model_state",
            _WAVEFORM + "test_hold_and_sample_end_deliver_the_declared_endpoint_values",
        ),
        negative=(
            _WAVEFORM + "test_undeclared_or_mismatched_temporal_conversions_are_refused",
            _WAVEFORM + "test_temporal_conversion_refuses_inconsistent_declarations",
        ),
    ),
    Scenario(
        "temporal-coupling",
        "native-method-participant",
        "transactional-window",
        ("fixed-step", "imex-conservation", "dae-bdf", "steady-response"),
        "lumped-state-participants",
        "single-device",
        "none",
        positive=(
            _PARTICIPANTS
            + "test_fixed_step_participant_runs_native_substeps_at_the_owner_order",
            _PARTICIPANTS
            + "test_conservation_imex_participant_spends_a_window_amount_at_uniform_rate",
            _PARTICIPANTS
            + "test_implicit_iterates_replay_the_checkpoint_and_advance_it_once",
            _PARTICIPANTS
            + "test_implicit_fixed_point_window_counts_the_work_of_every_iterate",
            _PARTICIPANTS
            + "test_dae_participant_resumes_its_exact_continuation_across_windows",
            _PARTICIPANTS
            + "test_lowering_derives_initial_exchanges_from_participant_checkpoints",
            _PARTICIPANTS
            + "test_adaptive_rollout_evidence_includes_rejected_attempt_work",
            _CONTINUATION
            + "test_adaptive_continuation_keeps_modified_newton_refresh_decisions",
        ),
        negative=(
            _PARTICIPANTS + "test_external_randomness_is_refused_by_replaying_routes",
            _PARTICIPANTS
            + "test_steady_response_is_instantaneous_and_refuses_window_amounts",
            _PARTICIPANTS
            + "test_adaptive_rollout_refuses_a_window_without_a_reliable_error_estimate",
        ),
    ),
    Scenario(
        "temporal-coupling",
        "transient-coupled-problem",
        "index-one-dae-lowering",
        ("finite-element-p1",),
        "unit-square-pair-cut-x1",
        "single-device",
        "none",
        positive=(
            _TRANSIENT
            + "test_bdf_transient_heat_with_moving_lift_converges_to_semidiscrete_reference",
            _TRANSIENT
            + "test_accepted_history_continuation_reproduces_one_adaptive_segment",
        ),
        negative=(
            _TRANSIENT
            + "test_mortar_multiplier_transient_is_refused_with_structural_evidence",
            _TRANSIENT + "test_transient_declarations_are_refused_without_explicit_roles",
        ),
    ),
    Scenario(
        "block-dae-views",
        "named-block-coordinates",
        "path-selection-and-regrouping",
        ("linalg-block-operators",),
        "coordinate-trees",
        "single-device",
        "none",
        positive=(
            _NAMED_BLOCKS
            + "test_nested_path_selection_restricts_embeds_and_pairs_exactly",
            _NAMED_BLOCKS
            + "test_coordinate_blocks_keep_transpose_and_hilbert_adjoint_distinct",
            _NAMED_BLOCKS
            + "test_regrouped_three_field_system_feeds_the_two_by_two_block_factorization",
            _NAMED_BLOCKS
            + "test_mapped_block_operator_uses_explicit_row_and_column_maps",
        ),
        negative=(
            _NAMED_BLOCKS + "test_named_block_refusals",
            _NAMED_BLOCKS
            + "test_named_block_jacobi_certifies_self_adjoint_only_for_paired_transfers",
        ),
    ),
    Scenario(
        "block-dae-views",
        "dae-coordinate-adapter",
        "named-variable-equation-blocks",
        ("reduced-dae-bdf",),
        "heat-flux-cell-and-kinematic-chain",
        "single-device",
        "none",
        positive=(
            _DAE_ADAPTER + "test_named_blocks_preserve_native_scales_and_distinct_roles",
            _DAE_ADAPTER + "test_named_root_setups_equal_native_root_jacobians",
            _DAE_ADAPTER + "test_named_event_setup_borders_the_stage_with_time_and_guard",
            _DAE_ADAPTER + "test_named_block_preconditioned_bdf_matches_native_array_dae",
            _DAE_ADAPTER + "test_named_setup_continues_accepted_bdf_history",
            _TRANSIENT
            + "test_named_block_transient_matches_native_array_dae_with_scales_and_roles",
        ),
        negative=(
            _DAE_ADAPTER
            + "test_unreduced_higher_index_and_unadmitted_systems_are_refused",
        ),
    ),
    Scenario(
        "block-dae-views",
        "coupled-block-solve",
        "static-condensation-and-field-split",
        ("finite-element-p2",),
        "unit-square-pair-cut-x1",
        "single-device",
        "none",
        positive=(
            _CONDENSATION
            + "test_exact_condensation_reproduces_the_uncondensed_direct_solution",
            _CONDENSATION
            + "test_approximate_field_split_preconditions_the_original_coupled_system",
        ),
        negative=(
            _CONDENSATION + "test_rank_deficient_pivot_is_refused_with_its_evidence",
            _CONDENSATION + "test_unqualified_condensations_are_refused",
            _CONDENSATION + "test_condensation_of_a_gauged_kernel_is_refused",
            _CONDENSATION
            + "test_certification_not_preconditioner_success_gates_acceptance",
        ),
    ),
    Scenario(
        "learning",
        "learned-coupled-component",
        "model-and-accelerator-authority",
        ("finite-element-p1", "virtual-element", "neural-field"),
        "two-region-plate",
        "single-device",
        "implicit-solution-map-and-fixed-krylov-work",
        positive=(
            _LEARNED
            + "test_learned_conductivity_gradient_is_the_implicit_solution_map_derivative",
            _LEARNED
            + "test_learned_conductivity_trains_through_accepted_solves_toward_the_analytic_field",
            _LEARNED
            + "test_learned_preconditioner_trains_only_through_fixed_krylov_work",
            _LEARNED
            + "test_learned_preconditioner_accelerates_without_changing_the_accepted_solution",
            _LEARNED
            + "test_initial_guess_proposals_stay_distinct_from_the_accepted_state",
            _LEARNED + "test_each_objective_trains_only_the_authorities_it_admits",
        ),
        negative=(
            _LEARNED
            + "test_a_singular_learned_preconditioner_cannot_produce_an_accepted_solution",
            _LEARNED
            + "test_parameter_binding_refuses_learned_inputs_that_would_change_the_equation",
            _LEARNED
            + "test_solution_map_objective_refuses_authorities_it_does_not_admit",
        ),
    ),
    Scenario(
        "learning",
        "learned-interface-law",
        "monotone-dissipative-contact",
        ("finite-element-p1", "virtual-element", "input-convex-network"),
        "two-region-plate",
        "single-device",
        "state-design-response",
        positive=(
            _LEARNED_LAW + "test_learned_contact_reproduces_the_analytic_nonlinear_plate",
            _LEARNED_LAW + "test_learned_contact_conserves_heat_and_dissipates",
            _LEARNED_LAW
            + "test_every_certified_potential_gives_a_monotone_dissipative_heat_flow",
            _LEARNED_LAW + "test_an_affine_potential_recovers_the_linear_conductance",
            _LEARNED_LAW
            + "test_state_design_response_is_the_derivative_of_accepted_nonlinear_solves",
            _EXAMPLE + "[learned-interface]",
        ),
        negative=(
            _LEARNED_LAW + "test_a_potential_without_a_convexity_certificate_is_refused",
            _LEARNED_LAW
            + "test_a_potential_that_is_not_a_model_cannot_change_the_contact_law",
            _LEARNED_LAW + "test_a_negative_baseline_conductance_is_refused",
            _LEARNED_LAW
            + "test_nonlinear_contact_refuses_implicit_parameter_derivatives",
        ),
    ),
    Scenario(
        "learning",
        "hybrid-numerical-field",
        "surrogate-interface-flux",
        ("finite-element-p1", "pinn-surrogate"),
        "two-region-plate",
        "single-device",
        "state-design-response",
        positive=(
            _HYBRID + "test_accepted_hybrid_response_matches_host_finite_differences",
            _HYBRID + "test_finite_element_floor_converges_at_second_order",
            _HYBRID + "test_dirichlet_neumann_coupling_reaches_the_finite_element_floor",
        ),
        negative=(
            _HYBRID + "test_surrogate_is_refused_as_a_solution_map_design",
            _HYBRID + "test_implicit_solver_objective_refuses_to_train_the_surrogate",
            _HYBRID + "test_design_outside_the_admitted_lane_is_refused",
        ),
    ),
    Scenario(
        "measurement-inverse",
        "coupled-observation-inverse",
        "point-average-and-wall-heat-observations",
        ("finite-element-p1", "virtual-element"),
        "two-region-plate",
        "single-device",
        "implicit-adjoint",
        positive=(
            _INVERSE + "test_observations_at_the_truth_match_the_analytic_plate",
            _INVERSE + "test_training_recovers_conductivity_and_heat_flux",
            _PARAMETERS + "test_observations_carry_their_measurement_semantics",
            _PARAMETERS + "test_solve_binds_every_refresh_parameter_explicitly",
            _EXAMPLE + "[inverse-problem]",
        ),
        negative=(
            _INVERSE + "test_incompatible_data_is_refused_by_the_comparison",
            _INVERSE + "test_misdeclared_observation_is_refused_at_construction",
            _INVERSE + "test_parameter_binding_errors_are_refused_at_solve",
            _PARAMETERS
            + "test_boundary_statistics_are_not_identified_with_other_samplings",
            _PARAMETERS + "test_trace_samples_must_sit_at_the_trace_sites",
            _PARAMETERS + "test_parameter_declarations_are_validated",
        ),
    ),
    Scenario(
        "measurement-inverse",
        "measurement-noise-likelihood",
        "whitening-marginals-and-log-determinant",
        (
            "covariance-diagonal",
            "covariance-cholesky",
            "covariance-low-rank",
            "covariance-kronecker",
            "covariance-circulant",
            "covariance-precision",
        ),
        "sensor-vectors",
        "single-device",
        "noise-parameter-gradient",
        positive=(
            _COMPARISON + "test_whitened_residual_realizes_the_covariance_quadratic",
            _COMPARISON + "test_factorless_covariances_keep_quadratic_and_normalization",
            _COMPARISON + "test_restricted_covariance_is_the_exact_gaussian_marginal",
            _COMPARISON + "test_independent_uncertainty_normalizes_only_active_values",
            _LIKELIHOOD
            + "test_parameter_dependent_correlated_noise_gradient_includes_the_log_determinant",
            _POSTERIOR
            + "test_parameter_dependent_noise_gradient_includes_the_log_determinant",
        ),
        negative=(
            _COMPARISON + "test_structured_covariances_refuse_partial_restriction",
            _COMPARISON
            + "test_incomplete_observed_validity_refuses_a_full_correlated_covariance",
            _COMPARISON
            + "test_prediction_invalid_on_a_correlated_value_fails_the_comparison",
            _COMPARISON
            + "test_unquantified_data_require_an_explicit_reference_weighting",
        ),
    ),
    Scenario(
        "control-uq-rom",
        "coupled-control",
        "implicit-euler-plant-and-mpc",
        ("finite-element-p1", "virtual-element"),
        "two-region-plate",
        "single-device",
        "linearized-transition",
        positive=(
            _CONTROL + "test_coupled_rollout_is_the_host_implicit_euler_recursion",
            _CONTROL + "test_matrix_free_linearization_follows_the_step_context",
            _CONTROL
            + "test_dense_mpc_matches_the_host_bounded_optimum_and_replays_on_the_coupled_plant",
        ),
        negative=(
            _CONTROL + "test_mpc_sensitivity_refuses_a_weakly_active_flux_bound",
            _CONTROL + "test_failed_transition_is_never_repaired",
            _CONTROL + "test_refined_full_order_plant_exceeds_the_dense_budget",
            _CONTROL + "test_control_must_enter_the_coupled_rows",
        ),
    ),
    Scenario(
        "control-uq-rom",
        "coupled-state-space",
        "ensemble-kalman-filter",
        ("finite-element-p1", "virtual-element"),
        "two-region-plate",
        "single-device",
        "none",
        positive=(
            _STATE_SPACE
            + "test_coupled_forecast_uses_semantic_keys_and_analysis_is_the_ensemble_kalman_update",
            _STATE_SPACE
            + "test_ensemble_filter_tracks_the_exact_kalman_filter_of_the_affine_transition",
            _STATE_SPACE
            + "test_process_noise_density_is_the_host_gaussian_about_the_accepted_step",
        ),
        negative=(
            _STATE_SPACE
            + "test_deterministic_or_singular_noise_simulator_has_no_transition_density",
            _STATE_SPACE
            + "test_failed_coupled_transition_is_a_transition_failure_of_the_filter_step",
            _STATE_SPACE
            + "test_sensor_noise_requires_declared_uncertainty_and_matching_identities",
        ),
    ),
    Scenario(
        "control-uq-rom",
        "coupled-reduced-order",
        "galerkin-region-swap",
        ("pod-galerkin-rom", "virtual-element"),
        "two-region-plate",
        "single-device",
        "implicit-solution-map",
        positive=(
            _ROM
            + "test_reduced_region_keeps_the_plan_bindings_and_observation_identities",
            _ROM
            + "test_rank_sweep_converges_to_the_full_order_plate_with_its_interface_residual",
            _ROM + "test_reduced_plate_has_the_full_order_discretization_error",
            _ROM
            + "test_parameter_derivatives_through_the_reduced_solve_match_full_order",
            _EXAMPLE + "[rom-swap]",
        ),
        negative=(
            _ROM + "test_basis_is_fixed_structure_and_a_change_is_a_new_problem",
            _ROM + "test_petrov_galerkin_reduction_is_refused",
            _ROM + "test_basis_of_another_component_is_refused",
            _ROM + "test_trainable_model_hidden_in_the_provider_is_refused",
            _ROM + "test_reduced_component_requires_a_component_galerkin_model",
        ),
    ),
    Scenario(
        "lifecycle",
        "adaptive-rebind",
        "one-sided-refinement-transaction",
        ("finite-element-p1", "finite-volume-cell-centered"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _REBIND + "test_rebind_reprepares_the_changed_side_and_remaps_its_state",
            _REBIND + "test_rebind_conserves_heat_and_retains_budgets",
            _REBIND + "test_unchanged_side_is_retained_bitwise_and_observations_rebuilt",
            _REBIND + "test_rebound_run_converges_to_the_rebound_reference",
        ),
        negative=(
            _REBIND + "test_unknown_model_state_transport_is_refused",
            _REBIND + "test_failed_target_preparation_leaves_old_owners_usable",
            _REBIND + "test_retaining_a_stale_observation_is_refused",
            _REBIND + "test_rejected_boundary_publishes_nothing",
        ),
    ),
    Scenario(
        "lifecycle",
        "coupled-restart",
        "checkpoint-rebuild-continue",
        ("finite-element-p1", "finite-volume-cell-centered"),
        "unit-square-pair-cut-x1-nonmatching",
        "single-device",
        "none",
        positive=(
            _RESTART + "test_restart_resumes_bitwise_from_rebuilt_owners",
            _RESTART + "test_checkpoints_hold_only_portable_arrays",
            _RESTART + "test_restart_into_a_rebound_topology_uses_the_rebind_relation",
        ),
        negative=(
            _RESTART + "test_restart_refuses_changed_identities_without_a_relation",
            _RESTART + "test_refused_rebind_admits_no_restart_relation",
        ),
    ),
    Scenario(
        "lifecycle",
        "composition-rebind",
        "cross-owner-transaction",
        (
            "lifecycle-composition",
            "training-kernel",
            "mesh-distribution",
            "execution-worksets",
        ),
        "owner-entries",
        "single-device",
        "none",
        positive=(
            _COMPOSITION + "test_accepted_rebind_publishes_one_consistent_composition",
            _COMPOSITION + "test_invalidating_a_derived_artifact_drops_it",
            _COMPOSITION
            + "test_numeric_refresh_commits_a_new_revision_with_identical_layout",
            _COMPOSITION + "test_parameters_bind_their_consumers_only_by_semantics",
            _COMPOSITION
            + "test_parameters_survive_only_through_a_same_semantics_rebinding",
            _TRAINING_KERNEL
            + "test_composition_entries_classify_every_durable_training_state_field",
            _TRAINING_KERNEL
            + "test_rebind_keeps_parameters_only_through_a_same_semantics_kernel_binding",
            _DISTRIBUTION + "test_migration_transport_publishes_a_conserving_repartition",
        ),
        negative=(
            _COMPOSITION + "test_refused_commit_returns_the_original_composition",
            _COMPOSITION
            + "test_retaining_a_derived_artifact_on_a_changed_structure_is_refused",
            _COMPOSITION + "test_state_cannot_be_invalidated",
            _COMPOSITION + "test_state_cannot_be_reprepared_from_nothing",
            _COMPOSITION + "test_every_source_entry_takes_exactly_one_disposition",
            _COMPOSITION + "test_numeric_refresh_refuses_structural_change",
            _COMPOSITION
            + "test_transport_must_consume_the_structure_it_was_prepared_from",
            _COMPOSITION + "test_ownership_migration_must_report_moved_content",
            _COMPOSITION + "test_commit_requires_an_explicit_host_boundary_decision",
            _TRAINING_KERNEL
            + "test_rebind_refuses_retained_optimizer_after_objective_or_schema_change",
            _DISTRIBUTION + "test_migration_transport_refuses_created_and_refined_rows",
            _DISTRIBUTION
            + "test_migration_transport_refuses_entries_of_other_distributions",
            _WORKSET + "test_workset_entry_is_stale_until_reprepared_for_a_new_topology",
        ),
    ),
    Scenario(
        "junctions-embedded",
        "plateau-border-junction",
        "n-way-sheet-border-exchange",
        ("thin-film-sheets", "plateau-border-finite-volume"),
        "double-bubble-multiregion-surface",
        "single-device",
        "none",
        positive=(
            _FOAM + "test_sheet_half_edge_fluxes_feed_the_junction_equal_and_opposite",
            _FOAM + "test_junction_is_one_explicit_incidence_of_the_three_border_sheets",
            _FOAM + "test_exchange_conserves_liquid_and_surfactant_to_roundoff",
            _FOAM + "test_split_publishes_reprepared_owners_and_explicit_transports",
            _FOAM + "test_split_conserves_content_and_keeps_the_border_cross_section",
            _FOAM + "test_refined_network_continues_the_exchange",
            _FOAM
            + "test_rendering_the_published_thickness_leaves_physics_bitwise_unchanged",
        ),
        negative=(
            _FOAM + "test_rupture_without_a_border_content_rule_is_refused",
            _FOAM + "test_t1_pop_without_a_border_content_rule_is_refused",
            _FOAM + "test_collapsing_a_border_has_no_declared_coarsening_rule",
            _FOAM + "test_unknown_border_state_transport_is_refused",
            _FOAM + "test_rejected_boundary_publishes_nothing",
        ),
    ),
    Scenario(
        "junctions-embedded",
        "interface-incidence",
        "ordered-junction-overlap-embedded",
        ("mesh-assembly", "multiregion-surface", "analytic-domain"),
        "brep-plate-and-double-bubble",
        "single-device",
        "none",
        positive=(
            _BINDINGS
            + "test_multiregion_sheets_bind_two_sided_walls_and_ordered_junctions",
            _BINDINGS
            + "test_overlap_is_unsided_and_embedded_incidence_declares_its_meaning",
        ),
        negative=(
            _MISSING_WITNESS + "[junction-of-two]",
            _MISSING_WITNESS + "[sided-overlap]",
            _MISSING_WITNESS + "[unsided-two-sided]",
        ),
    ),
    Scenario(
        "external",
        "fmi-host-coupling",
        "partitioned-fmu-native-exchange",
        ("fmi2-cosimulation-thermal-zone", "native-lumped-node"),
        "lumped-thermal-network",
        "host-orchestrated",
        "refused-opaque-external",
        positive=(
            _FMI_COUPLING
            + "test_binding_resolves_exact_factors_and_restore_from_the_model_description",
            _FMI_COUPLING
            + "test_explicit_route_without_restore_conserves_heat_and_converges",
            _FMI_COUPLING
            + "test_implicit_route_replays_every_iterate_from_the_restored_fmu_state",
            _FMI_COUPLING
            + "test_rejected_window_restores_the_fmu_and_the_retry_replays_bitwise",
            _FMI_COUPLING + "test_fmu_discard_rejects_the_window_and_restores_the_zone",
            _FMI_COUPLING + "test_uniform_rate_and_sample_realize_native_conduction",
            _FMI + "test_real_fmu_integration_event_and_actual_state_restore",
        ),
        negative=(
            _FMI_COUPLING
            + "test_binding_refuses_variables_the_model_description_does_not_support",
            _FMI_COUPLING + "test_variable_binding_refuses_ports_it_cannot_realize",
            _FMI_COUPLING
            + "test_rejected_window_without_restore_is_unrecoverable_and_refuses_to_continue",
            _FMI_COUPLING
            + "test_native_preparation_refuses_host_participants_and_leaves_the_fmu_usable",
            _FMI_COUPLING + "test_implicit_route_requires_real_fmu_state_restore",
            _FMI_COUPLING + "test_derivative_requests_through_the_fmu_are_refused",
            _FMI_COUPLING + "test_host_route_requires_a_host_participant",
        ),
        optional_providers=(
            OptionalProvider("fmpy", "python-module"),
            OptionalProvider("cc", "executable"),
        ),
    ),
    Scenario(
        "execution-wrappers",
        "pic-capability-matrix",
        "distributed-wrapper-preservation",
        (
            "cochain-3d",
            "psatd-global-fft",
            "psatd-local-guarded",
            "quasi-cylindrical-psatd",
            "reduced-1d-2d",
            "unstructured-whitney",
        ),
        "periodic-and-bounded-grids",
        "single-device-and-four-device-child",
        "none",
        positive=(
            _PIC_CAPABILITIES + "test_base_matrix_is_the_structural_protocol_set",
            _PIC_CAPABILITIES
            + "test_distributed_wrapper_publishes_exactly_its_executed_routes",
            _PIC_CAPABILITIES + "test_distributed_protocol_routes_on_four_devices",
            _PIC_CAPABILITIES + "test_spectral_observers_admit_huygens_sampling",
            _PIC_CAPABILITIES
            + "test_runtime_drifts_particles_by_the_admitted_grid_velocity",
        ),
        negative=(
            _PIC_CAPABILITIES + "test_published_but_refused_protocol_refuses_when_called",
            _PIC_CAPABILITIES
            + "test_cochain_huygens_sampling_is_refused_beside_pic_current",
            _PIC_CAPABILITIES + "test_distribution_refusals_are_declared_and_enforced",
            _PIC_CAPABILITIES + "test_withheld_protocols_are_refused_by_their_consumers",
            _PIC_CAPABILITIES
            + "test_observerless_standard_spectral_protocols_refuse_when_called",
        ),
    ),
    Scenario(
        "execution-wrappers",
        "distributed-pic",
        "domain-decomposed-particles-and-fields",
        ("reduced", "cochain", "psatd-global-fft", "psatd-local-guarded"),
        "slabs-and-blocks",
        "forced-host-devices-4",
        "none",
        positive=(
            _DISTRIBUTED_PIC
            + "test_halo_accumulation_and_guard_exchange_match_global_cells",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[reduced-4]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[cochain-4]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[cochain-2x2]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[spectral-global-fft-4]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[spectral-global-fft-2x2]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[spectral-local-guarded-4]",
            _DISTRIBUTED_PIC
            + "test_distributed_run_matches_single_device_run[spectral-local-guarded-2x2]",
            _DISTRIBUTED_PIC
            + "test_local_guarded_matches_global_fft_within_stencil_truncation",
            _DISTRIBUTED_PIC
            + "test_diagonal_migration_crosses_a_block_corner_in_one_packet",
            _DISTRIBUTED_PIC + "test_same_topology_restart_continues_bitwise",
        ),
        negative=(
            _DISTRIBUTED_PIC + "test_migration_overflow_rejects_the_whole_step",
            _DISTRIBUTED_PIC + "test_spectral_transforms_off_the_pic_mesh_are_refused",
            _DISTRIBUTED_PIC + "test_deposits_that_are_not_window_local_are_refused",
            _DISTRIBUTED_PIC + "test_size_one_mesh_axes_are_refused",
            _DISTRIBUTED_PIC
            + "test_guard_narrower_than_the_transfer_footprint_is_refused",
            _DISTRIBUTED_PIC
            + "test_processes_without_the_distributed_protocol_are_refused",
        ),
        host_devices=4,
    ),
    Scenario(
        "execution-wrappers",
        "coupled-lane-worksets",
        "signature-grouped-lanes",
        ("finite-element-p1",),
        "unit-strip-chain",
        "single-device",
        "none",
        positive=(
            _LANES + "test_homogeneous_strips_form_bounded_component_worksets",
            _LANES + "test_lane_residual_and_operators_match_per_component_reference",
            _LANES + "test_lane_solve_is_certified_like_the_reference",
            _LANES
            + "test_equal_programs_with_different_coefficients_share_lanes_with_own_data",
            _LANES + "test_caller_state_and_prepared_lanes_stay_reusable",
            _WORKSET + "test_filter_vmap_worksets_map_only_the_declared_item_lane",
        ),
        negative=(
            _LANES + "test_signature_separates_equal_shapes_with_different_programs",
            _LANES + "test_runtime_arguments_outside_the_prepared_signature_are_refused",
            _LANES + "test_lane_capacity_outside_the_bucket_range_is_refused",
        ),
    ),
    Scenario(
        "execution-wrappers",
        "coupled-lane-worksets",
        "execution-group-placement",
        ("finite-element-p1",),
        "unit-strip-chain",
        "forced-host-devices-4",
        "none",
        positive=(
            _LANES + "test_lanes_on_an_execution_group_match_single_device_reference",
        ),
        negative=(
            _LANES + "test_runtime_arguments_outside_the_prepared_signature_are_refused",
        ),
        host_devices=4,
    ),
    Scenario(
        "scaling",
        "prepared-route-scaling",
        "query-trace-and-boundary-operator-bounds",
        (
            "finite-element",
            "isogeometric-nurbs",
            "finite-difference-sbp",
            "finite-volume",
            "galerkin-2d-p1-dp0",
            "spectral-element-gll",
            "virtual-element",
        ),
        "benchmark-row-geometries",
        "single-device",
        "none",
        positive=(
            _QUERIES
            + "test_prepared_query_route_size_is_independent_of_the_coefficient_count",
            _SIDES + "test_fe_trace_route_is_local_to_facets",
            _SIDES + "test_fe_multiblock_traces_pad_mixed_cell_blocks",
        ),
        negative=(
            _GALERKIN + "test_resource_and_quadrature_limits_refuse_before_use",
            _FLAGSHIP
            + "test_unsupported_declarations_are_refused_at_preparation[galerkin-resident-bytes]",
            _FLAGSHIP + "test_dense_materialization_excess_is_refused_at_solve",
            _CONDENSATION
            + "test_unqualified_condensations_are_refused[materialization-budget]",
        ),
        benchmark_rows=(
            "prepared-query",
            "strong-form-query",
            "isogeometric-query",
            "finite-volume-face-trace",
            "interface-coupling",
            "coupled-assembly",
            "boundary-integral-law",
            "sem-vem-bem-flagship",
        ),
    ),
    Scenario(
        "scaling",
        "workset-scaling",
        "bounded-homogeneous-lanes",
        ("finite-element-p1",),
        "unit-strip-chain",
        "single-device",
        "none",
        positive=(
            _LANES + "test_homogeneous_strips_form_bounded_component_worksets",
            _WORKSET + "test_filter_vmap_worksets_broadcast_static_module_leaves",
        ),
        negative=(
            _LANES + "test_working_set_above_the_declared_bound_is_refused",
            _LANES
            + "test_lane_capacity_outside_the_bucket_range_is_refused[above-bucket-range]",
        ),
        benchmark_rows=(
            "homogeneous-worksets",
            "execution-group-parity",
            "coupled-observation-count",
            "coupled-temporal-samples",
        ),
    ),
    Scenario(
        "scaling",
        "coupled-consumer-scaling",
        "rebind-derivative-and-transition-reuse",
        ("finite-element-p1", "virtual-element", "fem-bem-scalar-3d"),
        "two-region-plate-and-tetrahedral-bipyramid",
        "single-device",
        "implicit-solution-map",
        positive=(
            _QUERIES
            + "test_prepared_query_reuses_its_route_across_coefficient_refreshes",
            _FEM_BEM_SCALAR
            + "test_compiled_conductivity_gradient_rebinds_without_host_synchronization",
        ),
        negative=(_CONTROL + "test_refined_full_order_plant_exceeds_the_dense_budget",),
        benchmark_rows=(
            "coupled-derivative",
            "coupled-repeated-bind",
            "conductivity-rebind",
            "coupled-transition",
        ),
    ),
)
_BY_ID = {scenario.scenario_id: scenario for scenario in SCENARIOS}
if len(_BY_ID) != len(SCENARIOS):
    raise RuntimeError("Numerical-interoperability scenario IDs must be unique.")


def select_scenarios(
    scenario_ids: Sequence[str] = (), families: Sequence[str] = (), /
) -> tuple[Scenario, ...]:
    """Return the registry-ordered union of the named scenarios and families."""
    unknown_ids = sorted(set(scenario_ids) - set(_BY_ID))
    if unknown_ids:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown_ids)}.")
    unknown_families = sorted(set(families) - set(FAMILIES))
    if unknown_families:
        raise ValueError(f"Unknown scenario families: {', '.join(unknown_families)}.")
    if not scenario_ids and not families:
        return SCENARIOS
    return tuple(
        scenario
        for scenario in SCENARIOS
        if scenario.scenario_id in scenario_ids or scenario.family in families
    )


def reference_path(reference: str, /) -> str:
    return reference.partition("::")[0]


def reference_matches(reference: str, nodeid: str, /) -> bool:
    """Whether ``nodeid`` is the referenced node or one of its parametrizations."""
    return nodeid == reference or (
        "[" not in reference and nodeid.startswith(reference + "[")
    )


def referenced_test_files(scenarios: Sequence[Scenario], /) -> tuple[str, ...]:
    """Sorted test files that the scenarios reference."""
    return tuple(
        sorted(
            {
                reference_path(reference)
                for scenario in scenarios
                for _, reference in scenario.references
            }
        )
    )


@dataclass(frozen=True)
class RouteObservation:
    """One pytest session over every node of the scenarios sharing a device count."""

    host_devices: int
    run: PytestRun
    wall_seconds: float


@dataclass(frozen=True)
class QualificationObservation:
    """Everything the runner observed; the report is a pure function of it."""

    collection: PytestCollection
    routes: tuple[RouteObservation, ...]
    providers: Mapping[str, bool]
    benchmark_rows: frozenset[str] | None


@dataclass(frozen=True)
class ScenarioOutcome:
    """Qualification outcome of one scenario and the rule that decided it."""

    outcome: Literal["passed", "failed", "inconclusive"]
    reason: str


def route_topology(host_devices: int, /) -> str:
    if host_devices == 1:
        return "in-process-pytest"
    return f"subprocess-pytest-forced-host-devices-{host_devices}"


def route_environment(host_devices: int, /) -> dict[str, str]:
    """Startup variables of a fresh interpreter with ``host_devices`` host devices."""
    if host_devices == 1:
        return {}
    flags = os.environ.get("XLA_FLAGS", "")
    forced = f"--xla_force_host_platform_device_count={host_devices}"
    return {"XLA_FLAGS": f"{flags} {forced}".strip()}


def registered_benchmark_rows(root: Path = _PROJECT_ROOT, /) -> frozenset[str] | None:
    """Row names of the benchmark campaign, or ``None`` when it cannot be imported.

    The campaign enables float64 on import, so it is inspected in a fresh
    interpreter rather than in the qualification process.
    """
    driver = Path(BENCHMARK_DRIVER)
    script = (
        f"import json, sys; sys.path.insert(0, {driver.parent.as_posix()!r}); "
        f"import {driver.stem} as campaign; "
        "print(json.dumps(sorted(campaign.ROWS)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = completed.stdout.strip().splitlines()
    if completed.returncode != 0 or not lines:
        return None
    return frozenset(json.loads(lines[-1]))


def observe_scenarios(
    scenarios: Sequence[Scenario], /, *, workers: int, route_timeout: float | None = None
) -> QualificationObservation:
    """Collect, then run every resolved node once per device-count route.

    ``route_timeout`` (seconds) bounds each fresh-interpreter route; a route
    that misses it is observed as a failed collection with no nodes.
    """
    root = _PROJECT_ROOT.as_posix()
    collection = collect_pytest_nodes(
        [f"{root}/{path}" for path in referenced_test_files(scenarios)],
        root=_PROJECT_ROOT,
    )
    by_devices: dict[int, set[str]] = {}
    for scenario in scenarios:
        nodes = by_devices.setdefault(scenario.host_devices, set())
        for _, reference in scenario.references:
            nodes.update(
                nodeid
                for nodeid in collection.nodeids
                if reference_matches(reference, nodeid)
            )
    routes = []
    for host_devices, nodeids in sorted(by_devices.items()):
        selection = [f"{root}/{nodeid}" for nodeid in sorted(nodeids)]
        started = time.perf_counter()
        if not selection:
            run = PytestRun(nodes=(), collection_failed=False)
        elif host_devices == 1:
            run = run_pytest(selection, root=_PROJECT_ROOT, workers=workers)
        else:
            run = run_pytest_subprocess(
                selection,
                root=_PROJECT_ROOT,
                workers=workers,
                environment=route_environment(host_devices),
                timeout=route_timeout,
            )
        routes.append(RouteObservation(host_devices, run, time.perf_counter() - started))
    providers = {
        provider.name: provider.available()
        for scenario in scenarios
        for provider in scenario.optional_providers
    }
    rows = (
        registered_benchmark_rows()
        if any(scenario.benchmark_rows for scenario in scenarios)
        else frozenset()
    )
    return QualificationObservation(collection, tuple(routes), providers, rows)


def _uncollectable(reference: str, failed_collectors: Sequence[str], /) -> bool:
    """Whether a failed module, class, package, or directory collector covers it."""
    path = reference_path(reference)
    return any(
        reference == collector
        or path == collector
        or reference.startswith(collector + "::")
        or path.startswith(collector + "/")
        for collector in failed_collectors
    )


def _decide(
    *,
    failed_roles: set[ScenarioRole],
    unresolved: bool,
    unregistered_rows: bool,
    uncollectable: bool,
    unobserved: bool,
    collection_failed: bool,
    registry_unavailable: bool,
    provider_unavailable: bool,
    skipped: bool,
) -> tuple[Literal["passed", "failed", "inconclusive"], str]:
    """Failures dominate; missing evidence is inconclusive; only a full pass passes."""
    if failed_roles == {"positive", "negative"}:
        return "failed", "failed-positive-evidence-and-negative-boundary"
    if failed_roles == {"positive"}:
        return "failed", "failed-positive-evidence"
    if failed_roles:
        return "failed", "failed-negative-boundary"
    if unresolved:
        return "failed", "unresolved-node-reference"
    if unregistered_rows:
        return "failed", "unregistered-benchmark-row"
    if uncollectable or (unobserved and collection_failed):
        return "inconclusive", "collection-failed"
    if unobserved:
        return "inconclusive", "nodes-not-observed"
    if registry_unavailable:
        return "inconclusive", "benchmark-registry-unavailable"
    if provider_unavailable:
        return "inconclusive", "optional-provider-unavailable"
    if skipped:
        return "inconclusive", "skipped-nodes"
    return "passed", "all-nodes-passed"


def classify_scenario(
    scenario: Scenario, observation: QualificationObservation, /
) -> tuple[ScenarioOutcome, dict[str, object], dict[str, float]]:
    """Return the outcome, raw observation, and measured node durations."""
    route = next(
        (
            item
            for item in observation.routes
            if item.host_devices == scenario.host_devices
        ),
        None,
    )
    observed: dict[str, NodeOutcome] = (
        {} if route is None else {node.nodeid: node for node in route.run.nodes}
    )
    run_collection_failed = route is not None and route.run.collection_failed
    failed_collectors = observation.collection.failed_collectors
    evidence: dict[ScenarioRole, list[dict[str, object]]] = {
        "positive": [],
        "negative": [],
    }
    durations: dict[str, float] = {}
    failed_roles: set[ScenarioRole] = set()
    unresolved: list[str] = []
    uncollectable: list[str] = []
    unobserved = skipped = unqualified = 0
    for role, reference in scenario.references:
        matched = [
            nodeid
            for nodeid in observation.collection.nodeids
            if reference_matches(reference, nodeid)
        ]
        if not matched:
            unqualified += 1
            if observation.collection.aborted or _uncollectable(
                reference, failed_collectors
            ):
                uncollectable.append(reference)
            else:
                unresolved.append(reference)
        nodes = []
        for nodeid in matched:
            node = observed.get(nodeid)
            if node is None:
                unobserved += 1
                unqualified += 1
                nodes.append({"nodeid": nodeid, "outcome": "not-observed", "message": ""})
                continue
            durations[nodeid] = node.duration_seconds
            unqualified += node.outcome != "passed"
            skipped += node.outcome == "skipped"
            if node.outcome == "failed":
                failed_roles.add(role)
            nodes.append(
                {"nodeid": nodeid, "outcome": node.outcome, "message": node.message}
            )
        evidence[role].append({"reference": reference, "nodes": nodes})
    providers = [
        {
            "name": provider.name,
            "kind": provider.kind,
            "available": observation.providers.get(provider.name, False),
        }
        for provider in scenario.optional_providers
    ]
    registered = observation.benchmark_rows
    rows = [
        {"row": row, "registered": registered is not None and row in registered}
        for row in scenario.benchmark_rows
    ]
    outcome, reason = _decide(
        failed_roles=failed_roles,
        unresolved=bool(unresolved),
        unregistered_rows=registered is not None
        and not all(row["registered"] for row in rows),
        uncollectable=bool(uncollectable),
        unobserved=bool(unobserved),
        collection_failed=run_collection_failed,
        registry_unavailable=bool(rows) and registered is None,
        provider_unavailable=not all(provider["available"] for provider in providers),
        skipped=bool(skipped),
    )
    raw: dict[str, object] = {
        "kind": "numerical-interoperability-scenario-observation",
        "scenario": scenario.scenario_id,
        "family": scenario.family,
        "route": {
            "host_devices": scenario.host_devices,
            "topology": route_topology(scenario.host_devices),
        },
        "positive": evidence["positive"],
        "negative": evidence["negative"],
        "unresolved_references": unresolved,
        "uncollectable_references": uncollectable,
        "optional_providers": providers,
        "benchmark_rows": rows,
        "unqualified_nodes": unqualified,
    }
    return ScenarioOutcome(outcome, reason), raw, durations


@dataclass(frozen=True)
class _Campaign:
    campaign_spec_id: str
    replay_id: str
    build_id: str
    environment_id: str
    backend: str
    precision: str


def _scenario_record(
    scenario: Scenario,
    observation: QualificationObservation,
    campaign: _Campaign,
    /,
) -> tuple[str, dict[str, object]]:
    identity = QualificationRuntimeIdentity(
        campaign.build_id,
        campaign.environment_id,
        campaign.backend,
        route_topology(scenario.host_devices),
        campaign.precision,
    )
    support = SupportTuple(
        CAPABILITY,
        {
            "family": scenario.family,
            **scenario.coordinates(),
            "backend": campaign.backend,
            "precision": campaign.precision,
        },
    )
    criterion = QualificationCriterion(
        support_tuple_id=support.support_tuple_id,
        metric="unqualified-nodes",
        unit="count",
        comparison="equal",
        target=0,
        aggregation="sum",
        uncertainty="deterministic",
        applicability=scenario.scenario_id,
        approval_id=APPROVAL_ID,
        issued_at=CRITERION_TICK,
    )
    start = CampaignStartRecord(
        campaign_spec_id=campaign.campaign_spec_id,
        criterion_id=criterion.criterion_id,
        resolved_run_spec_id=campaign.replay_id,
        support_tuple_id=support.support_tuple_id,
        started_at=STARTED_TICK,
    )
    classified, raw, durations = classify_scenario(scenario, observation)
    raw_artifact_id = canonical_fingerprint(raw)
    campaign_observation = CampaignObservationRecord(
        start_record_id=start.start_record_id,
        campaign_spec_id=campaign.campaign_spec_id,
        criterion_id=criterion.criterion_id,
        resolved_run_spec_id=campaign.replay_id,
        support_tuple_id=support.support_tuple_id,
        raw_artifact_ids=(raw_artifact_id,),
        observed_at=OBSERVED_TICK,
    )
    evidence = QualificationEvidence(
        _EVIDENCE_KIND.get(scenario.family, "scientific"),
        classified.outcome,
        (support.support_tuple_id,),
        build_id=identity.build_id,
        environment_id=identity.environment_id,
        backend=identity.backend,
        topology=identity.topology,
        precision=identity.precision,
        reduction="deterministic:sum",
        replay_id=campaign.replay_id,
        criteria_ids=(criterion.criterion_id,),
        raw_artifact_ids=(raw_artifact_id,),
        campaign_start_record_ids=(start.start_record_id,),
        campaign_observation_record_ids=(campaign_observation.observation_record_id,),
        reviewer_id=REVIEWER_ID,
        issued_at=ISSUED_TICK,
        expires_at=EXPIRES_TICK,
        reason=classified.reason,
        requalification_triggers=(
            "build-change",
            "environment-change",
            "scenario-registry-change",
        ),
    )
    validate_qualification_causality(criterion, start, campaign_observation, evidence)
    return classified.outcome, {
        "scenario": scenario.scenario_id,
        "family": scenario.family,
        "coordinates": scenario.coordinates(),
        "runtime_identity": dict(identity.to_record()),
        "support_tuple": support.to_record(),
        "criterion": criterion.to_record(),
        "campaign_start": start.to_record(),
        "raw_output": {**raw, "raw_artifact_id": raw_artifact_id},
        "campaign_observation": campaign_observation.to_record(),
        "evidence": evidence.to_record(),
        "timing": {
            "duration_seconds": sum(durations.values()),
            "node_duration_seconds": dict(sorted(durations.items())),
        },
    }


def _registry_record(scenarios: Sequence[Scenario], /) -> list[dict[str, object]]:
    return [
        {
            "scenario": scenario.scenario_id,
            "family": scenario.family,
            "positive": list(scenario.positive),
            "negative": list(scenario.negative),
            "optional_providers": [
                [provider.name, provider.kind] for provider in scenario.optional_providers
            ],
            "host_devices": scenario.host_devices,
            "benchmark_rows": list(scenario.benchmark_rows),
        }
        for scenario in scenarios
    ]


def qualification_report(
    scenarios: Sequence[Scenario],
    observation: QualificationObservation,
    /,
    *,
    build_id: str,
    provenance: Mapping[str, object],
    environment: Mapping[str, object],
    backend: str,
    precision: str,
) -> dict[str, object]:
    """Bind observed node outcomes into one causal record chain per scenario."""
    if not scenarios:
        raise ValueError("A qualification report needs at least one scenario.")
    selected = select_scenarios(tuple(scenario.scenario_id for scenario in scenarios))
    environment_record = dict(environment)
    environment_id = canonical_fingerprint(environment_record)
    campaign = _Campaign(
        campaign_spec_id=canonical_fingerprint(
            {
                "kind": "numerical-interoperability-qualification-campaign",
                "scenarios": [scenario.scenario_id for scenario in selected],
            }
        ),
        replay_id=canonical_fingerprint(_registry_record(selected)),
        build_id=build_id,
        environment_id=environment_id,
        backend=backend,
        precision=precision,
    )
    outcomes: dict[str, str] = {}
    records = []
    for scenario in selected:
        outcome, record = _scenario_record(scenario, observation, campaign)
        outcomes[scenario.scenario_id] = outcome
        records.append(record)
    passed = [key for key, value in outcomes.items() if value == "passed"]
    failed = [key for key, value in outcomes.items() if value == "failed"]
    inconclusive = [key for key, value in outcomes.items() if value == "inconclusive"]
    unselected = [
        scenario.scenario_id
        for scenario in SCENARIOS
        if scenario.scenario_id not in outcomes
    ]
    if failed:
        outcome = "failed"
    elif inconclusive or unselected:
        outcome = "inconclusive"
    else:
        outcome = "passed"
    return {
        "kind": "numerical-interoperability-qualification-report",
        "clock": "logical",
        "build_id": build_id,
        "provenance": dict(provenance),
        "environment": {**environment_record, "environment_id": environment_id},
        "replay_id": campaign.replay_id,
        "collection": {
            "collected_nodes": len(observation.collection.nodeids),
            "failed_collectors": list(observation.collection.failed_collectors),
        },
        "routes": [
            {
                "host_devices": route.host_devices,
                "topology": route_topology(route.host_devices),
                "environment_overrides": route_environment(route.host_devices),
                "observed_nodes": len(route.run.nodes),
                "collection_failed": route.run.collection_failed,
                "wall_seconds": route.wall_seconds,
            }
            for route in observation.routes
        ],
        "scenarios": records,
        "passed_scenarios": passed,
        "failed_scenarios": failed,
        "inconclusive_scenarios": inconclusive,
        "unselected_scenarios": unselected,
        "outcome": outcome,
    }


def _file_digests(root: Path, paths: Sequence[Path], /) -> list[dict[str, str]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(set(paths))
    ]


def qualification_build_id(root: Path = _PROJECT_ROOT, /) -> str:
    """Content-address the package build, the referenced tests, and this runner."""
    inputs = [
        *(root / path for path in referenced_test_files(SCENARIOS)),
        *(root / "tests").rglob("conftest.py"),
        *(root / "tests" / "_support").rglob("*.py"),
        root / "tools" / "numerical_interoperability_qualification.py",
        root / "tools" / "_pytest_outcomes.py",
        root / BENCHMARK_DRIVER,
    ]
    return canonical_fingerprint(
        {
            "kind": "numerical-interoperability-qualification-build",
            "source_build": source_build_fingerprint(root),
            "inputs": _file_digests(root, [path for path in inputs if path.is_file()]),
        }
    )


def git_provenance(root: Path = _PROJECT_ROOT, /) -> dict[str, object]:
    """Return the checked-out revision and whether the worktree differs from it."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": None, "git_dirty": None}
    return {"git_revision": revision, "git_dirty": bool(status.strip())}


def format_listing(scenarios: Sequence[Scenario], /) -> str:
    lines = []
    for family, title in FAMILIES.items():
        members = [scenario for scenario in scenarios if scenario.family == family]
        if not members:
            continue
        lines.append(f"{family} ({title})")
        for scenario in members:
            details = [
                f"positive={len(scenario.positive)}",
                f"negative={len(scenario.negative)}",
            ]
            if scenario.optional_providers:
                names = ",".join(
                    provider.name for provider in scenario.optional_providers
                )
                details.append(f"optional-providers={names}")
            if scenario.benchmark_rows:
                details.append(f"benchmark-rows={','.join(scenario.benchmark_rows)}")
            lines.append(f"  {scenario.scenario_id}  {' '.join(details)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--scenario", action="append", default=[], dest="scenarios")
    parser.add_argument(
        "--family", action="append", default=[], choices=tuple(FAMILIES), dest="families"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--route-timeout", type=float, metavar="SECONDS")
    arguments = parser.parse_args()
    timeout = arguments.route_timeout
    if timeout is not None and not (math.isfinite(timeout) and timeout > 0.0):
        parser.error("--route-timeout must be positive and finite.")
    try:
        scenarios = select_scenarios(arguments.scenarios, arguments.families)
    except ValueError as error:
        parser.error(str(error))
    if arguments.list:
        print(format_listing(scenarios), end="")
        return
    observation = observe_scenarios(
        scenarios, workers=arguments.workers, route_timeout=arguments.route_timeout
    )
    build_id = qualification_build_id()
    report = qualification_report(
        scenarios,
        observation,
        build_id=build_id,
        provenance={
            **git_provenance(),
            "source_build_fingerprint": source_build_fingerprint(_PROJECT_ROOT),
            "qualification_build_id": build_id,
        },
        environment=capture_environment().to_dict(),
        backend=jax.default_backend(),
        precision="float64" if jax.config.jax_enable_x64 else "float32",
    )
    if arguments.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        write_json_atomic(arguments.output, report)
    if report["failed_scenarios"] or report["inconclusive_scenarios"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
