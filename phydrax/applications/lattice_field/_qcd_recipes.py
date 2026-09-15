#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Small-volume QCD recipes lowering directly to native Phydrax contracts."""

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._lattice_boundary import LatticeBoundaryPhasePlan
from ...discretization._lattice_distribution import LatticeDecompositionPlan
from ...discretization._oriented_path import CellBoundaryPathPlan
from ...graph._gauge_transport import GaugeCovariantShiftPlan, GaugeStaplePlan
from ...graph._matrix_gauge import MatrixGaugeLinkSpace
from ...linalg import SpectralInterval
from ...metrix import SpecialUnitaryGroup
from ...metrix._gauge_representation import FundamentalGaugeRepresentation
from ...operators.path_integral._lattice_action import (
    compact_geometric_target_from_lattice_action,
)
from ...operators.path_integral._lattice_fermion import (
    AbstractLatticeDiracOperator,
    CloverWilsonDiracOperator,
    LatticeFermionResourcePolicy,
    NaikStaggeredDiracOperator,
    StaggeredDiracOperator,
    WilsonDiracOperator,
)
from ...operators.path_integral._pseudofermion import (
    dirac_normal_operator,
    FractionalPowerPseudofermionTerm,
    PseudofermionSolveRoles,
    TwoFlavorPseudofermionTerm,
)
from ...operators.path_integral._rational_approximation import (
    CertifiedRationalApproximation,
    generate_minimax_rational_approximation,
    plan_minimax_rational_approximation,
    power_rational_target,
)
from ...operators.path_integral._wilson_gauge import WilsonGaugeAction
from ...sampling._compact_group_hamiltonian import (
    CompactGeometricTarget,
    prepare_compact_group_hamiltonian_kernel,
    PreparedCompactGroupHamiltonianKernel,
)
from ...sampling._rhmc import (
    NestedForcePartition,
    NestedForcePlan,
    plan_rhmc,
    prepare_rhmc,
    PreparedRHMCKernel,
    RHMCPlan,
    RHMCResourcePolicy,
    SeparableActionRegistry,
    SeparableActionTerm,
)
from ._distributed_qcd import (
    DistributedGaugeTheoryPlan,
    DistributedHMCPlan,
    DistributedRHMCPlan,
    prepare_distributed_hmc,
    prepare_distributed_rhmc,
    PreparedDistributedGaugeTheory,
    PreparedDistributedHMC,
    PreparedDistributedRHMC,
)
from ._qcd_ensembles import MeasurementSchedule
from ._qcd_observables import native_wilson_clover_term


FermionVariant: TypeAlias = Literal["wilson", "clover"]
SpectralEvidence: TypeAlias = Literal["verified", "asserted"]
RecipeExecutionScope: TypeAlias = Literal[
    "single-domain-reference", "decomposed-global-reference"
]


def _identifier(value: str, name: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


def _finite_positive(value: float, name: str, /) -> float:
    resolved = float(value)
    if not isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return resolved


def _spectral_bounds(
    lower: float,
    upper: float,
    evidence: SpectralEvidence,
    /,
) -> tuple[float, float, SpectralEvidence]:
    lower_ = _finite_positive(lower, "spectral_lower")
    upper_ = _finite_positive(upper, "spectral_upper")
    if upper_ <= lower_:
        raise ValueError("spectral_upper must exceed spectral_lower.")
    if evidence not in ("verified", "asserted"):
        raise ValueError("spectral_evidence must be 'verified' or 'asserted'.")
    return lower_, upper_, evidence


def _rhmc_resource_payload(policy: RHMCResourcePolicy, /) -> dict[str, int]:
    return {
        "maximum_terms": policy.maximum_terms,
        "maximum_force_evaluations": policy.maximum_force_evaluations,
        "maximum_retained_bytes": policy.maximum_retained_bytes,
    }


def _spectral_interval(
    operator: AbstractLatticeDiracOperator,
    lower: float,
    upper: float,
    evidence: SpectralEvidence,
    /,
) -> SpectralInterval:
    normal = dirac_normal_operator(operator)
    return SpectralInterval(
        normal,
        lower,
        upper,
        evidence=evidence,
        scope="structural",
    )


class SU3GaugeGeometry(StrictModule, NonTrainableState):
    """One aligned SU(3) plaquette, staple, fermion-route, and decomposition bundle."""

    link_space: MatrixGaugeLinkSpace
    plaquettes: CellBoundaryPathPlan
    fermion_boundary: LatticeBoundaryPhasePlan
    representation: FundamentalGaugeRepresentation
    transport: GaugeCovariantShiftPlan
    staples: GaugeStaplePlan
    decomposition: LatticeDecompositionPlan | None
    lattice_shape: tuple[int, ...] = eqx.field(static=True)
    execution_scope: RecipeExecutionScope = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        link_space: MatrixGaugeLinkSpace,
        plaquettes: CellBoundaryPathPlan,
        fermion_boundary: LatticeBoundaryPhasePlan,
        transport: GaugeCovariantShiftPlan,
        staples: GaugeStaplePlan,
        /,
        *,
        decomposition: LatticeDecompositionPlan | None = None,
    ):
        if not isinstance(link_space, MatrixGaugeLinkSpace):
            raise TypeError("link_space must be MatrixGaugeLinkSpace.")
        if not isinstance(link_space.group, SpecialUnitaryGroup) or (
            link_space.group.dimension != 3
        ):
            raise TypeError("QCD geometry requires an SU(3) link space.")
        if not isinstance(plaquettes, CellBoundaryPathPlan):
            raise TypeError("plaquettes must be CellBoundaryPathPlan.")
        if not isinstance(fermion_boundary, LatticeBoundaryPhasePlan):
            raise TypeError("fermion_boundary must be LatticeBoundaryPhasePlan.")
        if not isinstance(transport, GaugeCovariantShiftPlan):
            raise TypeError("transport must be GaugeCovariantShiftPlan.")
        if not isinstance(staples, GaugeStaplePlan):
            raise TypeError("staples must be GaugeStaplePlan.")
        representation = FundamentalGaugeRepresentation(link_space.group)
        if (
            plaquettes.paths.topology_id != link_space.topology.topology_id
            or staples.link_space.link_space_id != link_space.link_space_id
            or staples.boundaries.boundary_plan_id != plaquettes.boundary_plan_id
            or transport.link_space_id != link_space.link_space_id
            or transport.boundary_id != fermion_boundary.plan_id
            or transport.representation_id != representation.representation_id
            or fermion_boundary.site_count != link_space.num_vertices
        ):
            raise ValueError("SU(3) gauge geometry contracts are not mutually aligned.")
        lattice_shape = fermion_boundary.topology.axis_sizes
        if decomposition is not None:
            if not isinstance(decomposition, LatticeDecompositionPlan):
                raise TypeError("decomposition must be LatticeDecompositionPlan or None.")
            if (
                decomposition.global_shape != lattice_shape
                or decomposition.periodic != fermion_boundary.topology.periodic
                or link_space.num_edges
                != decomposition.site_count * decomposition.dimension
            ):
                raise ValueError(
                    "Distributed decomposition and fermion boundary topology disagree."
                )
            canonical_edges = np.arange(
                decomposition.site_count * decomposition.dimension,
                dtype=np.int32,
            ).reshape((decomposition.site_count, decomposition.dimension))
            if not np.array_equal(
                np.asarray(transport.forward_edges), canonical_edges
            ) or not np.all(np.asarray(transport.forward_orientations) == 1):
                raise ValueError(
                    "Distributed QCD requires canonical site-major positive link routes."
                )
            scope: RecipeExecutionScope = "decomposed-global-reference"
        else:
            scope = "single-domain-reference"
        self.link_space = link_space
        self.plaquettes = plaquettes
        self.fermion_boundary = fermion_boundary
        self.representation = representation
        self.transport = transport
        self.staples = staples
        self.decomposition = decomposition
        self.lattice_shape = lattice_shape
        self.execution_scope = scope
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "su3-qcd-gauge-geometry",
                "link_space": link_space.link_space_id,
                "plaquettes": plaquettes.boundary_plan_id,
                "fermion_boundary": fermion_boundary.plan_id,
                "representation": representation.representation_id,
                "transport": transport.plan_id,
                "staples": staples.plan_id,
                "decomposition": (
                    None if decomposition is None else decomposition.plan_id
                ),
                "execution_scope": scope,
            }
        )


def build_su3_gauge_geometry(
    link_space: MatrixGaugeLinkSpace,
    plaquettes: CellBoundaryPathPlan,
    fermion_boundary: LatticeBoundaryPhasePlan,
    forward_sites: ArrayLike,
    forward_edges: ArrayLike,
    forward_orientations: ArrayLike,
    /,
    *,
    decomposition: LatticeDecompositionPlan | None = None,
    maximum_staples_per_link: int = 64,
) -> SU3GaugeGeometry:
    """Build canonical native representation, covariant shifts, and staple plans."""
    if not isinstance(link_space, MatrixGaugeLinkSpace):
        raise TypeError("link_space must be MatrixGaugeLinkSpace.")
    if not isinstance(link_space.group, SpecialUnitaryGroup) or (
        link_space.group.dimension != 3
    ):
        raise TypeError("QCD geometry requires SpecialUnitaryGroup(3).")
    representation = FundamentalGaugeRepresentation(link_space.group)
    transport = GaugeCovariantShiftPlan(
        link_space,
        representation,
        forward_sites,
        forward_edges,
        forward_orientations,
        fermion_boundary,
    )
    staples = GaugeStaplePlan(
        link_space,
        plaquettes,
        maximum_staples_per_link=maximum_staples_per_link,
    )
    return SU3GaugeGeometry(
        link_space,
        plaquettes,
        fermion_boundary,
        transport,
        staples,
        decomposition=decomposition,
    )


class QuenchedSU3Recipe(StrictModule, NonTrainableState):
    """Pure Wilson SU(3) theory lowered to product-Haar HMC."""

    geometry: SU3GaugeGeometry
    measurements: MeasurementSchedule
    beta: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    leapfrog_steps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: SU3GaugeGeometry,
        measurements: MeasurementSchedule,
        /,
        *,
        beta: float,
        step_size: float,
        leapfrog_steps: int,
        divergence_threshold: float = 1000.0,
    ):
        if not isinstance(geometry, SU3GaugeGeometry):
            raise TypeError("geometry must be SU3GaugeGeometry.")
        if not isinstance(measurements, MeasurementSchedule):
            raise TypeError("measurements must be MeasurementSchedule.")
        beta_ = _finite_positive(beta, "beta")
        step = _finite_positive(step_size, "step_size")
        threshold = _finite_positive(divergence_threshold, "divergence_threshold")
        leapfrog = int(leapfrog_steps)
        if leapfrog <= 0:
            raise ValueError("leapfrog_steps must be positive.")
        self.geometry = geometry
        self.measurements = measurements
        self.beta = beta_
        self.step_size = step
        self.leapfrog_steps = leapfrog
        self.divergence_threshold = threshold
        self.recipe_id = canonical_fingerprint(
            {
                "kind": "quenched-su3-wilson-recipe",
                "geometry": geometry.geometry_id,
                "measurements": measurements.schedule_id,
                "beta": beta_,
                "step_size": step,
                "leapfrog_steps": leapfrog,
                "divergence_threshold": threshold,
                "reference_measure": "product-haar",
            }
        )


class PreparedQuenchedSU3Recipe(StrictModule, NonTrainableState):
    recipe: QuenchedSU3Recipe
    gauge_action: WilsonGaugeAction
    target: CompactGeometricTarget
    kernel: PreparedCompactGroupHamiltonianKernel
    distributed_gauge: PreparedDistributedGaugeTheory | None
    distributed_hmc: PreparedDistributedHMC | None
    prepared_id: str = eqx.field(static=True)


def prepare_quenched_su3_recipe(
    recipe: QuenchedSU3Recipe,
    initial_links: ArrayLike,
    /,
) -> PreparedQuenchedSU3Recipe:
    """Bind a quenched recipe to one finite SU(3) gauge field."""
    if not isinstance(recipe, QuenchedSU3Recipe):
        raise TypeError("recipe must be QuenchedSU3Recipe.")
    links = jnp.asarray(initial_links)
    if links.shape != recipe.geometry.link_space.configuration_shape:
        raise ValueError("initial_links shape does not match the recipe link space.")
    if not bool(np.asarray(recipe.geometry.link_space.contains(links))):
        raise ValueError("initial_links must be finite SU(3) group members.")
    action = WilsonGaugeAction(
        recipe.geometry.link_space,
        recipe.geometry.plaquettes,
        plaquette_couplings=recipe.beta,
    )
    decomposition = recipe.geometry.decomposition
    if decomposition is None:
        target = compact_geometric_target_from_lattice_action(action)
        kernel = prepare_compact_group_hamiltonian_kernel(
            target,
            step_size=recipe.step_size,
            leapfrog_steps=recipe.leapfrog_steps,
            divergence_threshold=recipe.divergence_threshold,
        )
        distributed_gauge = None
        distributed_hmc = None
    else:
        distributed_plan = DistributedGaugeTheoryPlan(decomposition, recipe.beta)
        distributed_gauge = distributed_plan.prepare()
        distributed_hmc = prepare_distributed_hmc(
            DistributedHMCPlan(
                step_size=recipe.step_size,
                leapfrog_steps=recipe.leapfrog_steps,
                divergence_threshold=recipe.divergence_threshold,
            ),
            distributed_gauge,
            links,
            term_id=f"{recipe.recipe_id}:distributed-quenched",
        )
        kernel = distributed_hmc.kernel
        target = kernel.target
    return PreparedQuenchedSU3Recipe(
        recipe=recipe,
        gauge_action=action,
        target=target,
        kernel=kernel,
        distributed_gauge=distributed_gauge,
        distributed_hmc=distributed_hmc,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-quenched-su3-recipe",
                "recipe": recipe.recipe_id,
                "action": action.action_id,
                "target": target.target_id,
                "kernel": kernel.kernel_id,
                "distributed_gauge": (
                    None if distributed_gauge is None else distributed_gauge.prepared_id
                ),
                "distributed_hmc": (
                    None if distributed_hmc is None else distributed_hmc.prepared_id
                ),
                "initial_links": array_tree_fingerprint(np.asarray(links)),
            }
        ),
    )


class WilsonCloverNf2Recipe(StrictModule, NonTrainableState):
    """Two degenerate Wilson or clover flavors with exact pseudofermion exponent."""

    geometry: SU3GaugeGeometry
    measurements: MeasurementSchedule
    fermion_resources: LatticeFermionResourcePolicy
    rhmc_resources: RHMCResourcePolicy
    solves: PseudofermionSolveRoles
    beta: float = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    wilson_parameter: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    clover_coefficient: float = eqx.field(static=True)
    variant: FermionVariant = eqx.field(static=True)
    spectral_lower: float = eqx.field(static=True)
    spectral_upper: float = eqx.field(static=True)
    spectral_evidence: SpectralEvidence = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    trajectory_steps: int = eqx.field(static=True)
    fermion_force_substeps: int = eqx.field(static=True)
    gauge_force_substeps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: SU3GaugeGeometry,
        measurements: MeasurementSchedule,
        /,
        *,
        beta: float,
        mass: float,
        variant: FermionVariant = "wilson",
        clover_coefficient: float = 0.0,
        wilson_parameter: float = 1.0,
        lattice_spacing: float = 1.0,
        spectral_lower: float,
        spectral_upper: float,
        spectral_evidence: SpectralEvidence = "asserted",
        step_size: float,
        trajectory_steps: int,
        fermion_force_substeps: int = 1,
        gauge_force_substeps: int = 2,
        divergence_threshold: float = 1000.0,
        fermion_resources: LatticeFermionResourcePolicy | None = None,
        rhmc_resources: RHMCResourcePolicy | None = None,
        solves: PseudofermionSolveRoles | None = None,
    ):
        if not isinstance(geometry, SU3GaugeGeometry):
            raise TypeError("geometry must be SU3GaugeGeometry.")
        if not isinstance(measurements, MeasurementSchedule):
            raise TypeError("measurements must be MeasurementSchedule.")
        if variant not in ("wilson", "clover"):
            raise ValueError("variant must be 'wilson' or 'clover'.")
        beta_ = _finite_positive(beta, "beta")
        mass_ = float(mass)
        clover = float(clover_coefficient)
        if not isfinite(mass_) or not isfinite(clover) or clover < 0.0:
            raise ValueError(
                "mass and clover_coefficient must be finite; clover is nonnegative."
            )
        if (variant == "wilson" and clover != 0.0) or (
            variant == "clover" and clover == 0.0
        ):
            raise ValueError(
                "Wilson requires zero clover_coefficient; clover requires a positive value."
            )
        wilson = _finite_positive(wilson_parameter, "wilson_parameter")
        spacing = _finite_positive(lattice_spacing, "lattice_spacing")
        lower, upper, evidence = _spectral_bounds(
            spectral_lower, spectral_upper, spectral_evidence
        )
        step = _finite_positive(step_size, "step_size")
        steps = int(trajectory_steps)
        fermion_substeps = int(fermion_force_substeps)
        gauge_substeps = int(gauge_force_substeps)
        threshold = _finite_positive(divergence_threshold, "divergence_threshold")
        if min(steps, fermion_substeps, gauge_substeps) <= 0:
            raise ValueError("RHMC trajectory and force substeps must be positive.")
        fermion_policy = (
            LatticeFermionResourcePolicy()
            if fermion_resources is None
            else fermion_resources
        )
        rhmc_policy = RHMCResourcePolicy() if rhmc_resources is None else rhmc_resources
        solve_roles = PseudofermionSolveRoles() if solves is None else solves
        if not isinstance(fermion_policy, LatticeFermionResourcePolicy):
            raise TypeError("fermion_resources must be LatticeFermionResourcePolicy.")
        if not isinstance(rhmc_policy, RHMCResourcePolicy):
            raise TypeError("rhmc_resources must be RHMCResourcePolicy.")
        if not isinstance(solve_roles, PseudofermionSolveRoles):
            raise TypeError("solves must be PseudofermionSolveRoles.")
        self.geometry = geometry
        self.measurements = measurements
        self.fermion_resources = fermion_policy
        self.rhmc_resources = rhmc_policy
        self.solves = solve_roles
        self.beta = beta_
        self.mass = mass_
        self.wilson_parameter = wilson
        self.lattice_spacing = spacing
        self.clover_coefficient = clover
        self.variant = variant
        self.spectral_lower = lower
        self.spectral_upper = upper
        self.spectral_evidence = evidence
        self.step_size = step
        self.trajectory_steps = steps
        self.fermion_force_substeps = fermion_substeps
        self.gauge_force_substeps = gauge_substeps
        self.divergence_threshold = threshold
        self.recipe_id = canonical_fingerprint(
            {
                "kind": "nf2-wilson-clover-rhmc-recipe",
                "geometry": geometry.geometry_id,
                "measurements": measurements.schedule_id,
                "beta": beta_,
                "mass": mass_,
                "variant": variant,
                "clover_coefficient": clover,
                "wilson_parameter": wilson,
                "lattice_spacing": spacing,
                "spectral_interval": [lower, upper],
                "spectral_evidence": evidence,
                "step_size": step,
                "trajectory_steps": steps,
                "force_substeps": [fermion_substeps, gauge_substeps],
                "divergence_threshold": threshold,
                "fermion_resources": fermion_policy.policy_id,
                "rhmc_resources": _rhmc_resource_payload(rhmc_policy),
                "solve_roles": solve_roles.roles_id,
            }
        )


class PreparedWilsonCloverNf2Recipe(StrictModule, NonTrainableState):
    recipe: WilsonCloverNf2Recipe
    gauge_action: WilsonGaugeAction
    dirac: AbstractLatticeDiracOperator
    spectral_interval: SpectralInterval
    pseudofermion: TwoFlavorPseudofermionTerm
    registry: SeparableActionRegistry
    rhmc_plan: RHMCPlan
    kernel: PreparedRHMCKernel
    distributed_gauge: PreparedDistributedGaugeTheory | None
    distributed_rhmc: PreparedDistributedRHMC | None
    prepared_id: str = eqx.field(static=True)


def prepare_wilson_clover_nf2_recipe(
    recipe: WilsonCloverNf2Recipe,
    initial_links: ArrayLike,
    /,
    *,
    clover_term: ArrayLike | None = None,
) -> PreparedWilsonCloverNf2Recipe:
    """Lower an Nf=2 plan to native Wilson/clover, pseudofermion, and RHMC values."""
    if not isinstance(recipe, WilsonCloverNf2Recipe):
        raise TypeError("recipe must be WilsonCloverNf2Recipe.")
    links = jnp.asarray(initial_links)
    if links.shape != recipe.geometry.link_space.configuration_shape:
        raise ValueError("initial_links shape does not match the recipe link space.")
    if not bool(np.asarray(recipe.geometry.link_space.contains(links))):
        raise ValueError("initial_links must be finite SU(3) group members.")
    gauge_action = WilsonGaugeAction(
        recipe.geometry.link_space,
        recipe.geometry.plaquettes,
        plaquette_couplings=recipe.beta,
    )
    wilson = WilsonDiracOperator(
        recipe.geometry.fermion_boundary,
        recipe.geometry.representation,
        recipe.geometry.transport,
        links,
        mass=recipe.mass,
        wilson_parameter=recipe.wilson_parameter,
        lattice_spacing=recipe.lattice_spacing,
        dtype=links.dtype,
        resources=recipe.fermion_resources,
    )
    if recipe.variant == "wilson":
        if clover_term is not None:
            raise ValueError("clover_term must be omitted for a Wilson recipe.")
        dirac: AbstractLatticeDiracOperator = wilson
    else:
        base_clover = (
            native_wilson_clover_term(wilson)
            if clover_term is None
            else jnp.asarray(clover_term, dtype=wilson.source.dtype)
        )
        scaled_clover = recipe.clover_coefficient * base_clover
        dirac = CloverWilsonDiracOperator(
            wilson,
            scaled_clover,
            resources=recipe.fermion_resources,
        )
    interval = _spectral_interval(
        dirac,
        recipe.spectral_lower,
        recipe.spectral_upper,
        recipe.spectral_evidence,
    )
    pseudofermion = TwoFlavorPseudofermionTerm(
        dirac,
        interval,
        solves=recipe.solves,
    )
    force_plan = NestedForcePlan(
        (
            NestedForcePartition((1,), substeps=recipe.fermion_force_substeps),
            NestedForcePartition((0,), substeps=recipe.gauge_force_substeps),
        )
    )
    rhmc_plan = plan_rhmc(
        step_size=recipe.step_size,
        trajectory_steps=recipe.trajectory_steps,
        force_plan=force_plan,
        divergence_threshold=recipe.divergence_threshold,
        resources=recipe.rhmc_resources,
    )
    decomposition = recipe.geometry.decomposition
    if decomposition is None:
        gauge_term = SeparableActionTerm(
            gauge_action.action,
            term_id=f"{gauge_action.action_id}:rhmc-gauge-term",
        )
        registry = SeparableActionRegistry((gauge_term,), (pseudofermion,))
        kernel = prepare_rhmc(
            registry,
            rhmc_plan,
            links,
            geometry=recipe.geometry.link_space.geometry,
            local_coordinate_shape=recipe.geometry.link_space.local_coordinate_shape,
        )
        distributed_gauge = None
        distributed_rhmc = None
    else:
        distributed_plan = DistributedGaugeTheoryPlan(decomposition, recipe.beta)
        distributed_gauge = distributed_plan.prepare()
        distributed_composition = DistributedRHMCPlan(distributed_plan, rhmc_plan)
        distributed_rhmc = prepare_distributed_rhmc(
            distributed_composition,
            distributed_gauge,
            links,
            pseudofermion_terms=(pseudofermion,),
        )
        kernel = distributed_rhmc.kernel
        registry = kernel.registry
    return PreparedWilsonCloverNf2Recipe(
        recipe=recipe,
        gauge_action=gauge_action,
        dirac=dirac,
        spectral_interval=interval,
        pseudofermion=pseudofermion,
        registry=registry,
        rhmc_plan=rhmc_plan,
        kernel=kernel,
        distributed_gauge=distributed_gauge,
        distributed_rhmc=distributed_rhmc,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-nf2-wilson-clover-recipe",
                "recipe": recipe.recipe_id,
                "gauge_action": gauge_action.action_id,
                "dirac": dirac.operator_id,
                "interval": interval.certificate_id,
                "pseudofermion": pseudofermion.term_id,
                "registry": registry.registry_id,
                "rhmc": rhmc_plan.plan_id,
                "kernel": kernel.kernel_id,
                "distributed_gauge": (
                    None if distributed_gauge is None else distributed_gauge.prepared_id
                ),
                "distributed_rhmc": (
                    None if distributed_rhmc is None else distributed_rhmc.prepared_id
                ),
            }
        ),
    )


class StaggeredHisqStyleRHMCRecipe(StrictModule, NonTrainableState):
    """Naik-improved staggered RHMC with explicit rooted determinant power."""

    geometry: SU3GaugeGeometry
    measurements: MeasurementSchedule
    fermion_resources: LatticeFermionResourcePolicy
    rhmc_resources: RHMCResourcePolicy
    solves: PseudofermionSolveRoles
    beta: float = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    lattice_spacing: float = eqx.field(static=True)
    one_link_coefficient: float = eqx.field(static=True)
    three_link_coefficient: float = eqx.field(static=True)
    determinant_power: float = eqx.field(static=True)
    link_improvement_id: str = eqx.field(static=True)
    spectral_lower: float = eqx.field(static=True)
    spectral_upper: float = eqx.field(static=True)
    spectral_evidence: SpectralEvidence = eqx.field(static=True)
    num_poles: int = eqx.field(static=True)
    verification_points: int = eqx.field(static=True)
    rational_tolerance: float = eqx.field(static=True)
    step_size: float = eqx.field(static=True)
    trajectory_steps: int = eqx.field(static=True)
    fermion_force_substeps: int = eqx.field(static=True)
    gauge_force_substeps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    recipe_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: SU3GaugeGeometry,
        measurements: MeasurementSchedule,
        /,
        *,
        beta: float,
        mass: float,
        flavor_count: int,
        link_improvement_id: str,
        lattice_spacing: float = 1.0,
        one_link_coefficient: float = 9.0 / 8.0,
        three_link_coefficient: float = -1.0 / 24.0,
        spectral_lower: float,
        spectral_upper: float,
        spectral_evidence: SpectralEvidence = "asserted",
        num_poles: int = 12,
        verification_points: int = 16_385,
        rational_tolerance: float = 1.0e-7,
        step_size: float,
        trajectory_steps: int,
        fermion_force_substeps: int = 1,
        gauge_force_substeps: int = 2,
        divergence_threshold: float = 1000.0,
        fermion_resources: LatticeFermionResourcePolicy | None = None,
        rhmc_resources: RHMCResourcePolicy | None = None,
        solves: PseudofermionSolveRoles | None = None,
    ):
        if not isinstance(geometry, SU3GaugeGeometry):
            raise TypeError("geometry must be SU3GaugeGeometry.")
        if not isinstance(measurements, MeasurementSchedule):
            raise TypeError("measurements must be MeasurementSchedule.")
        beta_ = _finite_positive(beta, "beta")
        mass_ = float(mass)
        spacing = _finite_positive(lattice_spacing, "lattice_spacing")
        first = float(one_link_coefficient)
        third = float(three_link_coefficient)
        flavors = int(flavor_count)
        improvement = _identifier(link_improvement_id, "link_improvement_id")
        if not all(isfinite(value) for value in (mass_, first, third)):
            raise ValueError("Staggered mass and Naik coefficients must be finite.")
        if flavors <= 0:
            raise ValueError("flavor_count must be positive.")
        determinant_power = flavors / 8.0
        lower, upper, evidence = _spectral_bounds(
            spectral_lower, spectral_upper, spectral_evidence
        )
        poles = int(num_poles)
        points = int(verification_points)
        tolerance = _finite_positive(rational_tolerance, "rational_tolerance")
        step = _finite_positive(step_size, "step_size")
        steps = int(trajectory_steps)
        fermion_substeps = int(fermion_force_substeps)
        gauge_substeps = int(gauge_force_substeps)
        threshold = _finite_positive(divergence_threshold, "divergence_threshold")
        if poles <= 0 or points < max(129, 16 * (poles + 2)):
            raise ValueError("Rational pole/verification resources are insufficient.")
        if min(steps, fermion_substeps, gauge_substeps) <= 0:
            raise ValueError("RHMC trajectory and force substeps must be positive.")
        fermion_policy = (
            LatticeFermionResourcePolicy()
            if fermion_resources is None
            else fermion_resources
        )
        rhmc_policy = RHMCResourcePolicy() if rhmc_resources is None else rhmc_resources
        solve_roles = PseudofermionSolveRoles() if solves is None else solves
        if not isinstance(fermion_policy, LatticeFermionResourcePolicy):
            raise TypeError("fermion_resources must be LatticeFermionResourcePolicy.")
        if not isinstance(rhmc_policy, RHMCResourcePolicy):
            raise TypeError("rhmc_resources must be RHMCResourcePolicy.")
        if not isinstance(solve_roles, PseudofermionSolveRoles):
            raise TypeError("solves must be PseudofermionSolveRoles.")
        self.geometry = geometry
        self.measurements = measurements
        self.fermion_resources = fermion_policy
        self.rhmc_resources = rhmc_policy
        self.solves = solve_roles
        self.beta = beta_
        self.mass = mass_
        self.lattice_spacing = spacing
        self.one_link_coefficient = first
        self.three_link_coefficient = third
        self.determinant_power = determinant_power
        self.link_improvement_id = improvement
        self.spectral_lower = lower
        self.spectral_upper = upper
        self.spectral_evidence = evidence
        self.num_poles = poles
        self.verification_points = points
        self.rational_tolerance = tolerance
        self.step_size = step
        self.trajectory_steps = steps
        self.fermion_force_substeps = fermion_substeps
        self.gauge_force_substeps = gauge_substeps
        self.divergence_threshold = threshold
        self.recipe_id = canonical_fingerprint(
            {
                "kind": "staggered-hisq-style-rhmc-recipe",
                "claim": "staggered-naik-with-explicit-improved-link-identity",
                "geometry": geometry.geometry_id,
                "measurements": measurements.schedule_id,
                "beta": beta_,
                "mass": mass_,
                "flavor_count": flavors,
                "normal-determinant-power": determinant_power,
                "link_improvement_id": improvement,
                "lattice_spacing": spacing,
                "one_link_coefficient": first,
                "three_link_coefficient": third,
                "spectral_interval": [lower, upper],
                "spectral_evidence": evidence,
                "num_poles": poles,
                "verification_points": points,
                "rational_tolerance": tolerance,
                "step_size": step,
                "trajectory_steps": steps,
                "force_substeps": [fermion_substeps, gauge_substeps],
                "divergence_threshold": threshold,
                "fermion_resources": fermion_policy.policy_id,
                "rhmc_resources": _rhmc_resource_payload(rhmc_policy),
                "solve_roles": solve_roles.roles_id,
            }
        )


class PreparedStaggeredHisqStyleRHMCRecipe(StrictModule, NonTrainableState):
    recipe: StaggeredHisqStyleRHMCRecipe
    gauge_action: WilsonGaugeAction
    dirac: NaikStaggeredDiracOperator
    spectral_interval: SpectralInterval
    action_approximation: CertifiedRationalApproximation
    refresh_approximation: CertifiedRationalApproximation
    pseudofermion: FractionalPowerPseudofermionTerm
    registry: SeparableActionRegistry
    rhmc_plan: RHMCPlan
    kernel: PreparedRHMCKernel
    distributed_gauge: PreparedDistributedGaugeTheory | None
    distributed_rhmc: PreparedDistributedRHMC | None
    prepared_id: str = eqx.field(static=True)


def prepare_staggered_hisq_style_rhmc_recipe(
    recipe: StaggeredHisqStyleRHMCRecipe,
    improved_links: ArrayLike,
    /,
) -> PreparedStaggeredHisqStyleRHMCRecipe:
    """Lower explicit improved links to Naik staggered fractional-power RHMC."""
    if not isinstance(recipe, StaggeredHisqStyleRHMCRecipe):
        raise TypeError("recipe must be StaggeredHisqStyleRHMCRecipe.")
    links = jnp.asarray(improved_links)
    if links.shape != recipe.geometry.link_space.configuration_shape:
        raise ValueError("improved_links shape does not match the recipe link space.")
    if not bool(np.asarray(recipe.geometry.link_space.contains(links))):
        raise ValueError("improved_links must be finite SU(3) group members.")
    gauge_action = WilsonGaugeAction(
        recipe.geometry.link_space,
        recipe.geometry.plaquettes,
        plaquette_couplings=recipe.beta,
    )
    one_link = StaggeredDiracOperator(
        recipe.geometry.fermion_boundary,
        recipe.geometry.representation,
        recipe.geometry.transport,
        links,
        mass=recipe.mass,
        lattice_spacing=recipe.lattice_spacing,
        dtype=links.dtype,
        gauge_field_id=recipe.link_improvement_id,
        resources=recipe.fermion_resources,
    )
    dirac = NaikStaggeredDiracOperator(
        one_link,
        one_link_coefficient=recipe.one_link_coefficient,
        three_link_coefficient=recipe.three_link_coefficient,
    )
    interval = _spectral_interval(
        dirac,
        recipe.spectral_lower,
        recipe.spectral_upper,
        recipe.spectral_evidence,
    )
    action_target = power_rational_target(-recipe.determinant_power)
    refresh_target = power_rational_target(0.5 * recipe.determinant_power)
    action_plan = plan_minimax_rational_approximation(
        action_target,
        num_poles=recipe.num_poles,
        verification_points=recipe.verification_points,
        requested_tolerance=recipe.rational_tolerance,
    )
    refresh_plan = plan_minimax_rational_approximation(
        refresh_target,
        num_poles=recipe.num_poles,
        verification_points=recipe.verification_points,
        requested_tolerance=recipe.rational_tolerance,
    )
    action_approximation = generate_minimax_rational_approximation(interval, action_plan)
    refresh_approximation = generate_minimax_rational_approximation(
        interval, refresh_plan
    )
    if not bool(
        np.asarray(action_approximation.successful)
        & np.asarray(refresh_approximation.successful)
    ):
        raise ValueError("RHMC rational approximations failed their requested tolerance.")
    pseudofermion = FractionalPowerPseudofermionTerm(
        dirac,
        interval,
        action_approximation,
        refresh_approximation,
        determinant_power=recipe.determinant_power,
        solves=recipe.solves,
    )
    force_plan = NestedForcePlan(
        (
            NestedForcePartition((1,), substeps=recipe.fermion_force_substeps),
            NestedForcePartition((0,), substeps=recipe.gauge_force_substeps),
        )
    )
    rhmc_plan = plan_rhmc(
        step_size=recipe.step_size,
        trajectory_steps=recipe.trajectory_steps,
        force_plan=force_plan,
        divergence_threshold=recipe.divergence_threshold,
        resources=recipe.rhmc_resources,
    )
    decomposition = recipe.geometry.decomposition
    if decomposition is None:
        gauge_term = SeparableActionTerm(
            gauge_action.action,
            term_id=f"{gauge_action.action_id}:rhmc-gauge-term",
        )
        registry = SeparableActionRegistry((gauge_term,), (pseudofermion,))
        kernel = prepare_rhmc(
            registry,
            rhmc_plan,
            links,
            geometry=recipe.geometry.link_space.geometry,
            local_coordinate_shape=recipe.geometry.link_space.local_coordinate_shape,
        )
        distributed_gauge = None
        distributed_rhmc = None
    else:
        distributed_plan = DistributedGaugeTheoryPlan(decomposition, recipe.beta)
        distributed_gauge = distributed_plan.prepare()
        distributed_composition = DistributedRHMCPlan(distributed_plan, rhmc_plan)
        distributed_rhmc = prepare_distributed_rhmc(
            distributed_composition,
            distributed_gauge,
            links,
            pseudofermion_terms=(pseudofermion,),
        )
        kernel = distributed_rhmc.kernel
        registry = kernel.registry
    return PreparedStaggeredHisqStyleRHMCRecipe(
        recipe=recipe,
        gauge_action=gauge_action,
        dirac=dirac,
        spectral_interval=interval,
        action_approximation=action_approximation,
        refresh_approximation=refresh_approximation,
        pseudofermion=pseudofermion,
        registry=registry,
        rhmc_plan=rhmc_plan,
        kernel=kernel,
        distributed_gauge=distributed_gauge,
        distributed_rhmc=distributed_rhmc,
        prepared_id=canonical_fingerprint(
            {
                "kind": "prepared-staggered-hisq-style-rhmc-recipe",
                "recipe": recipe.recipe_id,
                "gauge_action": gauge_action.action_id,
                "dirac": dirac.operator_id,
                "interval": interval.certificate_id,
                "action_approximation": action_approximation.certificate_id,
                "refresh_approximation": refresh_approximation.certificate_id,
                "pseudofermion": pseudofermion.term_id,
                "improved_links": array_tree_fingerprint(np.asarray(links)),
                "registry": registry.registry_id,
                "rhmc": rhmc_plan.plan_id,
                "kernel": kernel.kernel_id,
                "distributed_gauge": (
                    None if distributed_gauge is None else distributed_gauge.prepared_id
                ),
                "distributed_rhmc": (
                    None if distributed_rhmc is None else distributed_rhmc.prepared_id
                ),
            }
        ),
    )


__all__ = [
    "FermionVariant",
    "PreparedQuenchedSU3Recipe",
    "PreparedStaggeredHisqStyleRHMCRecipe",
    "PreparedWilsonCloverNf2Recipe",
    "QuenchedSU3Recipe",
    "RecipeExecutionScope",
    "SU3GaugeGeometry",
    "SpectralEvidence",
    "StaggeredHisqStyleRHMCRecipe",
    "WilsonCloverNf2Recipe",
    "build_su3_gauge_geometry",
    "prepare_quenched_su3_recipe",
    "prepare_staggered_hisq_style_rhmc_recipe",
    "prepare_wilson_clover_nf2_recipe",
]
