"""Rod admission, immutable FE preparation, and the coupled residual and root."""

from __future__ import annotations

import dataclasses
from math import isfinite
from typing import Final, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.operators.mechanics import finite_strain_kinematics

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...discretization.fem import PreparedFiniteElementPointInterpolation
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditionerBuilder,
    ArraySpace,
    BlockFactorizationPreconditionerBuilder,
    BlockSelection,
    BorderedBlockAssemblyPlan,
    BoundedLUPreconditionerBuilder,
    CoordinateBlock,
    GMRES,
    LinearDerivativeSolvePolicy,
    LinearizationPolicy,
    LinearSolvePolicy,
    LinearSystem,
    MappedBlockFactorizationPreconditionerBuilder,
    MatrixFreeLocalBlockPreconditionerBuilder,
    PreconditioningPolicy,
    prepare as prepare_linear,
    PreparedLinearSolve,
    RHSLayout,
    SolveResourcePolicy,
    StructuredDirect,
    TolerancePolicy,
)
from ...nonlinear import (
    fischer_burmeister_map,
    FixedPointIteration,
    FixedPointProblem,
    implicit_root_output,
    JacobianPolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ...nuclear import ExposureState, HeavyMetalReference
from ...typing import parse
from ._acceptance import fuel_step_evidence, FuelAcceptancePolicy, FuelStepEvidence
from ._bundles import FuelModelBundle
from ._cladding_deformation import ZircaloyCreepState
from ._cladding_properties import ZIRCALOY_DENSITY
from ._constants import MOLAR_GAS_CONSTANT_J_PER_MOL_K
from ._coolant import PrescribedWallTemperature, SingleChannelCoolant
from ._coupled_coolant import CoupledChannelCoolant
from ._forms import (
    deformation,
    evaluate_fuel_forms,
    evaluate_fuel_gas,
    evaluate_fuel_loading,
    FuelFormEvaluation,
    FuelStepLoading,
)
from ._free_volume import FreeVolumeModel
from ._fuel_creep import FuelCreepState
from ._fuel_properties import UO2_THEORETICAL_DENSITY
from ._gas_block import FuelGasBlockResult
from ._gas_state import FissionGasSpeciesTransfers, FissionGasTransfers
from ._geometry import fuel_rod_geometry, FuelRodGeometry
from ._identity import (
    AxialInteraction,
    FUEL_GAS_SPECIES,
    FuelGeometryReduction,
)
from ._inventory import FuelEnergyLedger, FuelGasLedger
from ._irradiation import IrradiationHistory
from ._layout import (
    AXIAL_STRAIN_SCALE,
    CHANNEL_ENTHALPY_SCALE_J_PER_KG,
    CONTACT_PRESSURE_SCALE_PA,
    COOLANT_HEAT_FLUX_SCALE_W_PER_M2,
    FREE_GAS_SCALE_MOL,
    FRICTION_FORCE_SCALE_N,
    GAS_PRESSURE_SCALE_PA,
    HYDROSTATIC_STRESS_SCALE_PA,
    TEMPERATURE_SCALE_K,
)
from ._numerical_views import FuelNumericalSpecification, NumericalFreeVolumeModel
from ._plenum import plenum_response
from ._site_state import (
    CladdingSiteState,
    FuelSiteGas,
    FuelSiteHelium,
    FuelSiteMechanics,
    FuelSiteState,
)
from ._specification import FuelRodSpecification
from ._state import FuelRodFields, FuelRodState
from ._zones import zone_average


# Cold temperature of the fabricated rod and of a shut-down rod [K].
FABRICATION_TEMPERATURE_K: Final[float] = 300.0
# Pellet density window shared by the reference conductivity and creep sources.
_ADMITTED_DENSITY_FRACTION: Final[tuple[float, float]] = (0.95, 0.97)
_MASS_FRACTION_PER_PPM: Final[float] = 1.0e-6
# Largest time step [s] of the representative first step that stages the
# nonlinear template.
_TEMPLATE_TIME_STEP_S: Final[float] = 1.0

# Linear and nonlinear solver policy; every value enters the prepared plan IDs.
_PROBE_CHUNK_SIZE: Final[int] = 8
_GMRES_MAXIMUM_RESTART: Final[int] = 64
_LINEAR_RELATIVE_TOLERANCE: Final[float] = 1.0e-8
_LINEAR_ABSOLUTE_TOLERANCE: Final[float] = 1.0e-12
_LINEAR_MAXIMUM_STEPS: Final[int] = 256
_NONLINEAR_ABSOLUTE_RESIDUAL: Final[float] = 1.0e-9
_NONLINEAR_RELATIVE_RESIDUAL: Final[float] = 1.0e-10
_NONLINEAR_MAXIMUM_STEPS: Final[int] = 40
# Coordinate JVP probing reports every omitted layer coupling; four
# dimension-scaled epsilons cover the measured AD roundoff only.
_LEAKAGE_ROUNDOFF_MULTIPLIER: Final[float] = 4.0

type _ThermomechanicalArguments = tuple[
    FuelRodState, Array, Array, FuelModelBundle, FuelGasBlockResult, FuelStepLoading
]
type _PartitionArguments = tuple[
    FuelRodState, Array, Array, FuelModelBundle, FuelStepLoading
]
type _JointArguments = tuple[FuelRodState, Array, Array, FuelModelBundle]


class FuelRodAdmission(StrictModule):
    """Specification, history, coolant, and bundle checked for one rod model."""

    specification: FuelRodSpecification | FuelNumericalSpecification
    history: IrradiationHistory
    coolant: PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant
    bundle: FuelModelBundle
    reduction: FuelGeometryReduction = eqx.field(static=True)
    axial_interaction: AxialInteraction = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)


def admit_fuel_rod(
    specification: FuelRodSpecification,
    history: IrradiationHistory,
    coolant: PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant,
    /,
    *,
    bundle: FuelModelBundle,
    reduction: FuelGeometryReduction = "layered-radial",
    axial_interaction: AxialInteraction = "frictionless",
) -> FuelRodAdmission:
    """Check that the inputs describe one admitted rod.

    History zones must span the fuel stack, the pellet density must lie in the
    admitted 95-97% of theoretical density, a prescribed wall table must match
    the zones and cover the history, a single channel needs positive mass flux,
    a coupled channel must use the same history, and Coulomb interaction needs
    a ``CoulombFriction`` closure.
    """
    if not isinstance(specification, FuelRodSpecification) or not isinstance(
        history, IrradiationHistory
    ):
        raise TypeError("Admission requires fuel specification and irradiation history.")
    if not isinstance(bundle, FuelModelBundle) or not isinstance(
        coolant, (PrescribedWallTemperature, SingleChannelCoolant, CoupledChannelCoolant)
    ):
        raise TypeError("Admission requires source-bound bundle and supported coolant.")
    reduction = parse(reduction, FuelGeometryReduction, "reduction")
    axial_interaction = parse(axial_interaction, AxialInteraction, "axial_interaction")
    if not np.isclose(np.asarray(history.zone_edges)[-1], specification.rod.stack_length):
        raise ValueError("History zones must cover exactly the fabricated stack.")
    lowest_density, highest_density = _ADMITTED_DENSITY_FRACTION
    if not lowest_density <= specification.pellet.density_fraction <= highest_density:
        raise ValueError(
            "Reference conductivity/creep source intersection requires 95–97% TD."
        )
    if isinstance(coolant, PrescribedWallTemperature):
        if coolant.wall_temperatures.shape[1] != history.zone_count:
            raise ValueError("Wall table and irradiation zones must match by identity.")
        if (
            np.asarray(coolant.times)[0] > 0
            or np.asarray(coolant.times)[-1] < history.end_time
        ):
            raise ValueError(
                "Wall table must cover the entire declared irradiation history."
            )
        coolant_id = coolant.table_id
    elif isinstance(coolant, CoupledChannelCoolant):
        if coolant.history.history_id != history.history_id:
            raise ValueError("Qualified external channel and rod histories must match.")
        coolant_id = coolant.participant_id
    else:
        if np.any(np.asarray(history.coolant_mass_flux) <= 0.0):
            raise ValueError(
                "A single-channel coolant requires a positive coolant_mass_flux at "
                "every history time."
            )
        coolant_id = coolant.law_id
    if axial_interaction == "coulomb" and bundle.friction is None:
        raise ValueError("Coulomb interaction requires source-bound friction.")
    identity = canonical_fingerprint(
        {
            "kind": "admitted-reference-fuel-rod",
            "specification": specification.specification_id,
            "history": history.history_id,
            "coolant": coolant_id,
            "bundle": bundle.bundle_id,
            "reduction": reduction,
            "axial_interaction": axial_interaction,
        }
    )
    return FuelRodAdmission(
        specification=specification,
        history=history,
        coolant=coolant,
        bundle=bundle,
        reduction=reduction,
        axial_interaction=axial_interaction,
        admission_id=identity,
    )


def _split_joint_endpoint(point: Array, field_count: int, /) -> tuple[Array, Array]:
    """Field unknowns and fuel-site hydrostatic stresses [Pa] of a joint endpoint."""
    return point[:field_count], point[field_count:] * HYDROSTATIC_STRESS_SCALE_PA


def _pack_joint_endpoint(unknown: Array, hydrostatic_stress: Array, /) -> Array:
    """Joint endpoint coordinates of field unknowns and hydrostatic stresses [Pa]."""
    return jnp.concatenate((unknown, hydrostatic_stress / HYDROSTATIC_STRESS_SCALE_PA))


def _stop_gradient_inexact[T](tree: T, /) -> T:
    """Stop gradients through every inexact array leaf; other leaves pass through."""
    return jax.tree.map(
        lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_inexact_array(leaf) else leaf,
        tree,
    )


class FuelResidualOperator(StrictModule):
    """Prepared geometry, solver coordinates, and residual callbacks of one rod.

    Solver unknowns are the ``FuelRodFields`` leaves flattened field-major and
    divided by ``unknown_scale``.
    """

    admission: FuelRodAdmission
    geometry: FuelRodGeometry
    heavy_metal: HeavyMetalReference
    free_volume: FreeVolumeModel | NumericalFreeVolumeModel
    unknown_scale: Array
    bounds: Bounds
    linear_space: ArraySpace
    bordered_assembly: BorderedBlockAssemblyPlan | None
    joint_space: ArraySpace
    joint_bordered_assembly: BorderedBlockAssemblyPlan | None
    field_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    field_offsets: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def transform_jacobian(
        self, operator: AbstractLinearOperator, /
    ) -> tuple[AbstractLinearOperator, int]:
        plan = self.bordered_assembly
        if plan is None:
            raise RuntimeError("A layered Jacobian requires its bordered block assembly.")
        return plan.assemble(operator), plan.setup_matvec_count

    def flatten_fields(self, fields: FuelRodFields, /) -> Array:
        return jnp.concatenate(
            tuple(leaf.reshape(-1) for leaf in jax.tree.leaves(fields))
        )

    def fields(self, unknown: Array, /) -> FuelRodFields:
        physical = unknown * self.unknown_scale
        values = tuple(
            physical[start:stop].reshape(shape)
            for start, stop, shape in zip(
                self.field_offsets[:-1],
                self.field_offsets[1:],
                self.field_shapes,
                strict=True,
            )
        )
        return FuelRodFields(*values)

    def evaluate(
        self,
        unknown: Array,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        bundle: FuelModelBundle,
        frozen_gas: FuelGasBlockResult,
        loading: FuelStepLoading,
        /,
    ) -> FuelFormEvaluation:
        return evaluate_fuel_forms(
            self, self.fields(unknown), previous, time, dt, bundle, frozen_gas, loading
        )

    def residual_with_aux(
        self, unknown: Array, args: _ThermomechanicalArguments, /
    ) -> tuple[Array, FuelFormEvaluation]:
        """Thermomechanical rows at frozen gas, complementarity-mapped at the bounds."""
        previous, time, dt, bundle, frozen_gas, loading = args
        evaluation = self.evaluate(
            unknown, previous, time, dt, bundle, frozen_gas, loading
        )
        physical_rows = self.flatten_fields(evaluation.residual_fields)
        residual = fischer_burmeister_map(unknown, physical_rows, self.bounds)
        if not isinstance(residual, Array):
            raise RuntimeError("The rod complementarity map must return one array.")
        return residual, evaluation

    def joint_residual_with_aux(
        self, point: Array, args: _JointArguments, /
    ) -> tuple[Array, FuelFormEvaluation]:
        """Coupled endpoint equations with live loading and gas, for implicit derivatives.

        Unlike ``residual_with_aux`` nothing is frozen: loading and gas are
        recomputed from the arguments, and the hydrostatic-stress rows close
        the gas feedback.
        """
        if point.shape != self.joint_space.shape:
            raise ValueError(
                "Joint endpoint coordinates do not match the prepared field/site space."
            )
        previous, time, dt, bundle = args
        unknown, hydrostatic_stress = _split_joint_endpoint(
            point, self.unknown_scale.shape[0]
        )
        loading = evaluate_fuel_loading(self, previous, time, dt, bundle)
        gas = evaluate_fuel_gas(
            self, self.fields(unknown), hydrostatic_stress, previous, dt, bundle, loading
        )
        return self.joint_residual_at_gas(
            point, (previous, time, dt, bundle, gas, loading)
        )

    def joint_residual_at_gas(
        self, point: Array, args: _ThermomechanicalArguments, /
    ) -> tuple[Array, FuelFormEvaluation]:
        """Coupled endpoint equations at gas and loading already evaluated at ``point``.

        With the gas of ``point`` these are exactly the rows of
        ``joint_residual_with_aux``, in the same scaled solver coordinates.
        """
        field_count = self.unknown_scale.shape[0]
        residual, evaluation = self.residual_with_aux(point[:field_count], args)
        hydrostatic_rows = (
            point[field_count:]
            - evaluation.candidate.fuel_sites.mechanics.hydrostatic_stress
            / HYDROSTATIC_STRESS_SCALE_PA
        )
        return jnp.concatenate((residual, hydrostatic_rows)), evaluation

    def trial_valid(
        self, unknown: Array, _arguments: _ThermomechanicalArguments, /
    ) -> Array:
        """Whether a Newton trial is finite, positive, and orientation preserving.

        A refused trial is a solver event, not a scientific domain exit of an
        accepted state.
        """
        fields = self.fields(unknown)
        fuel_kinematics = finite_strain_kinematics(
            deformation(
                self.geometry.fuel,
                fields.displacement_fuel,
                fields.axial_strain_fuel,
            )
        )
        cladding_kinematics = finite_strain_kinematics(
            deformation(
                self.geometry.cladding,
                fields.displacement_cladding,
                fields.axial_strain_cladding,
            )
        )
        return (
            jnp.all(jnp.isfinite(unknown))
            & jnp.all(fields.temperature_fuel > 0)
            & jnp.all(fields.temperature_cladding > 0)
            & (fields.pressure > 0)
            & jnp.all(fuel_kinematics.admissible)
            & jnp.all(cladding_kinematics.admissible)
            & jnp.all(fields.free_moles >= 0)
            & (jnp.sum(fields.free_moles) > 0)
        )


@final
class FuelPartitionedMapEvaluation(StrictModule):
    evaluation: FuelFormEvaluation
    thermomechanical_result: NonlinearResult
    frozen_gas: FuelGasBlockResult


@final
class FuelRodCandidateResult(StrictModule):
    """One partitioned step endpoint with its thermomechanical and coupling results.

    ``joint_residual`` is the live coupled residual at the endpoint in scaled
    solver coordinates (see ``FuelResidualOperator.joint_residual_with_aux``).
    """

    state: Array
    evaluation: FuelFormEvaluation
    thermomechanical_result: NonlinearResult
    coupling_result: NonlinearResult
    joint_residual: Array
    coupling_gas_admissible: Array
    thermomechanical_residual_evaluations: Array
    thermomechanical_iterations: Array
    gas_block_evaluations: Array
    verified_gas: FuelGasBlockResult


class _PartitionedEndpoint(NamedTuple):
    """Outputs of the converged partitioned map, with checked types."""

    mapped: FuelPartitionedMapEvaluation
    thermomechanical_unknown: Array
    thermomechanical_residual_evaluations: Array
    thermomechanical_iterations: Array
    gas_block_evaluations: Array


class _EndpointVerification(NamedTuple):
    """Endpoint gas and the live coupled residual at the partitioned endpoint.

    ``joint_residual`` holds the scaled rows of ``joint_residual_with_aux`` at
    ``point``: thermomechanics with the gas re-evaluated at the endpoint, and
    the hydrostatic-stress closure.
    """

    gas: FuelGasBlockResult
    point: Array
    joint_residual: Array
    admissible: Array


def _partitioned_endpoint(coupling: NonlinearResult, /) -> _PartitionedEndpoint:
    """Unpack the fixed-point result into the values the coupling map produced."""
    mapped = coupling.auxiliary
    if not isinstance(mapped, FuelPartitionedMapEvaluation):
        raise TypeError("The fixed-point result lost its endpoint block auxiliary.")
    thermomechanical_unknown = mapped.thermomechanical_result.state
    if not isinstance(thermomechanical_unknown, Array):
        raise TypeError("Thermomechanical solution must be one coordinate array.")
    work = coupling.evaluation_work
    if not isinstance(work, dict):
        raise TypeError("The coupling map must report its block work.")
    residual_evaluations = work["thermomechanical_residual_evaluations"]
    gas_block_evaluations = work["gas_block_evaluations"]
    iterations = work["thermomechanical_iterations"]
    if (
        not isinstance(residual_evaluations, Array)
        or not isinstance(gas_block_evaluations, Array)
        or not isinstance(iterations, Array)
    ):
        raise TypeError("Coupling block work must be reported as arrays.")
    return _PartitionedEndpoint(
        mapped=mapped,
        thermomechanical_unknown=thermomechanical_unknown,
        thermomechanical_residual_evaluations=residual_evaluations,
        thermomechanical_iterations=iterations,
        gas_block_evaluations=gas_block_evaluations,
    )


class PreparedFuelRod(StrictModule):
    """Geometry, solver coordinates, and nonlinear solvers of one admitted rod."""

    kernel: FuelResidualOperator
    nonlinear_problem: NonlinearSystemProblem
    nonlinear_template: PreparedNonlinearSolve
    joint_linear_policy: LinearSolvePolicy
    coupling_method: FixedPointIteration
    coupling_termination: NonlinearTermination
    coupling_tolerance: float = eqx.field(static=True)
    execution_signature: tuple[str, int, int, int, int, int, int, str] = eqx.field(
        static=True
    )

    @property
    def admission(self) -> FuelRodAdmission:
        return self.kernel.admission

    @property
    def geometry(self) -> FuelRodGeometry:
        return self.kernel.geometry

    @property
    def heavy_metal(self) -> HeavyMetalReference:
        return self.kernel.heavy_metal

    @property
    def free_volume(self) -> FreeVolumeModel | NumericalFreeVolumeModel:
        return self.kernel.free_volume

    @property
    def prepared_id(self) -> str:
        return self.kernel.prepared_id

    def initial_state(
        self,
        /,
        *,
        parameters: FuelModelBundle | None = None,
        fill_pressure_scale: ArrayLike = 1.0,
    ) -> FuelRodState:
        bundle = self.admission.bundle if parameters is None else parameters
        scale = jnp.asarray(fill_pressure_scale, dtype=jnp.float64)
        if scale.ndim != 0:
            raise ValueError("fill_pressure_scale must be a scalar.")
        scale = eqx.error_if(
            scale,
            ~jnp.isfinite(scale) | (scale <= 0.0),
            "fill_pressure_scale must be finite and positive.",
        )
        return _fabricated_state(self.kernel, bundle, scale)

    def pack_unknown(self, state: FuelRodState, /) -> Array:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Fuel state belongs to a different prepared FE identity.")
        return self.kernel.flatten_fields(state.fields) / self.kernel.unknown_scale

    def residual(
        self,
        unknown: Array,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        frozen_gas: FuelGasBlockResult,
        /,
        *,
        parameters: FuelModelBundle | None = None,
    ) -> Array:
        bundle = self.admission.bundle if parameters is None else parameters
        loading = evaluate_fuel_loading(self.kernel, previous, time, dt, bundle)
        residual = self.nonlinear_problem.residual(
            unknown, (previous, time, dt, bundle, frozen_gas, loading)
        )
        if not isinstance(residual, Array):
            raise RuntimeError("The coupled rod residual must be one coordinate array.")
        return residual

    def candidate_evaluation(
        self, result: FuelRodCandidateResult, /
    ) -> FuelFormEvaluation:
        if not isinstance(result, FuelRodCandidateResult):
            raise TypeError("A fuel candidate must be a FuelRodCandidateResult.")
        evaluation = result.evaluation
        if evaluation.candidate.prepared_id != self.prepared_id:
            raise ValueError("Final fuel evaluation belongs to another prepared rod.")
        return evaluation

    def solve_candidate(
        self,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        /,
        *,
        parameters: FuelModelBundle | None = None,
    ) -> FuelRodCandidateResult:
        """Solve one partitioned backward-Euler step and attach its implicit derivative.

        The primal solve sees stopped-gradient copies of the step arguments;
        derivatives with respect to them come only from the joint endpoint
        residual through the implicit-function theorem.
        """
        bundle = self.admission.bundle if parameters is None else parameters
        active_arguments = (previous, time, dt, bundle)
        primal_previous, primal_time, primal_dt, primal_bundle = _stop_gradient_inexact(
            active_arguments
        )
        # Loading depends on the accepted history, not on the endpoint guess.
        # The joint residual recomputes it from the live arguments.
        primal_loading = evaluate_fuel_loading(
            self.kernel, primal_previous, primal_time, primal_dt, primal_bundle
        )
        coupling, endpoint = self._solve_partitioned_endpoint(
            primal_previous, primal_time, primal_dt, primal_bundle, primal_loading
        )
        verification = self._verify_endpoint(
            endpoint,
            primal_previous,
            primal_time,
            primal_dt,
            primal_bundle,
            primal_loading,
        )
        unknown, evaluation = self._attach_implicit_output(
            active_arguments, coupling, endpoint, verification
        )
        return FuelRodCandidateResult(
            state=unknown,
            evaluation=evaluation,
            thermomechanical_result=endpoint.mapped.thermomechanical_result,
            coupling_result=coupling,
            joint_residual=verification.joint_residual,
            coupling_gas_admissible=verification.admissible,
            # The endpoint verification adds one thermomechanical residual
            # evaluation and one gas-block evaluation.
            thermomechanical_residual_evaluations=(
                endpoint.thermomechanical_residual_evaluations + 1
            ),
            thermomechanical_iterations=endpoint.thermomechanical_iterations,
            gas_block_evaluations=endpoint.gas_block_evaluations + 1,
            verified_gas=verification.gas,
        )

    def _solve_partitioned_endpoint(
        self,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        bundle: FuelModelBundle,
        loading: FuelStepLoading,
        /,
    ) -> tuple[NonlinearResult, _PartitionedEndpoint]:
        """Fixed-point iteration of gas then thermomechanics, from the accepted state."""
        problem = FixedPointProblem(
            self.partitioned_mapping,
            problem_id=f"{self.prepared_id}/endpoint-partition",
            has_aux=True,
            evaluation_work=True,
        )
        coupling = self.coupling_method.solve(
            problem,
            self.pack_joint(previous),
            termination=self.coupling_termination,
            args=(previous, time, dt, bundle, loading),
        )
        return coupling, _partitioned_endpoint(coupling)

    def _verify_endpoint(
        self,
        endpoint: _PartitionedEndpoint,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        bundle: FuelModelBundle,
        loading: FuelStepLoading,
        /,
    ) -> _EndpointVerification:
        """Re-evaluate gas at the endpoint and the live coupled residual there.

        The certificate is the residual of the coupled system whose root the
        implicit derivative linearizes, in the scaled coordinates of the
        coupling termination, so it is independent of how many gas-state
        values the gas block carries.
        """
        candidate = endpoint.mapped.evaluation.candidate
        hydrostatic_stress = candidate.fuel_sites.mechanics.hydrostatic_stress
        verified_gas = evaluate_fuel_gas(
            self.kernel,
            candidate.fields,
            hydrostatic_stress,
            previous,
            dt,
            bundle,
            loading,
        )
        point = _pack_joint_endpoint(
            endpoint.thermomechanical_unknown, hydrostatic_stress
        )
        joint_residual, _ = self.kernel.joint_residual_at_gas(
            point, (previous, time, dt, bundle, verified_gas, loading)
        )
        return _EndpointVerification(
            gas=verified_gas,
            point=point,
            joint_residual=joint_residual,
            admissible=jnp.all(verified_gas.gas.admissible)
            & jnp.all(verified_gas.helium.admissible)
            & jnp.all(verified_gas.species.admissible),
        )

    def _attach_implicit_output(
        self,
        active_arguments: _JointArguments,
        coupling: NonlinearResult,
        endpoint: _PartitionedEndpoint,
        verification: _EndpointVerification,
        /,
    ) -> tuple[Array, FuelFormEvaluation]:
        """Field unknowns and evaluation with the joint implicit derivative attached.

        Primal values are the stopped endpoint and evaluation; the derivative
        is attached only when the whole step converged.
        """
        mapped = endpoint.mapped
        evaluation = mapped.evaluation
        joint_residual = verification.joint_residual
        derivative_successful = (
            mapped.thermomechanical_result.successful
            & coupling.successful
            & verification.admissible
            & evaluation.local_converged
            & evaluation.physical
            & evaluation.in_range
            & jnp.all(jnp.isfinite(joint_residual))
            & (jnp.linalg.norm(joint_residual) <= self.coupling_tolerance)
        )
        primal_endpoint = jax.lax.stop_gradient(verification.point)
        attached_endpoint, attached_evaluation = implicit_root_output(
            primal_endpoint,
            active_arguments,
            _stop_gradient_inexact(evaluation),
            derivative_successful,
            residual_and_output=self.kernel.joint_residual_with_aux,
            root_space=self.kernel.joint_space,
            prepare_linear_solve=self.prepare_joint_solve,
        )
        if not isinstance(attached_evaluation, FuelFormEvaluation):
            raise TypeError(
                "The implicit output attachment must return the fuel form evaluation."
            )
        return attached_endpoint[
            : self.kernel.unknown_scale.shape[0]
        ], attached_evaluation

    def partitioned_mapping(
        self, point: Array, args: _PartitionArguments, /
    ) -> tuple[Array, FuelPartitionedMapEvaluation, dict[str, Array]]:
        """One coupling pass: gas at ``point``, then thermomechanics at that gas."""
        previous, time, dt, bundle, loading = args
        unknown, hydrostatic_stress = _split_joint_endpoint(
            point, self.kernel.unknown_scale.shape[0]
        )
        frozen_gas = evaluate_fuel_gas(
            self.kernel,
            self.kernel.fields(unknown),
            hydrostatic_stress,
            previous,
            dt,
            bundle,
            loading,
        )
        refreshed = refresh_nonlinear(
            self.nonlinear_template,
            self.nonlinear_problem,
            unknown,
            args=(previous, time, dt, bundle, frozen_gas, loading),
        )
        result = solve_prepared_nonlinear(refreshed)
        evaluation = result.auxiliary
        if not isinstance(evaluation, FuelFormEvaluation):
            raise TypeError("The thermomechanical solve lost its final form evaluation.")
        solution = result.state
        if not isinstance(solution, Array):
            raise TypeError("Thermomechanical solution must be one coordinate array.")
        mapped = _pack_joint_endpoint(
            solution, evaluation.candidate.fuel_sites.mechanics.hydrostatic_stress
        )
        work = {
            "thermomechanical_residual_evaluations": result.diagnostics.residual_evaluations,
            "thermomechanical_iterations": result.diagnostics.iterations,
            "gas_block_evaluations": jnp.asarray(1, dtype=jnp.int32),
        }
        return (
            mapped,
            FuelPartitionedMapEvaluation(
                evaluation=evaluation,
                thermomechanical_result=result,
                frozen_gas=frozen_gas,
            ),
            work,
        )

    def candidate_evidence(
        self,
        result: FuelRodCandidateResult,
        previous: FuelRodState,
        time: Array,
        dt: Array,
        step_index: Array,
        /,
        *,
        parameters: FuelModelBundle | None = None,
        policy: FuelAcceptancePolicy | None = None,
    ) -> FuelStepEvidence:
        return fuel_step_evidence(
            self,
            result,
            previous,
            time,
            dt,
            step_index,
            parameters=parameters,
            policy=policy,
        )

    def pack_joint(self, state: FuelRodState, /) -> Array:
        return _pack_joint_endpoint(
            self.pack_unknown(state), state.fuel_sites.mechanics.hydrostatic_stress
        )

    def prepare_joint_solve(
        self, operator: AbstractLinearOperator, /, *, rhs_count: int = 1
    ) -> PreparedLinearSolve:
        """Prepare the joint derivative solve on the prepared joint endpoint space."""
        if rhs_count < 1:
            raise ValueError("rhs_count must be positive.")
        space = self.kernel.joint_space
        if (
            operator.source.space_id != space.space_id
            or operator.target.space_id != space.space_id
            or not operator.source.compatible(space)
            or not operator.target.compatible(space)
        ):
            raise ValueError(
                "Joint derivative source and target must be the prepared endpoint space."
            )
        assembly = self.kernel.joint_bordered_assembly
        assembled = operator if assembly is None else assembly.assemble(operator)
        return prepare_linear(
            LinearSystem(assembled, problem_id=f"{self.prepared_id}/joint-derivative"),
            self.joint_linear_policy,
            rhs_layout=RHSLayout((rhs_count,)),
        )

    def branch_signature(
        self, state: FuelRodState, /, *, parameters: FuelModelBundle | None = None
    ) -> Array:
        """Active branches: contact, stick, gas saturation, HBS, cladding plasticity.

        Two states with different signatures lie on different smooth branches
        of the step map.
        """
        bundle = self.admission.bundle if parameters is None else parameters
        in_contact = state.fields.contact_pressure > 0
        geometry = self.geometry
        fuel_interface = geometry.fuel_interface
        contact_count = geometry.interface_lengths.shape[0]
        interface_radius = (
            fuel_interface.reference_radius
            + fuel_interface.values(state.fields.displacement_fuel)[:, 0]
        )
        radius = zone_average(
            interface_radius,
            fuel_interface.weight,
            fuel_interface.interface_index,
            contact_count,
        )
        if geometry.layered:
            axial_stretch = 1 + state.fields.axial_strain_fuel[geometry.interface_zone]
        else:
            tangent = (
                fuel_interface.axial_gradient(state.fields.displacement_fuel)
                .at[:, 1]
                .add(1)
            )
            axial_stretch = zone_average(
                jnp.linalg.norm(tangent, axis=-1),
                fuel_interface.weight,
                fuel_interface.interface_index,
                contact_count,
            )
        # Coulomb limit of the friction force on each contact zone [N]; a
        # frictionless bundle has a zero limit.
        coefficient = 0.0 if bundle.friction is None else bundle.friction.coefficient
        stick_limit = (
            coefficient
            * state.fields.contact_pressure
            * 2
            * jnp.pi
            * radius
            * geometry.interface_lengths
            * axial_stretch
        )
        stick = jnp.abs(state.fields.friction_force) < stick_limit
        gas = state.fuel_sites.gas.total
        diagnostics = state.fuel_sites.gas.diagnostics
        saturated = state.fuel_sites.constitutive_evaluated & (
            diagnostics.fractional_coverage >= diagnostics.saturation_coverage
        )
        return jnp.concatenate(
            (
                in_contact,
                stick,
                saturated,
                gas.hbs_fraction > 0,
                state.cladding_sites.plastic_active,
            )
        ).astype(jnp.int32)

    def is_cooled(
        self,
        state: FuelRodState,
        /,
        *,
        target_temperature: ArrayLike = FABRICATION_TEMPERATURE_K,
        tolerance: ArrayLike = 1.0,
    ) -> Array:
        """Whether the rod is unpowered at ``target_temperature`` within ``tolerance``."""
        return (
            jnp.all(
                jnp.abs(state.fields.temperature_fuel - target_temperature) <= tolerance
            )
            & jnp.all(
                jnp.abs(state.fields.temperature_cladding - target_temperature)
                <= tolerance
            )
            & jnp.all(self.admission.history.conditions(state.time).linear_power == 0)
        )

    def surface_oxide(self, state: FuelRodState, /) -> Array:
        """Zone-mean oxide thickness [m] at every cladding outer-surface point."""
        geometry = self.geometry
        zone_oxide = zone_average(
            state.cladding_sites.oxide_thickness,
            geometry.cladding.site_weight,
            geometry.cladding.site_zone,
            geometry.layer_lengths.shape[0],
        )
        return zone_oxide[geometry.cladding_outer.zone]


def _filled_record[T: StrictModule](record_type: type[T], value: Array, /) -> T:
    """``record_type`` with every field set to ``value``."""
    return record_type(**{field.name: value for field in dataclasses.fields(record_type)})


def _fabricated_fields(
    kernel: FuelResidualOperator, fill_scale: Array, /
) -> FuelRodFields:
    """Cold, unloaded fields holding the fabricated fill gas."""
    geometry, specification = kernel.geometry, kernel.admission.specification
    fuel_node_count = geometry.fuel.node_count
    cladding_node_count = geometry.cladding.node_count
    zone_count = geometry.layer_lengths.shape[0]
    component_count = 1 if geometry.layered else 2
    axial_count = zone_count if geometry.layered else 0
    contact_count = geometry.interface_lengths.shape[0]
    coolant = kernel.admission.coolant
    fields = FuelRodFields(
        temperature_fuel=jnp.full(
            (fuel_node_count,), FABRICATION_TEMPERATURE_K, dtype=jnp.float64
        ),
        temperature_cladding=jnp.full(
            (cladding_node_count,), FABRICATION_TEMPERATURE_K, dtype=jnp.float64
        ),
        displacement_fuel=jnp.zeros(
            (fuel_node_count, component_count), dtype=jnp.float64
        ),
        displacement_cladding=jnp.zeros(
            (cladding_node_count, component_count), dtype=jnp.float64
        ),
        axial_strain_fuel=jnp.zeros((axial_count,), dtype=jnp.float64),
        axial_strain_cladding=jnp.zeros((axial_count,), dtype=jnp.float64),
        contact_pressure=jnp.zeros((contact_count,), dtype=jnp.float64),
        friction_force=jnp.zeros((contact_count,), dtype=jnp.float64),
        # Fill gas cooled at constant volume to the fabrication temperature.
        pressure=jnp.asarray(
            specification.rod.fill_pressure
            * FABRICATION_TEMPERATURE_K
            / specification.rod.fill_temperature
        ),
        coolant_heat_flux=jnp.zeros(
            (geometry.coolant_nodes.shape[0],), dtype=jnp.float64
        ),
        free_moles=jnp.ones((len(FUEL_GAS_SPECIES),), dtype=jnp.float64),
        channel_state=coolant.initial_state
        if isinstance(coolant, CoupledChannelCoolant)
        else jnp.zeros((0,), dtype=jnp.float64),
    )
    zero_zones = jnp.zeros((zone_count,))
    volumes = plenum_response(
        specification,
        geometry,
        kernel.free_volume,
        fields,
        jnp.full((zone_count,), FABRICATION_TEMPERATURE_K),
        zero_zones,
        zero_zones,
        zero_zones,
        jnp.asarray(FABRICATION_TEMPERATURE_K),
        zero_zones,
        jnp.ones((zone_count,)),
    ).components
    fill_moles = (
        fill_scale
        * specification.rod.fill_pressure
        * volumes.total_volume
        / (MOLAR_GAS_CONSTANT_J_PER_MOL_K * specification.rod.fill_temperature)
        * jnp.asarray(specification.rod.fill_gas_mole_fractions)
    )
    return eqx.tree_at(
        lambda value: (value.free_moles, value.pressure),
        fields,
        (fill_moles, fill_scale * fields.pressure),
    )


def _fabricated_fuel_sites(
    geometry: FuelRodGeometry,
    specification: FuelRodSpecification | FuelNumericalSpecification,
    bundle: FuelModelBundle,
    /,
) -> FuelSiteState:
    """Fresh, unstressed fuel histories at every fuel site."""
    site_count = geometry.fuel.site_count
    zero = jnp.zeros((site_count,), dtype=jnp.float64)
    zero_tensor = jnp.zeros((site_count, 3, 3), dtype=jnp.float64)
    zero_status = jnp.zeros((site_count,), dtype=jnp.int32)
    unset = jnp.zeros_like(zero, dtype=jnp.bool_)
    return FuelSiteState(
        mechanics=FuelSiteMechanics(
            creep=FuelCreepState(
                previous_equivalent_stress=zero, time_since_stress_increase=zero
            ),
            creep_log_strain=zero_tensor,
            densification_strain=zero,
            permanent_relocation_strain=zero,
            hydrostatic_stress=zero,
            cauchy_stress=jnp.zeros((site_count, 3, 3), dtype=jnp.float64),
        ),
        gas=FuelSiteGas(
            total=bundle.fission_gas.initial_state(
                (site_count,), grain_radius=specification.pellet.grain_radius
            ),
            transfers=_filled_record(FissionGasTransfers, zero),
            diagnostics=bundle.fission_gas.initial_diagnostics((site_count,)),
            species=bundle.fission_gas.initial_species_state((site_count,)),
            species_transfers=_filled_record(
                FissionGasSpeciesTransfers,
                jnp.zeros((site_count, 2), dtype=jnp.float64),
            ),
            retained_species=jnp.zeros(
                (site_count, len(FUEL_GAS_SPECIES)), dtype=jnp.float64
            ),
            species_linear_status=zero_status,
            species_linear_condition_estimate=zero,
        ),
        helium=FuelSiteHelium(
            state=bundle.helium.initial_state(
                (site_count,), grain_radius=specification.pellet.grain_radius
            ),
            released_increment=zero,
            diffusion_to_boundary=zero,
            grain_growth_sweeping=zero,
            radial_cell_width=jnp.full_like(
                zero,
                specification.pellet.grain_radius / bundle.helium.radial_cell_count,
            ),
            linear_status=zero_status,
            linear_condition_estimate=zero,
            linear_conditioning_available=jnp.zeros_like(zero, dtype=jnp.bool_),
        ),
        maximum_temperature=jnp.full_like(zero, FABRICATION_TEMPERATURE_K),
        constitutive_evaluated=unset,
    )


def _fabricated_cladding_sites(
    geometry: FuelRodGeometry,
    specification: FuelRodSpecification | FuelNumericalSpecification,
    bundle: FuelModelBundle,
    /,
) -> CladdingSiteState:
    """Fresh cladding histories with the fabricated oxide and hydrogen."""
    site_count = geometry.cladding.site_count
    cladding = specification.cladding
    zero = jnp.zeros((site_count,), dtype=jnp.float64)
    # Fabricated hydrogen mass per outer-surface area [kg/m^2] in the metal
    # left under the initial oxide.
    initial_hydrogen_areal_mass = (
        cladding.initial_hydrogen_ppm
        * _MASS_FRACTION_PER_PPM
        * ZIRCALOY_DENSITY
        * (
            (
                cladding.outer_radius
                - bundle.corrosion.metal_thickness_consumed(
                    cladding.initial_oxide_thickness
                )
            )
            ** 2
            - cladding.inner_radius**2
        )
        / (2 * cladding.outer_radius)
    )
    return CladdingSiteState(
        creep=ZircaloyCreepState.initialize(
            jnp.zeros((site_count, 3, 3), dtype=jnp.float64)
        ),
        inelastic_log_strain=jnp.zeros((site_count, 3, 3), dtype=jnp.float64),
        equivalent_plastic_strain=zero,
        fast_fluence=zero,
        oxide_thickness=jnp.full_like(zero, cladding.initial_oxide_thickness),
        hydrogen_ppm=jnp.full_like(zero, cladding.initial_hydrogen_ppm),
        maximum_temperature=jnp.full_like(zero, FABRICATION_TEMPERATURE_K),
        hydrogen_areal_mass=jnp.full_like(zero, initial_hydrogen_areal_mass),
        cauchy_stress=jnp.zeros((site_count, 3, 3), dtype=jnp.float64),
        plastic_active=jnp.zeros_like(zero, dtype=jnp.bool_),
        constitutive_evaluated=jnp.zeros_like(zero, dtype=jnp.bool_),
    )


def _fabricated_state(
    kernel: FuelResidualOperator, bundle: FuelModelBundle, fill_scale: Array, /
) -> FuelRodState:
    """Cold fabricated rod at time zero with ``fill_scale`` times the fill gas."""
    geometry, specification = kernel.geometry, kernel.admission.specification
    zone_count = geometry.layer_lengths.shape[0]
    fuel_site_count = geometry.fuel.site_count
    zero = jnp.asarray(0.0, dtype=jnp.float64)
    fields = _fabricated_fields(kernel, fill_scale)
    fuel_sites = _fabricated_fuel_sites(geometry, specification, bundle)
    cladding_sites = _fabricated_cladding_sites(geometry, specification, bundle)
    # Fuel sites are stored layer by layer with equal counts per layer.
    sites_per_zone = fuel_site_count // zone_count
    radii = geometry.fuel.site_coordinates[:, 0].reshape((zone_count, sites_per_zone))
    weights = (
        geometry.fuel.site_weight.reshape((zone_count, sites_per_zone))
        / geometry.layer_lengths[:, None]
    )
    radial_power = bundle.radial_power.initial_state(
        radii,
        weights=weights,
        outer_radius=specification.pellet.radius,
        heavy_metal=kernel.heavy_metal,
    )
    zero_species = jnp.zeros((len(FUEL_GAS_SPECIES),), dtype=jnp.float64)
    return FuelRodState(
        time=zero,
        fields=fields,
        fuel_sites=fuel_sites,
        cladding_sites=cladding_sites,
        exposure=ExposureState.zeros((fuel_site_count,)),
        radial_power=radial_power,
        gas=FuelGasLedger(
            fill_moles=fields.free_moles,
            produced_moles=zero_species,
            retained_moles=zero_species,
            released_moles=zero_species,
            exported_moles=zero_species,
            balance_error=zero_species,
        ),
        energy=_filled_record(FuelEnergyLedger, zero),
        prepared_id=kernel.prepared_id,
    )


@eqx.filter_jit
def _prepare_fuel_nonlinear(
    problem: NonlinearSystemProblem,
    unknown: Array,
    method: NewtonKrylov,
    termination: NonlinearTermination,
    args: _ThermomechanicalArguments,
    /,
) -> PreparedNonlinearSolve:
    """Stage the initial linearization with rod and material arrays as dynamic inputs."""
    return prepare_nonlinear(
        problem, unknown, method=method, termination=termination, args=args
    )


class _FieldLayout(NamedTuple):
    """One ``FuelRodFields`` entry of the flat solver coordinates."""

    name: str
    shape: tuple[int, ...]
    unknown_scale: float  # physical value of one solver unit
    lower_bound: float  # complementarity lower bound of the physical value


class _FieldCoordinates(NamedTuple):
    """Field-major flat solver coordinates derived from the field layout."""

    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    offsets: tuple[int, ...]
    unknown_scale: Array
    lower_bound: Array

    @property
    def size(self) -> int:
        return self.offsets[-1]

    def start(self, name: str, /) -> int:
        return self.offsets[self.names.index(name)]


def _field_layout(
    geometry: FuelRodGeometry,
    specification: FuelRodSpecification,
    coolant: PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant,
    /,
) -> tuple[_FieldLayout, ...]:
    """Shape, unknown scale, and lower bound of every field, in field order.

    A layered rod has radial displacement and one axial strain per layer; an
    RZ rod has two displacement components and no axial-strain unknowns.
    """
    zone_count = geometry.layer_lengths.shape[0]
    fuel_node_count = geometry.fuel.node_count
    cladding_node_count = geometry.cladding.node_count
    component_count, axial_count = (1, zone_count) if geometry.layered else (2, 0)
    contact_count = geometry.interface_lengths.shape[0]
    channel_shape = (
        coolant.initial_state.shape
        if isinstance(coolant, CoupledChannelCoolant)
        else (0,)
    )
    unbounded = -np.inf
    layout = (
        _FieldLayout(
            name="temperature_fuel",
            shape=(fuel_node_count,),
            unknown_scale=TEMPERATURE_SCALE_K,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="temperature_cladding",
            shape=(cladding_node_count,),
            unknown_scale=TEMPERATURE_SCALE_K,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="displacement_fuel",
            shape=(fuel_node_count, component_count),
            unknown_scale=specification.pellet.radius,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="displacement_cladding",
            shape=(cladding_node_count, component_count),
            unknown_scale=specification.cladding.outer_radius,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="axial_strain_fuel",
            shape=(axial_count,),
            unknown_scale=AXIAL_STRAIN_SCALE,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="axial_strain_cladding",
            shape=(axial_count,),
            unknown_scale=AXIAL_STRAIN_SCALE,
            lower_bound=unbounded,
        ),
        # Contact pressure is the only bounded unknown: it is complementary to
        # the contact gap row.
        _FieldLayout(
            name="contact_pressure",
            shape=(contact_count,),
            unknown_scale=CONTACT_PRESSURE_SCALE_PA,
            lower_bound=0.0,
        ),
        _FieldLayout(
            name="friction_force",
            shape=(contact_count,),
            unknown_scale=FRICTION_FORCE_SCALE_N,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="pressure",
            shape=(),
            unknown_scale=GAS_PRESSURE_SCALE_PA,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="coolant_heat_flux",
            shape=(geometry.coolant_nodes.shape[0],),
            unknown_scale=COOLANT_HEAT_FLUX_SCALE_W_PER_M2,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="free_moles",
            shape=(len(FUEL_GAS_SPECIES),),
            unknown_scale=FREE_GAS_SCALE_MOL,
            lower_bound=unbounded,
        ),
        _FieldLayout(
            name="channel_state",
            shape=channel_shape,
            unknown_scale=CHANNEL_ENTHALPY_SCALE_J_PER_KG,
            lower_bound=unbounded,
        ),
    )
    if tuple(entry.name for entry in layout) != tuple(
        field.name for field in dataclasses.fields(FuelRodFields)
    ):
        raise RuntimeError("The solver field layout must follow FuelRodFields order.")
    return layout


def _field_coordinates(layout: tuple[_FieldLayout, ...], /) -> _FieldCoordinates:
    sizes = tuple(np.prod(entry.shape, dtype=np.int64).item() for entry in layout)
    offsets = tuple(np.cumsum((0,) + sizes, dtype=np.int64).tolist())

    def expand(values: tuple[float, ...], /) -> Array:
        return jnp.asarray(
            np.concatenate(
                tuple(
                    np.full((size,), value, dtype=np.float64)
                    for size, value in zip(sizes, values, strict=True)
                )
            )
        )

    return _FieldCoordinates(
        names=tuple(entry.name for entry in layout),
        shapes=tuple(entry.shape for entry in layout),
        offsets=offsets,
        unknown_scale=expand(tuple(entry.unknown_scale for entry in layout)),
        lower_bound=expand(tuple(entry.lower_bound for entry in layout)),
    )


def _normalized_coordinate_identity(
    geometry: FuelRodGeometry,
    layout: tuple[_FieldLayout, ...],
    reduction: FuelGeometryReduction,
    /,
) -> str:
    """Identity of the solver coordinate layout, independent of rod identity."""
    body_roles = []
    for name, body in (("fuel", geometry.fuel), ("cladding", geometry.cladding)):
        route = body.value_route
        if not isinstance(route, PreparedFiniteElementPointInterpolation):
            raise TypeError(
                "Coordinate preparation requires a finite-element point route."
            )
        field_roles = tuple(
            (
                field.name,
                field.representation,
                field.conformity,
                None if field.form_type is None else field.form_type.to_dict(),
            )
            for field in route.discretization.field_spaces
            if field.name == route.field_name
        )
        if len(field_roles) != 1:
            raise ValueError("Normalized coordinates require one explicit FE field role.")
        body_roles.append(
            (
                name,
                route.discretization.mesh.topology.topology_id,
                field_roles,
                np.asarray(route.dof_routes),
                np.asarray(body.site_zone),
            )
        )
    return canonical_fingerprint(
        {
            "kind": "fuel-normalized-coordinate-layout",
            "reduction": reduction,
            "field_owner": (FuelRodFields.__module__, FuelRodFields.__qualname__),
            "field_coordinates": tuple((entry.name, entry.shape) for entry in layout),
            "gas_ledger_owner": (FuelGasLedger.__module__, FuelGasLedger.__qualname__),
            "packing": "field-major-native-fe-node-order",
            "bodies": tuple(body_roles),
            "joint_site_owner": (FuelSiteState.__module__, FuelSiteState.__qualname__),
            "joint_site_field": "mechanics.hydrostatic_stress",
            "joint_site_order": "fuel-native-quadrature-order",
            "interface_zone": np.asarray(geometry.interface_zone),
            "coolant_nodes": np.asarray(geometry.coolant_nodes),
            "dtype": np.dtype(np.float64).str,
        }
    )


def _prepared_identity(
    admission: FuelRodAdmission,
    geometry: FuelRodGeometry,
    coordinate_identity: str,
    coupling_budget: int,
    coupling_tolerance: float,
    resources: SolveResourcePolicy,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "admission": admission.admission_id,
            "geometry": geometry.geometry_id,
            "method": "total-lagrangian-partitioned-implicit-backward-euler-vi",
            "linear_route": "layered-banded-bordered-direct"
            if geometry.layered
            else "rz-matrix-free-block-schur",
            "linearization": "stored-native-jvp",
            "candidate_evaluation": "native-block-and-fixed-point-final-auxiliary",
            "coupling": (
                "native-fixed-point",
                coupling_budget,
                coupling_tolerance,
                "original-accepted-history",
                "endpoint-joint-residual",
                "symmetric-relative-gas-feedback",
            ),
            "joint_derivative": "endpoint-fields-and-native-fuel-site-hydrostatic",
            "implicit_output": "native-joint-root-and-primal-receipt",
            "helium_spatial_closure": "native-exact-shell-spherical-helmholtz",
            "normalized_coordinate_layout": coordinate_identity,
            "linear_resources": (
                resources.factorization_bytes,
                resources.workspace_bytes,
                resources.krylov_basis_bytes,
                resources.preconditioner_bytes,
                resources.recycling_state_bytes,
            ),
        }
    )


def _coordinate_block(
    space: ArraySpace, positions: tuple[int, ...], identity: str, /
) -> CoordinateBlock:
    member = ArraySpace(
        (len(positions),), dtype=jnp.float64, space_id=f"{space.space_id}/{identity}"
    )
    return CoordinateBlock(member, tuple((index, index + 1) for index in positions))


def _linear_tolerance() -> TolerancePolicy:
    return TolerancePolicy(
        relative=_LINEAR_RELATIVE_TOLERANCE,
        absolute=_LINEAR_ABSOLUTE_TOLERANCE,
        max_steps=_LINEAR_MAXIMUM_STEPS,
    )


def _layered_assembly(
    geometry: FuelRodGeometry,
    coordinates: _FieldCoordinates,
    space: ArraySpace,
    resources: SolveResourcePolicy,
    /,
    *,
    include_hydrostatic: bool = False,
) -> BorderedBlockAssemblyPlan:
    """Banded per-layer blocks bordered by the coordinates that couple layers.

    Each radial layer owns its temperatures, displacements and (for the joint
    space) hydrostatic sites; contact, friction, gas and coolant unknowns form
    the communicating border.
    """
    members: list[tuple[str, CoordinateBlock]] = []
    private: list[int] = []
    fuel_node_count = geometry.fuel.radial_node_count
    cladding_node_count = geometry.cladding.radial_node_count
    fuel_temperature = coordinates.start("temperature_fuel")
    cladding_temperature = coordinates.start("temperature_cladding")
    fuel_displacement = coordinates.start("displacement_fuel")
    cladding_displacement = coordinates.start("displacement_cladding")
    hydrostatic = coordinates.size
    site_groups: list[list[int]] = [[] for _ in range(fuel_node_count)]
    if include_hydrostatic:
        routes = np.asarray(geometry.fuel.value_route.dof_routes, dtype=np.int32)
        if routes.ndim != 2 or routes.shape[1] != 2:
            raise ValueError(
                "Layered hydrostatic coordinates require P1 interval support routes."
            )
        for site, support in enumerate(routes.tolist()):
            left, right = sorted(support)
            if left < 0 or right >= fuel_node_count or right - left != 1:
                raise ValueError(
                    "Each hydrostatic site must lie on one adjacent radial-node pair."
                )
            site_groups[left].append(site)
        sites_per_layer = routes.shape[0]
    else:
        sites_per_layer = 0
    for layer in range(geometry.layer_lengths.shape[0]):
        positions: list[int] = []
        for node in range(fuel_node_count):
            positions.extend(
                (
                    fuel_temperature + layer * fuel_node_count + node,
                    fuel_displacement + layer * fuel_node_count + node,
                )
            )
            positions.extend(
                hydrostatic + layer * sites_per_layer + site for site in site_groups[node]
            )
        for node in range(cladding_node_count):
            # The outer temperature controls oxide metal loss throughout a
            # layer, so it couples nonlocally and joins the border.
            if node != cladding_node_count - 1:
                positions.append(
                    cladding_temperature + layer * cladding_node_count + node
                )
            positions.append(cladding_displacement + layer * cladding_node_count + node)
        private.extend(positions)
        name = f"radial-layer:{layer}"
        members.append((name, _coordinate_block(space, tuple(positions), name)))
    private_set = set(private)
    border = tuple(index for index in range(space.size) if index not in private_set)
    members.append(
        ("communicating-border", _coordinate_block(space, border, "communicating-border"))
    )
    selection = BlockSelection(space, tuple(members))
    local_size = selection.target.spaces[0].size
    # Zone-averaged fuel/cladding conductivity enters the interface gap law, so
    # every thermal coordinate of a layer couples to every other one. Interleaved
    # displacement and hydrostatic-site coordinates sit between them in local
    # order; the exact admissible band is therefore the complete layer block.
    bandwidth = local_size - 1
    return BorderedBlockAssemblyPlan(
        selection,
        local_bandwidth=(bandwidth, bandwidth),
        verify_local_independence=True,
        probe_chunk_size=_PROBE_CHUNK_SIZE,
        max_local_size=local_size,
        max_border_size=len(border),
        max_workspace_bytes=resources.workspace_bytes,
        max_output_bytes=resources.factorization_bytes,
        leakage_roundoff_multiplier=_LEAKAGE_ROUNDOFF_MULTIPLIER,
    )


def _rz_linear_policy(
    coordinates: _FieldCoordinates,
    space: ArraySpace,
    resources: SolveResourcePolicy,
    /,
    *,
    hydrostatic_count: int = 0,
) -> LinearSolvePolicy:
    """Right-preconditioned GMRES with a block-Schur preconditioner.

    Temperatures and displacements (and hydrostatic sites) form the field
    block, preconditioned node by node; every other unknown forms a border
    factored by bounded LU.
    """
    displacement_stop = coordinates.start("axial_strain_fuel")
    thermal_stop = coordinates.start("displacement_fuel")
    hydrostatic = coordinates.size
    field_positions = tuple(range(displacement_stop)) + tuple(
        range(hydrostatic, hydrostatic + hydrostatic_count)
    )
    field_block = _coordinate_block(space, field_positions, "fields")
    border_block = _coordinate_block(
        space, tuple(range(displacement_stop, hydrostatic)), "border"
    )
    global_selection = BlockSelection(
        space, (("fields", field_block), ("border", border_block))
    )
    field_space = field_block.space
    if not isinstance(field_space, ArraySpace):
        raise RuntimeError("Fuel field coordinates require one array space.")
    thermal = _coordinate_block(field_space, tuple(range(thermal_stop)), "thermal")
    mechanical = _coordinate_block(
        field_space, tuple(range(thermal_stop, field_space.size)), "mechanical"
    )
    field_selection = BlockSelection(
        field_space, (("thermal", thermal), ("mechanical", mechanical))
    )
    thermal_space, mechanical_space = thermal.space, mechanical.space
    if not isinstance(thermal_space, ArraySpace) or not isinstance(
        mechanical_space, ArraySpace
    ):
        raise RuntimeError("Fuel block coordinates require array spaces.")
    thermal_nodes = BlockSelection(
        thermal_space,
        tuple(
            (
                f"thermal-node:{node}",
                _coordinate_block(thermal_space, (node,), f"node:{node}"),
            )
            for node in range(thermal_space.size)
        ),
    )
    displacement_count = displacement_stop - thermal_stop
    if hydrostatic_count:
        displacement = _coordinate_block(
            mechanical_space, tuple(range(displacement_count)), "displacement"
        )
        hydrostatic_block = _coordinate_block(
            mechanical_space,
            tuple(range(displacement_count, mechanical_space.size)),
            "hydrostatic",
        )
        mechanical_selection = BlockSelection(
            mechanical_space,
            (("displacement", displacement), ("hydrostatic", hydrostatic_block)),
        )
        displacement_space = displacement.space
        hydrostatic_space = hydrostatic_block.space
        if not isinstance(displacement_space, ArraySpace) or not isinstance(
            hydrostatic_space, ArraySpace
        ):
            raise RuntimeError("Joint mechanical subblocks require array spaces.")
    else:
        displacement_space = mechanical_space
        hydrostatic_space = None
        mechanical_selection = None
    mechanical_nodes = BlockSelection(
        displacement_space,
        tuple(
            (
                f"displacement-node:{node // 2}",
                _coordinate_block(
                    displacement_space, (node, node + 1), f"node:{node // 2}"
                ),
            )
            for node in range(0, displacement_count, 2)
        ),
    )
    thermal_solver = MatrixFreeLocalBlockPreconditionerBuilder(
        thermal_nodes,
        local_size=1,
        probe_chunk_size=_PROBE_CHUNK_SIZE,
        max_workspace_bytes=resources.workspace_bytes,
        max_output_bytes=resources.preconditioner_bytes,
    )
    mechanical_solver: AbstractPreconditionerBuilder = (
        MatrixFreeLocalBlockPreconditionerBuilder(
            mechanical_nodes,
            local_size=2,
            probe_chunk_size=_PROBE_CHUNK_SIZE,
            max_workspace_bytes=resources.workspace_bytes,
            max_output_bytes=resources.preconditioner_bytes,
        )
    )
    if hydrostatic_space is not None and mechanical_selection is not None:
        hydrostatic_sites = BlockSelection(
            hydrostatic_space,
            tuple(
                (
                    f"hydrostatic-site:{site}",
                    _coordinate_block(hydrostatic_space, (site,), f"site:{site}"),
                )
                for site in range(hydrostatic_space.size)
            ),
        )
        hydrostatic_solver = MatrixFreeLocalBlockPreconditionerBuilder(
            hydrostatic_sites,
            local_size=1,
            probe_chunk_size=_PROBE_CHUNK_SIZE,
            max_workspace_bytes=resources.workspace_bytes,
            max_output_bytes=resources.preconditioner_bytes,
        )
        mechanical_solver = MappedBlockFactorizationPreconditionerBuilder(
            mechanical_selection,
            BlockFactorizationPreconditionerBuilder(
                mechanical_solver, hydrostatic_solver, "diagonal"
            ),
        )
    field_solver = MappedBlockFactorizationPreconditionerBuilder(
        field_selection,
        BlockFactorizationPreconditionerBuilder(
            thermal_solver, mechanical_solver, "diagonal"
        ),
    )
    border_solver = BoundedLUPreconditionerBuilder(
        max_size=border_block.space.size,
        probe_chunk_size=_PROBE_CHUNK_SIZE,
        max_workspace_bytes=resources.workspace_bytes,
        max_output_bytes=resources.preconditioner_bytes,
    )
    builder = MappedBlockFactorizationPreconditionerBuilder(
        global_selection,
        BlockFactorizationPreconditionerBuilder(field_solver, border_solver, "ldu"),
    )
    return LinearSolvePolicy(
        GMRES(restart=min(_GMRES_MAXIMUM_RESTART, space.size)),
        tolerance=_linear_tolerance(),
        preconditioning=PreconditioningPolicy(builder, side="right"),
        resources=resources,
    )


class _LinearRoutes(NamedTuple):
    """Coordinate spaces and linear solve routes of the field and joint systems."""

    space: ArraySpace
    assembly: BorderedBlockAssemblyPlan | None
    policy: LinearSolvePolicy
    joint_space: ArraySpace
    joint_assembly: BorderedBlockAssemblyPlan | None
    joint_policy: LinearSolvePolicy


def _linear_routes(
    geometry: FuelRodGeometry,
    coordinates: _FieldCoordinates,
    coordinate_identity: str,
    resources: SolveResourcePolicy,
    /,
) -> _LinearRoutes:
    """Banded bordered direct solves for a layered rod, block-Schur GMRES for RZ."""
    space = ArraySpace(
        (coordinates.size,),
        dtype=jnp.float64,
        space_id=f"{coordinate_identity}/coupled-coordinates",
    )
    joint_space = ArraySpace(
        (coordinates.size + geometry.fuel.site_count,),
        dtype=jnp.float64,
        space_id=f"{coordinate_identity}/joint-endpoint-coordinates",
    )
    if geometry.layered:
        policy = LinearSolvePolicy(
            StructuredDirect(),
            resources=resources,
            tolerance=_linear_tolerance(),
            derivative_solve=LinearDerivativeSolvePolicy(route="primal-factors"),
        )
        return _LinearRoutes(
            space=space,
            assembly=_layered_assembly(geometry, coordinates, space, resources),
            policy=policy,
            joint_space=joint_space,
            joint_assembly=_layered_assembly(
                geometry, coordinates, joint_space, resources, include_hydrostatic=True
            ),
            joint_policy=policy,
        )
    return _LinearRoutes(
        space=space,
        assembly=None,
        policy=_rz_linear_policy(coordinates, space, resources),
        joint_space=joint_space,
        joint_assembly=None,
        joint_policy=_rz_linear_policy(
            coordinates,
            joint_space,
            resources,
            hydrostatic_count=geometry.fuel.site_count,
        ),
    )


def _nonlinear_method(
    kernel: FuelResidualOperator, routes: _LinearRoutes, /
) -> tuple[NonlinearSystemProblem, NewtonKrylov]:
    """Thermomechanical problem at frozen gas and its Newton-Krylov method."""
    assembly = routes.assembly
    problem = NonlinearSystemProblem(
        kernel.residual_with_aux,
        trial_validity=kernel.trial_valid,
        trial_validity_id="positive-temperature-gas-orientation-trials",
        problem_id=kernel.prepared_id,
        state_space=routes.space,
        residual_space=routes.space,
        has_aux=True,
    )
    jacobian_policy = JacobianPolicy(
        linearization_policy=LinearizationPolicy("store"),
        operator_transform=kernel.transform_jacobian if assembly is not None else None,
        operator_transform_id=None if assembly is None else assembly.plan_id,
    )
    method = NewtonKrylov(jacobian_policy=jacobian_policy, linear_policy=routes.policy)
    return problem, method


def _stage_nonlinear_template(
    kernel: FuelResidualOperator,
    problem: NonlinearSystemProblem,
    method: NewtonKrylov,
    termination: NonlinearTermination,
    /,
) -> PreparedNonlinearSolve:
    """Stage the template at the fabricated rod over a representative first step."""
    admission = kernel.admission
    initial = _fabricated_state(kernel, admission.bundle, jnp.asarray(1.0))
    initial_dt = jnp.asarray(min(_TEMPLATE_TIME_STEP_S, admission.history.end_time))
    initial_time = jnp.asarray(0.0, dtype=initial_dt.dtype)
    initial_loading = evaluate_fuel_loading(
        kernel, initial, initial_time, initial_dt, admission.bundle
    )
    initial_gas = evaluate_fuel_gas(
        kernel,
        initial.fields,
        initial.fuel_sites.mechanics.hydrostatic_stress,
        initial,
        initial_dt,
        admission.bundle,
        initial_loading,
    )
    return _prepare_fuel_nonlinear(
        problem,
        kernel.flatten_fields(initial.fields) / kernel.unknown_scale,
        method,
        termination,
        (
            initial,
            initial_time,
            initial_dt,
            admission.bundle,
            initial_gas,
            initial_loading,
        ),
    )


def prepare_fuel_rod(
    admission: FuelRodAdmission,
    /,
    *,
    fuel_cells: int = 16,
    cladding_cells: int = 4,
    quadrature_order: int = 2,
    axial_cells_per_zone: int = 1,
    nonlinear_termination: NonlinearTermination | None = None,
    linear_resources: SolveResourcePolicy | None = None,
    maximum_coupling_evaluations: int = 4,
    coupling_tolerance: float = 1e-9,
) -> PreparedFuelRod:
    """Build the finite-element geometry, solver coordinates, and solvers of a rod.

    ``fuel_cells`` and ``cladding_cells`` are radial cells per body;
    ``axial_cells_per_zone`` refines the axisymmetric mesh. The gas and
    thermomechanical blocks are coupled by at most
    ``maximum_coupling_evaluations`` fixed-point evaluations, and a step's
    joint residual must not exceed ``coupling_tolerance``.
    """
    if not isinstance(admission, FuelRodAdmission):
        raise TypeError("prepare_fuel_rod requires an admitted rod.")
    coupling_budget = int(maximum_coupling_evaluations)
    coupling_tolerance = float(coupling_tolerance)
    if coupling_budget < 1 or not isfinite(coupling_tolerance) or coupling_tolerance <= 0:
        raise ValueError(
            "Partitioned coupling requires a positive finite tolerance and evaluation budget."
        )
    specification = admission.specification
    if not isinstance(specification, FuelRodSpecification):
        raise TypeError(
            "Numerical execution views cannot be re-admitted as fabricated designs."
        )
    resources = SolveResourcePolicy() if linear_resources is None else linear_resources
    if not isinstance(resources, SolveResourcePolicy):
        raise TypeError("linear_resources must be SolveResourcePolicy or None.")
    geometry = fuel_rod_geometry(
        specification,
        admission.history.zone_edges,
        layered=admission.reduction == "layered-radial",
        fuel_cells=fuel_cells,
        cladding_cells=cladding_cells,
        quadrature_order=quadrature_order,
        axial_cells_per_zone=axial_cells_per_zone,
    )
    heavy_metal = HeavyMetalReference.oxide_fuel(
        fuel_density=UO2_THEORETICAL_DENSITY * specification.pellet.density_fraction,
        uranium_235_weight_fraction=specification.pellet.uranium_235_weight_fraction,
        oxygen_to_metal_ratio=specification.pellet.oxygen_to_metal_ratio,
    )
    layout = _field_layout(geometry, specification, admission.coolant)
    coordinates = _field_coordinates(layout)
    coordinate_identity = _normalized_coordinate_identity(
        geometry, layout, admission.reduction
    )
    prepared_id = _prepared_identity(
        admission,
        geometry,
        coordinate_identity,
        coupling_budget,
        coupling_tolerance,
        resources,
    )
    routes = _linear_routes(geometry, coordinates, coordinate_identity, resources)
    kernel = FuelResidualOperator(
        admission=admission,
        geometry=geometry,
        heavy_metal=heavy_metal,
        free_volume=FreeVolumeModel(specification),
        unknown_scale=coordinates.unknown_scale,
        bounds=Bounds(coordinates.lower_bound),
        linear_space=routes.space,
        bordered_assembly=routes.assembly,
        joint_space=routes.joint_space,
        joint_bordered_assembly=routes.joint_assembly,
        field_shapes=coordinates.shapes,
        field_offsets=coordinates.offsets,
        prepared_id=prepared_id,
    )
    problem, method = _nonlinear_method(kernel, routes)
    termination = (
        NonlinearTermination(
            absolute_residual=_NONLINEAR_ABSOLUTE_RESIDUAL,
            relative_residual=_NONLINEAR_RELATIVE_RESIDUAL,
            maximum_steps=_NONLINEAR_MAXIMUM_STEPS,
        )
        if nonlinear_termination is None
        else nonlinear_termination
    )
    return PreparedFuelRod(
        kernel=kernel,
        nonlinear_problem=problem,
        nonlinear_template=_stage_nonlinear_template(
            kernel, problem, method, termination
        ),
        joint_linear_policy=routes.joint_policy,
        coupling_method=FixedPointIteration(),
        coupling_termination=NonlinearTermination(
            absolute_residual=coupling_tolerance,
            relative_residual=0.0,
            maximum_steps=coupling_budget,
            maximum_evaluations=coupling_budget,
        ),
        coupling_tolerance=coupling_tolerance,
        execution_signature=(
            admission.reduction,
            geometry.layer_lengths.shape[0],
            geometry.fuel.node_count,
            geometry.cladding.node_count,
            quadrature_order,
            admission.bundle.fission_gas.mode_count,
            admission.bundle.helium.radial_cell_count,
            admission.bundle.bundle_id,
        ),
    )
