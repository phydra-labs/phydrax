#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from enum import StrEnum
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._differentiation import DerivativeContract
from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._strict import StrictModule
from .._trainable import NonTrainableState, ParameterOwner, partition_parameters
from ..typing import checked, parse, PRNGKey
from ._conditioning import (
    AtomisticConditioning,
    AtomisticFieldReference,
    AtomisticModelOutputs,
    AtomisticObservable,
    AtomisticObservableRequest,
)
from ._graph import (
    AtomisticGraphExecutionPlan,
    AtomisticGraphLayout,
    AtomisticGraphPayload,
)
from ._model_inputs import AtomisticGraphInputs
from ._neighborhood_bundle import (
    AtomisticNeighborhoodRequirement,
    AtomisticNeighborhoodRole,
)
from ._types import (
    AtomisticBatch,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    AtomisticSpeciesKind,
)


_ENERGY_REQUEST = AtomisticObservableRequest((AtomisticObservable.ENERGY,))


class AtomisticStressConvention(StrEnum):
    """Named meaning of a homogeneous-strain energy derivative.

    Both members use the column deformation ``F = I + strain`` about the cell
    origin at fixed fractional coordinates and fixed integer images: row lattice
    vectors map to ``H @ F.T`` and positions to ``origin + (r - origin) @ F.T``.
    ``stress = sym(dE/dstrain) / |det H|`` in system energy per cubic length,
    positive under tension (pressure ``-trace(stress) / 3``).

    ``CAUCHY_TENSION_POSITIVE``: every axis is periodic and ``|det H|`` is the
    physical cell volume.
    ``CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME``: some axes are nonperiodic and
    ``|det H|`` is the declared embedding volume of the full invertible 3x3 cell
    (the nonperiodic rows set the embedding thickness), not an intrinsic bulk
    volume.
    """

    CAUCHY_TENSION_POSITIVE = "cauchy-tension-positive-column-strain"
    CAUCHY_TENSION_POSITIVE_EMBEDDING_VOLUME = (
        "cauchy-tension-positive-column-strain-embedding-volume"
    )


class AtomisticPotentialCapabilities(StrictModule, NonTrainableState):
    """Static execution capabilities, never a scientific stability claim."""

    conservative_energy: bool = eqx.field(static=True)
    finite_geometry: bool = eqx.field(static=True)
    orthorhombic_periodic: bool = eqx.field(static=True)
    triclinic_periodic: bool = eqx.field(static=True)
    cell_derivative: bool = eqx.field(static=True)
    local_energy: bool = eqx.field(static=True)
    local_energy_delta: bool = eqx.field(static=True)
    dynamic_species: bool = eqx.field(static=True)
    species_kind: AtomisticSpeciesKind = eqx.field(static=True)
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        conservative_energy: bool = True,
        finite_geometry: bool = True,
        orthorhombic_periodic: bool = False,
        triclinic_periodic: bool = False,
        cell_derivative: bool = False,
        local_energy: bool = True,
        local_energy_delta: bool = False,
        dynamic_species: bool = False,
        species_kind: AtomisticSpeciesKind = AtomisticSpeciesKind.ATOMIC_NUMBER,
    ) -> None:
        if not isinstance(species_kind, AtomisticSpeciesKind):
            raise TypeError("species_kind must be AtomisticSpeciesKind.")
        values = {
            "conservative_energy": bool(conservative_energy),
            "finite_geometry": bool(finite_geometry),
            "orthorhombic_periodic": bool(orthorhombic_periodic),
            "triclinic_periodic": bool(triclinic_periodic),
            "cell_derivative": bool(cell_derivative),
            "local_energy": bool(local_energy),
            "local_energy_delta": bool(local_energy_delta),
            "dynamic_species": bool(dynamic_species),
            "species_kind": species_kind,
        }
        for name, value in values.items():
            setattr(self, name, value)
        self.capabilities_id = canonical_fingerprint(
            {"kind": "atomistic-potential-capabilities", **values}
        )


class AtomisticPotentialRequirements(StrictModule, NonTrainableState):
    """Prepared-context requirements used to avoid unused runtime allocations."""

    bonded_geometry: bool = eqx.field(static=True)
    reciprocal_grid: bool = eqx.field(static=True)
    state_required: bool = eqx.field(static=True)
    field_dependencies: tuple[AtomisticFieldReference, ...]
    neighborhoods: tuple[AtomisticNeighborhoodRequirement, ...]
    requirements_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        bonded_geometry: bool = False,
        reciprocal_grid: bool = False,
        state_required: bool = False,
        field_dependencies: tuple[AtomisticFieldReference, ...] = (),
        neighborhoods: tuple[AtomisticNeighborhoodRequirement, ...] = (),
    ) -> None:
        if not isinstance(state_required, bool):
            raise TypeError("state_required must be a boolean.")
        dependencies = tuple(field_dependencies)
        if any(not isinstance(value, AtomisticFieldReference) for value in dependencies):
            raise TypeError(
                "field_dependencies must contain AtomisticFieldReference values."
            )
        if len({value.reference_id for value in dependencies}) != len(dependencies):
            raise ValueError("Field dependencies must be unique.")
        dependencies = tuple(sorted(dependencies, key=lambda value: value.reference_id))
        spatial = tuple(neighborhoods)
        if any(
            not isinstance(value, AtomisticNeighborhoodRequirement) for value in spatial
        ):
            raise TypeError(
                "neighborhoods must contain AtomisticNeighborhoodRequirement values."
            )
        if len({value.requirement_id for value in spatial}) != len(spatial):
            raise ValueError("Neighborhood requirements must be unique.")
        spatial = tuple(sorted(spatial, key=lambda value: value.requirement_id))
        self.bonded_geometry = bool(bonded_geometry)
        self.reciprocal_grid = bool(reciprocal_grid)
        self.state_required = state_required
        self.field_dependencies = dependencies
        self.neighborhoods = spatial
        self.requirements_id = canonical_fingerprint(
            {
                "kind": "atomistic-potential-requirements",
                "bonded_geometry": self.bonded_geometry,
                "reciprocal_grid": self.reciprocal_grid,
                "state_required": state_required,
                "field_dependencies": tuple(value.reference_id for value in dependencies),
                "neighborhoods": tuple(value.requirement_id for value in spatial),
            }
        )

    def neighborhood(
        self, role: AtomisticNeighborhoodRole, /
    ) -> AtomisticNeighborhoodRequirement:
        """Resolve one unambiguous role within this model or term's requirements."""
        selected = parse(role, AtomisticNeighborhoodRole, "role")
        matches = tuple(value for value in self.neighborhoods if value.role is selected)
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {selected.value} requirement; found {len(matches)}."
            )
        return matches[0]


class _AtomisticConfiguration(Protocol):
    @property
    def cutoff(self) -> float: ...


class AbstractAtomisticModel(StrictModule, ParameterOwner):
    """Typed atomistic model with an explicitly declared energetic capability."""

    configuration: eqx.AbstractVar[_AtomisticConfiguration]
    scale: eqx.AbstractVar[AtomisticScaleContract]
    precision: eqx.AbstractVar[AtomisticPrecisionPolicy]
    architecture_id: eqx.AbstractVar[str]
    method_id: eqx.AbstractVar[str]

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """Declared value/parameter response; absent declarations are not admission."""
        return None

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities(
            conservative_energy=False, local_energy=False
        )

    @property
    def requirements(self) -> AtomisticPotentialRequirements:
        """A learned graph model consumes only its directed graph and cutoff."""
        return AtomisticPotentialRequirements(
            neighborhoods=(
                AtomisticNeighborhoodRequirement(
                    self.architecture_id,
                    AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH,
                    float(self.configuration.cutoff),
                ),
            ),
        )

    @abstractmethod
    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        raise NotImplementedError

    @abstractmethod
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        """Evaluate one bound geometry with explicit scientific input identities."""
        raise NotImplementedError


class AbstractAtomisticPotential(AbstractAtomisticModel):
    """Scalar-energy owner; conservative responses derive from that same energy."""

    @property
    def capabilities(self) -> AtomisticPotentialCapabilities:
        return AtomisticPotentialCapabilities()

    @checked
    def energy(
        self,
        batch: AtomisticBatch,
        execution: AtomisticGraphExecutionPlan,
        /,
        *,
        positions: Array | None = None,
        payload: AtomisticGraphPayload | None = None,
        cell_vectors: Array | None = None,
        conditioning: AtomisticConditioning | None = None,
        key: PRNGKey | None = None,
        image_counts: Array | None = None,
    ) -> Array:
        """Evaluate case energies, refusing failed geometry, conditions or outputs.

        Graph preparation is a host boundary: without ``payload`` the one-batch
        layout of ``batch`` is prepared here. Supply a prepared payload (bound
        to ``batch`` through ``AtomisticGraphPayload.admit``) for transformed
        evaluation and reuse it while certified.
        """
        self._validate_batch(batch)
        coordinate = batch.positions if positions is None else positions
        if self.capabilities.species_kind is AtomisticSpeciesKind.ATOMIC_NUMBER:
            coordinate = eqx.error_if(
                coordinate,
                jnp.any(batch.atom_mask & ~batch.element_mask),
                "Atomic-number potentials cannot evaluate non-element particles.",
            )
        prepared = (
            AtomisticGraphLayout.for_batch(
                batch, execution, cutoff=self.configuration.cutoff
            ).prepare(batch)
            if payload is None
            else payload.admit(batch)
        )
        if prepared.execution.plan_id != execution.plan_id:
            raise ValueError(
                "The graph payload was prepared under another execution plan."
            )
        graph = prepared.graph(
            cutoff=self.configuration.cutoff,
            positions=coordinate,
            cell_vectors=cell_vectors,
            selection=self.requirements.neighborhood(
                AtomisticNeighborhoodRole.DIRECTED_IMAGE_GRAPH
            ).selection,
            image_counts=image_counts,
        )
        result = self.evaluate_graph(
            AtomisticGraphInputs(
                graph,
                batch.scale,
                species_kind=self.capabilities.species_kind,
                conditioning=conditioning,
                key=key,
            ),
            _ENERGY_REQUEST,
        )
        return eqx.error_if(
            result.energy,
            ~jnp.all(result.successful),
            "Atomistic model evaluation failed; energies are not valid.",
        )


class AbstractPreparedAtomisticModel(StrictModule):
    """A numerical realization bound to its exact scientific model."""

    model: eqx.AbstractVar[AbstractAtomisticModel]
    prepared_id: eqx.AbstractVar[str]
    capabilities: eqx.AbstractVar[AtomisticPotentialCapabilities]
    requirements: eqx.AbstractVar[AtomisticPotentialRequirements]

    @property
    def derivative_contract(self) -> DerivativeContract | None:
        """A realization must declare its own support; preparation can reduce regularity."""
        return None

    @property
    def configuration(self) -> _AtomisticConfiguration:
        return self.model.configuration

    @property
    def scale(self) -> AtomisticScaleContract:
        return self.model.scale

    @property
    def precision(self) -> AtomisticPrecisionPolicy:
        return self.model.precision

    def _validate_batch(self, batch: AtomisticBatch, /) -> None:
        self.model._validate_batch(batch)

    @abstractmethod
    def evaluate_graph(
        self,
        inputs: AtomisticGraphInputs,
        request: AtomisticObservableRequest | None = None,
        /,
    ) -> AtomisticModelOutputs:
        raise NotImplementedError


class AbstractPreparedAtomisticPotential(AbstractPreparedAtomisticModel):
    """Prepared scalar-energy execution form."""

    @abstractmethod
    def energy(self, context: Any, /) -> tuple[Any, Any]:
        """Return scalar energy and fixed-schema auxiliary evidence."""
        raise NotImplementedError


def atomistic_model_revision(
    model: AbstractAtomisticModel | AbstractPreparedAtomisticModel, /
) -> NumericRevision:
    """Return the numeric revision of the actual model's parameter-role owner.

    The semantic provenance names the architecture and force method; the numeric
    content is every PARAMETER-role leaf, keyed by its tree path. Fixed and
    model-state leaves never enter it. This is a host boundary: the parameters
    must be concrete arrays, so call it outside `jit`, `vmap`, and `grad`.
    """

    if isinstance(model, AbstractPreparedAtomisticModel):
        source = model.model
    elif isinstance(model, AbstractAtomisticModel):
        source = model
    else:
        raise TypeError("model must implement a native atomistic model contract.")
    parameters = partition_parameters(source)[0]
    return NumericRevision(
        SemanticProvenance(
            {
                "kind": (
                    "atomistic-potential"
                    if isinstance(source, AbstractAtomisticPotential)
                    else "atomistic-property-model"
                ),
                "architecture_id": source.architecture_id,
                "method_id": source.method_id,
            }
        ),
        {
            jax.tree_util.keystr(path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(parameters)[0]
        },
    )


__all__ = [
    "AbstractAtomisticModel",
    "AbstractAtomisticPotential",
    "AbstractPreparedAtomisticModel",
    "AbstractPreparedAtomisticPotential",
    "AtomisticPotentialCapabilities",
    "AtomisticPotentialRequirements",
    "AtomisticStressConvention",
    "atomistic_model_revision",
]
