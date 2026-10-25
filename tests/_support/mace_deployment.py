"""Tiny native MACE models and deployment recipes for deployment tests.

The architecture is the smallest two-layer standard MACE exercising a residual
interaction, an equivariant hidden degree, and correlation two; its weights are
random native initializations, not a qualified pretrained model.
"""

from typing import Any

import jax
import numpy as np

import phydrax as phx
from tests._support.atomistic_bundles import (
    neighborhood_bundle_plan,
    prepare_program,
    role_capacity,
)


SPECIES = (1, 8)


def electronvolt_units() -> Any:
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def tiny_mace(*, seed: int = 0, cutoff: float = 3.0) -> Any:
    architecture = phx.nn.atomistic.MACEArchitecture(
        species=SPECIES,
        cutoff=cutoff,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=2.0,
    )
    return phx.nn.atomistic.MACEPotential(
        electronvolt_units().scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(seed),
    )


def graph_execution(
    *,
    accumulation: phx.sparse.RelationAccumulation = "fast",
    search_cell: Any = None,
    search_margin: float = 0.0,
) -> Any:
    """Particle image graph execution; bitwise replay needs ordered accumulation.

    Training problems declare ``search_cell`` (the reusable layout's route-search
    envelope, a row lattice) and its deformation ``search_margin``.
    """
    return phx.atomistic.AtomisticGraphExecutionPlan(
        32,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=343,
        ),
        streamed=phx.sparse.StreamedRelationPlan(accumulation=accumulation),
        search_cell=search_cell,
        search_margin=search_margin,
    )


def finite_frame() -> Any:
    """Explicit nonperiodic search domain of finite deployment molecules."""
    return phx.discretization.PeriodicCell(
        40.0 * np.eye(3), origin=np.full((3,), -20.0), periodic_axes=(False,) * 3
    )


def provider_plan(
    model: Any,
    *,
    skin: float = 0.4,
    accumulation: phx.sparse.RelationAccumulation = "fast",
    resources: dict[str, Any] | None = None,
) -> Any:
    """Native provider recipe with explicit finite and periodic resource plans.

    ``resources`` overrides the bundle resources of both plans (for example a
    capacity ladder); the default is ``bundle_resources(skin=skin)``.
    """
    declared = bundle_resources(skin=skin) if resources is None else resources
    requirements = phx.atomistic.LearnedGraphPotentialTerm(
        model
    ).requirements.neighborhoods
    return phx.atomistic.NativeAtomisticProviderPlan(
        model,
        graph_execution(accumulation=accumulation),
        finite_resources=neighborhood_bundle_plan(
            requirements, frame=finite_frame(), **declared
        ),
        periodic_resources=neighborhood_bundle_plan(requirements, **declared),
    )


def bundle_resources(*, skin: float = 0.4) -> dict[str, Any]:
    """Explicit image resources of the periodic learned-graph dynamics.

    The source search storage, receiver degree and image stencil use the
    per-case segment layout with a cell deformation margin; the directed graph
    role stores the same routes and degree.  ``provider_plan`` binds the same
    resources to its finite (with ``finite_frame``) and periodic plans.
    """
    capacity = phx.discretization.ParticleImageCapacity(
        maximum_particles_per_cell=16,
        maximum_edges=1024,
        maximum_degree=64,
        maximum_images=343,
        maximum_case_edges=1024,
    )
    return {
        "search": phx.atomistic.AtomisticCellListImageSearch(
            maximum_candidate_slots=10**6, deformation_margin=0.1
        ),
        "skin": skin,
        "search_capacities": (capacity,),
        "role_capacities": (role_capacity(1024, 64),),
    }


def learned_program(
    model: Any,
    system: Any,
    *,
    skin: float = 0.4,
    accumulation: phx.sparse.RelationAccumulation = "deterministic",
) -> Any:
    """The provider recipe's learned-graph program bound to explicit resources."""
    return prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            [
                phx.atomistic.LearnedGraphPotentialTerm(
                    model, allow_periodic=system.cell is not None
                )
            ]
        ),
        system,
        graph_execution=graph_execution(accumulation=accumulation),
        **bundle_resources(skin=skin),
    )


def calculator_plan(model: Any) -> Any:
    return phx.atomistic.interchange.NativeASECalculatorPlan(
        provider_plan(model), electronvolt_units()
    )


def periodic_dynamics(
    model: Any, system: Any, *, step_size: float = 0.25
) -> tuple[Any, Any]:
    """Fixed-cell NVE dynamics over the native provider recipe.

    Ordered accumulation makes in-process restarts continue bitwise on every
    backend; fresh-process GPU restarts also need ``compiler_profile``.
    """

    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        learned_program(model, system),
        phx.atomistic.VelocityVerletPlan(step_size),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return dynamics, thermodynamic


def strained_water_problem() -> Any:
    """Periodic E/F/S labels of a fixed teacher MACE on two strained cells.

    One full-batch window per update with fitted E/F/S scales; ordered
    accumulation makes labels and continuation bitwise within one process.
    """

    import jax.numpy as jnp
    import numpy as np

    cell = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    magnitudes = (-0.015, 0.02)
    structures = []
    for magnitude in magnitudes:
        deformation = np.eye(3) + magnitude * np.array(
            [[1.0, 0.2, 0.0], [0.2, -0.5, 0.1], [0.0, 0.1, 0.7]]
        )
        structures.append(
            phx.atomistic.AtomicStructure(
                jnp.asarray([8, 1, 1]),
                jnp.asarray(positions @ deformation.T),
                jnp.asarray([15.999, 1.008, 1.008]),
                electronvolt_units().scale,
                cell=jnp.asarray(cell @ deformation.T),
                periodic_axes=jnp.asarray([True, True, True]),
            )
        )
    batch = phx.atomistic.AtomisticBatch.from_structures(structures)
    # The reusable layout's search envelope covers every strained cell.
    execution = graph_execution(
        accumulation="deterministic", search_cell=cell, search_margin=0.1
    )
    labels = phx.atomistic.energy_and_forces(
        tiny_mace(seed=11), batch, execution, compute_stress=True
    )
    source = phx.atomistic.atomistic_training_source(
        batch,
        mass_unit=phx.units.DALTON,
        energy=labels.energy,
        forces=labels.forces,
        stress=labels.stress,
        record_ids=tuple(f"strained-water-{magnitude}" for magnitude in magnitudes),
    )
    return phx.atomistic.AtomisticTrainingProblem(
        phx.atomistic.atomistic_training_loader(source),
        execution,
        (
            phx.atomistic.AtomisticEnergyObjective(),
            phx.atomistic.AtomisticForceObjective(),
            phx.atomistic.AtomisticStressObjective(),
        ),
        scale=electronvolt_units().scale,
        cutoff=3.0,
        fields=phx.atomistic.AtomisticTrainingFields(cell="cell", periodic="periodic"),
    )
