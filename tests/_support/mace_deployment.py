"""Tiny native MACE models and deployment recipes for deployment tests.

The architecture is the smallest two-layer standard MACE exercising a residual
interaction, an equivariant hidden degree, and correlation two; its weights are
random native initializations, not a qualified pretrained model.
"""

from typing import Any

import jax
import numpy as np

import phydrax as phx


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


def graph_execution() -> Any:
    return phx.atomistic.AtomisticGraphExecutionPlan(
        32,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=1024,
            maximum_degree=64,
            maximum_images=343,
        ),
    )


def provider_plan(model: Any, *, skin: float = 0.4) -> Any:
    return phx.atomistic.NativeAtomisticProviderPlan(
        model,
        graph_execution(),
        finite_neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(256),
        skin=skin,
        deformation_margin=0.1,
    )


def calculator_plan(model: Any) -> Any:
    return phx.atomistic.interchange.NativeASECalculatorPlan(
        provider_plan(model), electronvolt_units()
    )


def periodic_dynamics(
    model: Any, system: Any, *, step_size: float = 0.25
) -> tuple[Any, Any]:
    """Fixed-cell NVE dynamics over the same native provider recipe."""

    provider = provider_plan(model).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        provider.program,
        provider.neighborhood,
        phx.atomistic.VelocityVerletPlan(step_size),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return dynamics, thermodynamic


def strained_water_problem() -> Any:
    """Periodic E/F/S labels of a fixed teacher MACE on two strained cells."""

    import jax.numpy as jnp
    import numpy as np

    cell = np.array([[4.2, 0.0, 0.0], [0.9, 4.0, 0.0], [0.5, -0.6, 4.4]])
    positions = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    structures = []
    for magnitude in (-0.015, 0.02):
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
    execution = graph_execution()
    labels = phx.atomistic.energy_and_forces(
        tiny_mace(seed=11), batch, execution, compute_stress=True
    )
    return phx.atomistic.AtomisticTrainingProblem(
        batch,
        execution,
        cutoff=3.0,
        training_energy=labels.energy,
        training_forces=labels.forces,
        training_stress=labels.stress,
    )
