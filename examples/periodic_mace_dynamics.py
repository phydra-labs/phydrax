"""Periodic native MACE: E/F/S in a skew cell, NVE dynamics, and exact restart.

Two water-like molecules sit in a triclinic cell smaller than twice the
cutoff, requiring an image-aware graph rather than a unique-image assumption.
The same prepared program supplies the provider's energy/forces/stress and the
dynamics forces. A portable restart bundles the pickle-free model artifact
with the dynamics state; resuming from it reproduces the uninterrupted
trajectory exactly. The model is a random native initialization.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
CELL = np.array([[4.6, 0.0, 0.0], [0.9, 4.4, 0.0], [0.5, -0.6, 4.8]])
POSITIONS = np.array(
    [
        [0.2, 0.3, 0.1],
        [1.16, 0.3, 0.1],
        [-0.04, 1.23, 0.1],
        [2.5, 2.4, 2.6],
        [3.46, 2.4, 2.6],
        [2.26, 3.33, 2.6],
    ]
)
NUMBERS = np.array([8, 1, 1, 8, 1, 1])
MASSES = np.array([15.999, 1.008, 1.008, 15.999, 1.008, 1.008])


def build_model() -> phx.nn.atomistic.MACEPotential:
    architecture = phx.nn.atomistic.MACEArchitecture(
        species=(1, 8),
        cutoff=3.0,
        radial_basis_count=4,
        cutoff_power=5,
        channel_count=4,
        hidden_degree=1,
        edge_degree=2,
        interactions=("real-agnostic", "real-agnostic-residual"),
        correlations=(2, 2),
        radial_widths=(8,),
        readout_width=4,
        average_neighbor_count=6.0,
    )
    return phx.nn.atomistic.MACEPotential(
        UNITS.scale,
        architecture,
        atomic_energies=np.asarray([[-1.0, -2.0]], dtype=np.float64),
        key=jax.random.key(0),
    )


def prepare(
    model: phx.nn.atomistic.MACEPotential,
) -> tuple[
    phx.atomistic.NativeAtomisticProvider,
    phx.atomistic.PreparedAtomisticDynamics,
    phx.atomistic.PreparedThermodynamicStateTable,
]:
    structure = phx.atomistic.AtomicStructure(
        jnp.asarray(NUMBERS),
        jnp.asarray(POSITIONS),
        jnp.asarray(MASSES),
        UNITS.scale,
        cell=jnp.asarray(CELL),
        periodic_axes=jnp.asarray([True, True, True]),
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(structure, UNITS).prepare()
    execution = phx.atomistic.AtomisticGraphExecutionPlan(
        64,
        backend="particle",
        image_capacity=phx.discretization.ParticleImageCapacity(
            maximum_particles_per_cell=16,
            maximum_edges=2048,
            maximum_degree=96,
            maximum_images=343,
        ),
    )
    provider = phx.atomistic.NativeAtomisticProviderPlan(
        model,
        execution,
        finite_neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(64),
        skin=0.4,
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        provider.program,
        provider.neighborhood,
        phx.atomistic.VelocityVerletPlan(0.25),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return provider, dynamics, thermodynamic


def run(
    dynamics: phx.atomistic.PreparedAtomisticDynamics,
    thermodynamic: phx.atomistic.PreparedThermodynamicStateTable,
    state: phx.atomistic.AtomisticDynamicsState,
    steps: int,
) -> phx.atomistic.AtomisticDynamicsState:
    for _ in range(steps):
        state = dynamics.step(state, thermodynamic)
    return state


def main() -> None:
    model = build_model()
    provider, dynamics, thermodynamic = prepare(model)
    evaluation = provider.evaluate_state(jnp.asarray(POSITIONS), None, None).evaluation
    print("energy [eV]:", float(evaluation.energy))
    print("stress [eV/Ang^3]:\n", np.asarray(evaluation.stress))

    velocity = 0.002 * jax.random.normal(jax.random.key(1), POSITIONS.shape)
    state = dynamics.initialize_state(
        jnp.asarray(POSITIONS), thermodynamic, velocity=velocity, key=jax.random.key(2)
    )
    state = run(dynamics, thermodynamic, state, 20)
    reference = run(dynamics, thermodynamic, state, 10)
    print(
        "NVE cumulative balance residual [eV]:",
        float(reference.energy.cumulative_balance_residual),
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "restart"
        plan = phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic)
        phx.atomistic.write_atomistic_restart(path, plan, state, model=model)

        restored_model = phx.atomistic.read_atomistic_restart_model(path).model
        _, restored_dynamics, restored_thermodynamic = prepare(restored_model)
        template = restored_dynamics.initialize_state(
            jnp.asarray(POSITIONS),
            restored_thermodynamic,
            velocity=jnp.zeros_like(velocity),
            key=jax.random.key(2),
        )
        resumed = phx.atomistic.read_atomistic_restart(
            path,
            phx.atomistic.AtomisticCheckpointPlan(
                restored_dynamics, restored_thermodynamic
            ),
            template,
        ).state
        continued = run(restored_dynamics, restored_thermodynamic, resumed, 10)
    identical = bool(
        jnp.array_equal(continued.kinematics.positions, reference.kinematics.positions)
    )
    print("restart reproduces the uninterrupted trajectory:", identical)


if __name__ == "__main__":
    main()
