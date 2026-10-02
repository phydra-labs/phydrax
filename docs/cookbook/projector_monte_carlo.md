# Sparse coefficient projector Monte Carlo

This complete bounded recipe uses two bosons on two sites with local cutoff three, hopping `t=1`, and on-site interaction `U=2` in hartrees. It prepares native outgoing columns, a frozen positive guide, a dimensionless number observable, semistochastic propagation, raw-history analysis, and checkpoint continuation. The [executable example](https://github.com/phydra-labs/phydrax/blob/dev/examples/projector_monte_carlo.py) is the standalone driver; the [guide](../guides_projector_monte_carlo.md) explains scientific limits.

The initial vector has exactly two particles; every Hamiltonian term conserves that number. This recipe uses the packed product domain without a ranked finite basis. An application requiring explicit charge-domain admission should also declare its exact charge group/target rather than infer one from the initial vector.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import final

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax import StrictModule
from phydrax.operators.quantum import LogAmplitude
from phydrax.operators.quantum.lattice import (
    LocalOperatorPlan,
    LocalSpacePlan,
    QuantumAddress,
    QuantumColumnResourcePolicy,
    QuantumConfigurationDomain,
    QuantumGuide,
    QuantumLatticeColumnOperator,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    prepare_quantum_lattice_columns,
)
from phydrax.solver import (
    ProjectorEstimatorPolicy,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloStatus,
    analyze_projector_monte_carlo,
    initialize_projector_monte_carlo,
    prepare_projector_monte_carlo,
    read_projector_monte_carlo_checkpoint,
    solve_projector_monte_carlo,
    step_projector_monte_carlo,
    transport_projector_monte_carlo_resources,
    write_projector_monte_carlo_checkpoint,
    write_projector_monte_carlo_result,
)
from phydrax.typing import Float64, Scalar
from phydrax.units import HARTREE, ONE, derived_unit

jax.config.update("jax_enable_x64", True)

left = LocalSpacePlan.boson("left", 3)
right = LocalSpacePlan.boson("right", 3)
creation = np.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, np.sqrt(2.0), 0.0]],
    dtype=np.complex128,
)
number = np.diag(np.asarray([0.0, 1.0, 2.0], dtype=np.float64))
interaction = np.diag(np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
create_left = LocalOperatorPlan(left, "create", creation, (1,))
annihilate_right = LocalOperatorPlan(right, "annihilate", creation.T.conj(), (-1,))
interaction_left = LocalOperatorPlan(left, "pair-energy", interaction, (0,))
interaction_right = LocalOperatorPlan(right, "pair-energy", interaction, (0,))
number_left = LocalOperatorPlan(left, "number", number, (0,))
specification = QuantumLatticeSpecification(
    (left, right),
    (
        QuantumLatticeTerm(
            (create_left, annihilate_right),
            coefficient=-1.0,
            add_adjoint=True,
            label="hopping",
        ),
        QuantumLatticeTerm((interaction_left,), coefficient=2.0, label="left-U"),
        QuantumLatticeTerm((interaction_right,), coefficient=2.0, label="right-U"),
    ),
)
resources = QuantumColumnResourcePolicy(
    maximum_monomials=8,
    maximum_factors_per_monomial=2,
    maximum_transition_table_bytes=16_384,
    maximum_raw_routes=64,
    maximum_column_targets=16,
    maximum_decoded_coordinate_bytes=1_024,
    maximum_workspace_bytes=131_072,
)
domain = QuantumConfigurationDomain(specification, species_ids=("boson", "boson"))
hamiltonian = QuantumLatticeColumnOperator(
    prepare_quantum_lattice_columns(specification, resources), domain
)
number_specification = QuantumLatticeSpecification(
    (left, right), (QuantumLatticeTerm((number_left,), label="left-number"),)
)
number_operator = QuantumLatticeColumnOperator(
    prepare_quantum_lattice_columns(number_specification, resources), domain
)


@final
class ConstantGuide(StrictModule):
    __strict_contract__ = True
    logarithm: Float64[Scalar]

    def __call__(self, address: QuantumAddress, /) -> LogAmplitude:
        return LogAmplitude(
            self.logarithm,
            jnp.asarray(1.0 + 0.0j, dtype=jnp.complex128),
        )


guide = QuantumGuide(
    domain,
    ConstantGuide(logarithm=jnp.asarray(np.log(3.0), dtype=jnp.float64)),
    provider_id="constant-three",
    mapping_id="packed-address-constant",
    globally_positive=True,
)
keys = jnp.stack(
    (
        domain.address(jnp.asarray([2, 0], dtype=jnp.int32)).key_words,
        domain.address(jnp.asarray([1, 1], dtype=jnp.int32)).key_words,
        domain.address(jnp.asarray([0, 2], dtype=jnp.int32)).key_words,
    )
)
problem = ProjectorMonteCarloProblem(
    hamiltonian,
    keys,
    jnp.asarray([10.0, 10.0, 10.0], dtype=jnp.complex128),
    dt=0.005,
    energy_unit=HARTREE,
    inverse_energy_unit=derived_unit("inverse-hartree", ((HARTREE, -1),)),
    provenance_id="two-boson-cookbook",
    guide=guide,
    observables=(number_operator,),
    observable_units=(ONE,),
)


def make_plan(scale: int = 1) -> ProjectorMonteCarloPlan:
    return ProjectorMonteCarloPlan(
        replicas=2,
        support_capacity=3 * scale,
        group_capacity=9 * scale,
        event_capacity=4 * scale,
        attempt_capacity=512 * scale,
        source_capacity=1_024 * scale,
        history_capacity=128 * scale,
        maximum_retained_bytes=4_194_304 * scale,
        maximum_workspace_bytes=4_194_304 * scale,
        spawn_policy="semistochastic",
        compression="threshold",
        controller="double-log",
        boost=2.0,
        relative_threshold=100.0,
        absolute_threshold=float("inf"),
        theta=0.1,
        initial_shift=-1.2,
        target_population=90.0,
    )


prepared = prepare_projector_monte_carlo(problem, make_plan())
state = initialize_projector_monte_carlo(prepared, jr.key(7))
result = solve_projector_monte_carlo(prepared, state, steps=32)
if int(result.status) != ProjectorMonteCarloStatus.SUCCESS:
    raise RuntimeError(f"Propagation refused: {int(result.status)}; inspect result.evidence")

with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "projector-checkpoint.npz"
    write_projector_monte_carlo_checkpoint(checkpoint, prepared, result.state)
    template = initialize_projector_monte_carlo(prepared, jr.key(999))
    restored = read_projector_monte_carlo_checkpoint(checkpoint, prepared, template)
    # Restoration uses the archived key, not jr.key(999).
    transition = step_projector_monte_carlo(prepared, restored)
    resource_refusals = (
        ProjectorMonteCarloStatus.HISTORY_EXHAUSTED,
        ProjectorMonteCarloStatus.ATTEMPT_LIMIT,
        ProjectorMonteCarloStatus.WORK_LIMIT,
        ProjectorMonteCarloStatus.INTERMEDIATE_GROUP_OVERFLOW,
        ProjectorMonteCarloStatus.FINAL_SUPPORT_OVERFLOW,
    )
    if int(transition.status) in resource_refusals:
        enlarged = prepare_projector_monte_carlo(problem, make_plan(2))
        replay_state, restart_relation = transport_projector_monte_carlo_resources(
            prepared, enlarged, transition.state
        )
        print("Explicit resource transport:", restart_relation)
        prepared = enlarged
        transition = step_projector_monte_carlo(prepared, replay_state)
    if not bool(transition.accepted):
        raise RuntimeError(f"Step refused: {int(transition.status)}; no fresh-key retry")
    continued = solve_projector_monte_carlo(prepared, transition.state, steps=31)
    if int(continued.status) != ProjectorMonteCarloStatus.SUCCESS:
        raise RuntimeError(f"Continuation refused: {int(continued.status)}")
    analysis = analyze_projector_monte_carlo(
        prepared,
        continued,
        policy=ProjectorEstimatorPolicy(burn_in=8, history_depths=(0, 1, 8)),
    )
    print("Projected exploratory value:", analysis.projected.exploratory_value)
    print("Projected qualified value/status:", analysis.projected.value, analysis.projected.status)
    print("Energy and number units:", analysis.observable_units)
    print("Replica energy and number statuses:", tuple(item.status for item in analysis.replicas))
    print("Systematic assumptions:", analysis.systematic)
    write_projector_monte_carlo_result(
        Path(directory) / "projector-result.npz",
        prepared,
        continued,
        run_id="two-boson-cookbook-run",
        analysis=analysis,
    )
```

The constant guide makes the represented population three times the physical coefficient norm; its physical metric is `1/9`. Initial/trial coefficients above are **physical** coefficients; initialization transforms them once. Each additional observable has an explicit unit; number is dimensionless, not implicitly an energy.

The tiny history illustrates the API, not statistically resolved ground-state production. Check `statistically_valid`, `status`, common-block/correlation evidence, denominator evidence, and weight diagnostics before reporting an estimate. Real Fieller or complex origin gates can refuse even when `exploratory_value` is finite. Do not replace a refusal with an instantaneous ratio, silently shorten a history window, infer deterministic records from constants, or increase capacities with a new random stream.

For this finite physical control the independent ground energy is `(U - sqrt(U**2 + 16*t**2))/2` and the symmetric ground-state left occupancy is one. Reaching those values in a short run is not proof of stationarity, population/history convergence, or absence of timestep/sign bias. No output values or passing qualification are asserted here. The archive in this recipe lives in a temporary directory; choose durable caller-owned paths for actual runs.
