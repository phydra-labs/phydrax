# Machine learning interoperability

Phydrax lets a learned component (a network, a fitted estimator, a functional
model with its own state, an external runtime) sit inside a native owner: a PDE
functional, a nonlinear solve, a Krylov iteration, a finite-volume plan, a
controller. The two sides never trust each other implicitly. The component says
what it *is* through declared array roles, ports, regularity, precision, and
randomness. The owner says what the component is *allowed to decide* through
its authority, and it keeps ownership of acceptance, residuals, and answers.

This guide walks through those contracts in the order an integration meets
them, then lists the qualification gates that exercise them end to end. The
executable blocks below share one namespace and run in order.

## Array roles

Every array leaf of a model PyTree has exactly one training role,
`phx.ArrayRole`:

- `PARAMETER` — differentiated and updated by training;
- `FIXED` — never updated (normalizer statistics, geometry, prepared data);
- `MODEL_STATE` — carried forward by the model itself (running statistics,
  counters) and committed only with an accepted update.

Trainability is declared, never inferred from dtype. A module field declares the
role of its whole value subtree with `phx.parameter_field()`,
`phx.fixed_field()`, or `phx.model_state_field()`; each accepts the keyword
arguments of `equinox.field`, and `static=True` is a `TypeError` because static
fields are structure, not arrays. Components that subclass `phx.ParameterOwner`
(every `phx.AbstractArrayModel` does) turn their unannotated inexact arrays into
PARAMETER leaves.

`phx.resolve_array_roles(tree)` reads the declarations with a path-aware walk and
returns a static `phx.RoleResolution` (`paths`, `roles`, `violations`,
`unclassified`, `filter_spec(...)`); it never raises. Precedence, top-down:

1. `Domain`, `phx.NonTrainableState`, and `phx.ExplicitFreeze` nodes are
   terminal: every leaf below them is FIXED.
2. An explicit field role applies to its value's subtree; the nearest
   declaration wins.
3. A `parameter_field` or `model_state_field` holding a terminal node is a
   `role-field-on-terminal-value` violation: a declaration never unfreezes.
4. Outside terminal nodes, containers inherit their field's role, and
   unannotated inexact arrays below a `ParameterOwner` are PARAMETER.
5. Below a plain `NonTrainableState`, a trainable `ParameterOwner` or a
   `parameter_field` is a `parameter-under-fixed-ancestor` violation (a silent
   freeze). An `ExplicitFreeze` node ends that audit for its subtree.
6. Any remaining inexact array is unclassified. Integer arrays and non-array
   leaves are FIXED unless declared MODEL_STATE.

Training boundaries call `phx.require_parameter_roles(tree, context=...)`. It
raises a `ValueError` naming every violation and unclassified path with its
remedy, and it also rejects callables or static fields that hide inexact arrays
from the tree (closure cells, defaults, partial arguments, bound instances).
`phx.partition_parameters(tree)` applies the same role check and splits the tree
into `(parameters, model_state, fixed)` lanes with `None` holes (terminal nodes
stay whole in `fixed`); `phx.combine_parameters` restores the tree.

```python executable
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class Calibrated(phx.StrictModule):
    weight: jax.Array = phx.parameter_field()
    shift: jax.Array = phx.fixed_field()
    running_mean: jax.Array = phx.model_state_field()

    def __call__(self, x):
        return self.weight * (x - self.shift - self.running_mean)


model = Calibrated(jnp.ones(3), jnp.zeros(3), jnp.zeros(3))
resolution = phx.resolve_array_roles(model)
assert resolution.roles == (
    phx.ArrayRole.PARAMETER,
    phx.ArrayRole.FIXED,
    phx.ArrayRole.MODEL_STATE,
)

parameters, model_state, fixed = phx.partition_parameters(model)
assert parameters.shift is None and model_state.weight is None
assert eqx.tree_equal(phx.combine_parameters(parameters, model_state, fixed), model)


class Undeclared(phx.StrictModule):
    weight: jax.Array


try:
    phx.require_parameter_roles(Undeclared(jnp.ones(2)), context="guide example")
except ValueError as error:
    assert "Unclassified inexact array leaves" in str(error)
else:
    raise AssertionError("an undeclared inexact array must be refused")
```

See [API → Array roles and lanes](api/phydrax.md#array-roles-and-lanes).

## Explicit freeze

Freezing is a declaration, not a side effect. A `phx.NonTrainableState` node
freezes its subtree and audits it: a trainable component below it is a role
violation, because freezing it would be silent. A `phx.ExplicitFreeze` node
freezes its subtree on purpose and is exempt from that audit.

`phx.uq.FrozenModel(model)` is the canonical `ExplicitFreeze` holder for a
trained model. Every array below it is FIXED, its execution contract and ports
are those of the wrapped model (freezing changes no evaluation claim), and
`as_trainable()` returns the wrapped model without copying its leaves. Fitted
estimators arrive frozen: `phx.ml.fit(...)` returns a `phx.ml.FitResult` whose
`model` is a `FrozenModel` and whose `as_trainable()` returns the trainable
executable, for example as a warm start.

An explicit `phx.nn.parameters.ParameterSubspace` selection is itself a role
declaration for one request. It may select only PARAMETER or unclassified
inexact leaves (`ParameterSubspace.array_leaf_paths(tree)` lists them); FIXED
leaves, MODEL_STATE leaves, and anything below a terminal node are refused.
`FunctionalSolver.solve(..., parameter_subspace=subspace)` then trains exactly
that selection, for example one coefficient of a fitted closure while its
intercept stays bitwise unchanged.

```python executable
network = phx.nn.models.MLP(
    in_size=2, out_size="scalar", width_size=8, depth=1, key=jr.key(0)
)
frozen = phx.uq.FrozenModel(network)
assert jax.tree_util.tree_leaves(phx.partition_parameters(frozen)[0]) == []
assert frozen.as_trainable() is network


class SilentFreeze(phx.StrictModule, phx.NonTrainableState):
    network: phx.nn.models.MLP


# Holding a trainable network below plain fixed state is a violation;
# holding its explicitly frozen form is not.
assert phx.resolve_array_roles(SilentFreeze(network)).violations
assert not phx.resolve_array_roles(SilentFreeze(frozen)).violations

paths = phx.nn.parameters.ParameterSubspace.array_leaf_paths(network)
subspace = phx.nn.parameters.ParameterSubspace.from_leaf_paths(network, paths[:1])
assert subspace.leaf_paths == paths[:1]
```

See [API → Explicit model subspaces](api/nn/parameters.md#explicit-model-subspaces)
and [Native machine learning](guides/ml.md).

## Model state

The MODEL_STATE lane holds numbers a model advances itself, such as running
statistics. `phx.nn.models.FunctionalJAXAdapter` wraps any functional model with
explicit lanes:

```python
apply(parameters, model_state, input, key, *, inference) -> (output, next_model_state)
```

`parameters` are PARAMETER, `model_state` is MODEL_STATE, and
`next_model_state` must match `model_state` in structure, shape, and dtype.
Evaluation (`__call__`) returns the output only and never changes the model.
`transition(x, key=...)` returns the output together with an adapter holding
the *candidate* next state. The training kernel commits that candidate only
with an accepted update: a rejected update commits neither parameters nor model
state, an accepted one commits both together, and checkpoints restore both.
Inference is explicit: `inference` is passed to every `apply` call and switched
with `phx.nn.layers.inference_mode(tree)`.

The same transaction rule holds when training is coupled to a discrete plant:
`phx.lifecycle.coupled_training_step` commits plant and training state
together, a physical rejection restores parameters, model state, optimizer
state, targets, and cursors exactly, and `phx.lifecycle.CoupledTrainingPolicy`
decides whether a valid physical step survives a training rejection
(`PHYSICAL_MAY_COMMIT`) or is rejected with it (`JOINTLY_REQUIRED`).

Domain fields (`domain.Model(...)`) refuse models that carry MODEL_STATE
arrays, and `FunctionalSolver` carries the model-state lane unchanged.

```python executable
def running_center(parameters, model_state, x, key, *, inference):
    del key
    if inference:
        return parameters["scale"] * (x - model_state["mean"]), model_state
    mean = jnp.mean(x)
    next_state = {"mean": 0.5 * model_state["mean"] + 0.5 * mean}
    return parameters["scale"] * (x - mean), next_state


adapter = phx.nn.models.FunctionalJAXAdapter(
    running_center,
    {"scale": jnp.asarray(2.0)},
    {"mean": jnp.asarray(0.0)},
    in_size=4,
    out_size=4,
    inference=False,
)
x = jnp.asarray([1.0, 2.0, 3.0, 6.0])
assert jnp.array_equal(adapter(x), adapter(x))  # evaluation never advances state

output, candidate = adapter.transition(x)
assert float(adapter.model_state["mean"]) == 0.0  # nothing committed
assert float(candidate.model_state["mean"]) == 1.5

evaluated = phx.nn.layers.inference_mode(candidate)
assert jnp.allclose(evaluated(x), 2.0 * (x - 1.5))
```

See [API → Stateful functional models](api/nn/wrappers.md#stateful-functional-models)
and [Functional training runtime](guides_functional_training.md).

## Lanes

Roles decide what is differentiated and committed; lanes decide what is mapped.
`phx.LaneLayout(kind, mapped_paths)` declares which leaves carry a leading lane
axis (axis 0), with `kind` one of `"item"`, `"case"`, or `"member"`; every other
leaf is shared across the lane. The layout validates that every mapped leaf
exists and that all share one lane size (`lane_size`), returns an
`equinox.filter_vmap` axis tree (`in_axes`), and indexes lanes (`take`).
`LaneLayout.from_predicate(tree, predicate, kind=...)` maps every array leaf
whose path satisfies a predicate.

Lanes are independent of roles, so FIXED data may be lane-mapped. An ensemble
(`phx.uq.HomogeneousFunctionEnsemble`) keeps one stacked PyTree whose layout
maps member leaves; by default every array leaf carries the member axis, so each
member keeps its own FIXED normalizer while only parameters train. Serial and
vectorized member evaluation agree. When training is coupled to a plant,
per-lane parameters need a case or member layout aligned with the plant cases
and commit lane by lane.

```python executable
members = tuple(
    Calibrated(jnp.full(3, 1.0 + member), jnp.full(3, float(member)), jnp.zeros(3))
    for member in range(4)
)
ensemble = phx.uq.HomogeneousFunctionEnsemble.from_members(members)
stacked, layout = ensemble.model, ensemble.layout
assert ".shift" in layout.mapped_paths  # FIXED data on the member lane

serial = jnp.stack([member(x[:3]) for member in members])
vmapped = eqx.filter_vmap(
    lambda member, value: member(value), in_axes=(layout.in_axes(stacked), None)
)(stacked, x[:3])
assert jnp.allclose(serial, vmapped)
assert eqx.tree_equal(layout.take(stacked, 2), members[2])
```

## Intrinsic execution vs bound authority

Two contracts describe a component, and they are deliberately separate.

**What the model is.** `AbstractArrayModel.model_execution_contract()` returns
a `phx.ModelExecutionContract`, independent of any owner:

- `derivative` — a `phx.DerivativeContract` whose regularity is the model's
  value regularity;
- `execution` — `phx.ExecutionCapabilities` (execution tier, `jit`, `vmap`,
  `host_only`, `stateful`);
- `precision` — a `phx.ComponentPrecisionContract`;
- `randomness` — a `phx.RandomnessContract`;
- `ports`, construction certificates, and semantic provenance.

`None` means undeclared. The default contract is conservative: regularity,
precision, and randomness undeclared. Model families with known structure
(such as `phx.nn.models.MLP`) declare them. `contract_id` content-addresses the
contract. The contract never declares authority.

**What the model may decide.** `phx.ComponentAuthority` names what a component
is trusted to decide inside its owner:

| Authority | Role inside the owner |
|---|---|
| `ACCELERATOR` | changes how much work a native iteration takes, never its answer (preconditioners) |
| `DISCRETIZATION` | part of the discrete operator (face closures, accepted-step corrections) |
| `MODEL` | defines the physical model (a learned constitutive law) |
| `SURROGATE` | approximates a field or map (a PINN trial function, an operator proposal) |
| `DECISION` | chooses actions (feedback policies, controller weights) |

`MODEL`, `DISCRETIZATION`, and `SURROGATE` components define residual values;
`ACCELERATOR` and `DECISION` components do not. Authority is conferred by
binding, never by the model, and comes from exactly one of:

- **the owning slot** — an inline component is held in a slot class
  (`phx.AbstractComponentSlot`) that declares its authority, slot semantic ID,
  and admissibility requirements; for example `phx.solver.LearnedStepCorrection`
  confers `DISCRETIZATION`;
- **an explicit binding** — `phx.bind_component(model, slot_or_authority)`
  returns a `phx.ComponentBinding` for a model held separately from its owner;
  the model stays a dynamic child whose arrays keep their roles;
- **frontend root authority** — `phx.solver.FunctionalSolver` trains
  parameters without an owning slot with `SURROGATE` authority on
  physical-residual objectives. `phx.solver.train_components` has no root
  authority: every PARAMETER leaf needs a slot or binding.

`ComponentBinding.contract(request=..., policy=...)` forms the bound
`phx.ComponentContract`: authority, slot identity, the intrinsic model
contract, port binding evidence, requirements, an optional derivative admission
made for this authority, and a content-addressed `bound_semantic_id`.
Construction fails closed when a requirement lacks its evidence.

```python executable
contract = network.model_execution_contract()
assert contract.execution.tier == "native-jax"
assert contract.randomness.mode == "deterministic"

bound = phx.bind_component(network, phx.ComponentAuthority.SURROGATE).contract()
assert bound.authority is phx.ComponentAuthority.SURROGATE
assert bound.slot_semantic_id is None  # an authority-only binding
assert bound.model_contract.contract_id == contract.contract_id

corrector = phx.nn.models.MLP(in_size=4, out_size=2, width_size=8, depth=1, key=jr.key(1))
in_slot = phx.bind_component(corrector, phx.solver.LearnedStepCorrection).contract()
assert in_slot.authority is phx.ComponentAuthority.DISCRETIZATION
assert in_slot.slot_semantic_id == "solver.accepted-step-transform"
```

See [API → Model execution contracts](api/differentiation.md#model-execution-contracts)
and [API → Component binding](api/differentiation.md#component-binding).

## Ports

A `phx.ValuePort` gives one input or output value an explicit scientific
identity: semantic ID, event shape, per-component IDs, representation, variance,
and optionally support space, per-component dimensions, frame, normalization,
and semantic event axes. `port_id` content-addresses every declared field.
`phx.ModelPorts(inputs=..., outputs=...)` orders a model's or slot's ports.

Ports bind only through an explicit `phx.PortMapping` of
`(model_port_id, owner_port_id)` pairs. Names, shapes, and order never establish
identity. `phx.resolve_port_mapping(model_ports, owner_ports, mapping)` requires
every model port to be mapped (owner ports may stay unused) and:

- refuses any mismatch of semantic ID, component IDs, event shape,
  representation, or variance;
- compares dimensions, semantic axes, frame, normalization, and space when both
  sides declare them, and records each aspect a side left undeclared in the
  returned `phx.PortBindingEvidence` (`unverified`, `dimensions_verified`,
  `axes_verified`, ...).

`bind_component(..., owner_ports=..., port_mapping=...)` requires a mapping
exactly when the model declares ports and owner ports are supplied. Domain
fields bind ports the same way: `domain.Model(*deps, port_mapping=...)` maps each
model input port to the `domain.value_port(label)` of one dependency, packs
dependencies in `deps` order without repacking, and publishes the evidence as the
field's `port_binding`. A fitted estimator declares ports from its schemas, so it
cannot be bound without a mapping:

```python
space_time = phx.domain.Interval1d(0.0, 1.0) @ phx.domain.TimeInterval(0.0, 1.0)
x_port, t_port = space_time.value_port("x"), space_time.value_port("t")
result = phx.ml.fit(
    recipe,
    features,
    targets,
    feature_schema=phx.ml.FeatureSchema.from_ports((x_port, t_port)),
    target_schema=phx.ml.TargetSchema.from_port(conductivity_port),
)
mapping = phx.PortMapping(
    inputs=[(x_port.port_id, x_port.port_id), (t_port.port_id, t_port.port_id)]
)
kappa = space_time.Model("x", "t", port_mapping=mapping)(result.model)
evidence = kappa.port_binding
# space_time.Model("x", "t")(result.model) raises: an explicit port_mapping is required.
```

```python executable
def scalar_port(semantic_id, representation):
    return phx.ValuePort(
        semantic_id,
        event_shape=(),
        component_ids=(semantic_id,),
        representation=representation,
    )


x_port = scalar_port("x", "coordinate")
t_port = scalar_port("t", "coordinate")
kappa_port = scalar_port("kappa", "coefficient-field")
closure_ports = phx.ModelPorts(inputs=(x_port, t_port), outputs=(kappa_port,))
owner_ports = phx.ModelPorts(inputs=(t_port, x_port), outputs=(kappa_port,))

in_order = phx.PortMapping(
    inputs=[(x_port.port_id, x_port.port_id), (t_port.port_id, t_port.port_id)],
    outputs=[(kappa_port.port_id, kappa_port.port_id)],
)
evidence = phx.resolve_port_mapping(closure_ports, owner_ports, in_order)
assert evidence.inputs == (
    (x_port.port_id, x_port.port_id),
    (t_port.port_id, t_port.port_id),
)
assert not evidence.dimensions_verified  # neither side declared dimensions

crossed = phx.PortMapping(
    inputs=[(x_port.port_id, t_port.port_id), (t_port.port_id, x_port.port_id)],
    outputs=[(kappa_port.port_id, kappa_port.port_id)],
)
try:
    phx.resolve_port_mapping(closure_ports, owner_ports, crossed)
except ValueError as error:
    assert "semantic_id mismatch" in str(error)
else:
    raise AssertionError("a crossed port mapping must be refused")
```

Discrete fields carry ports too. A `phx.discretization.DiscreteFieldFunctionView`
over a finite-element, finite-volume, or spectral reconstruction publishes the
reconstruction's `value_port`, and pointwise sums and differences with another
field require equal units, event shape, frame, semantic axes, normalization, and
variance. Finite-element and finite-volume representations of one typed state
therefore compose (their difference is an ordinary field), while a field
declared in another frame or unit is refused when the expression is built,
before anything evaluates.

See [API → Ports](api/differentiation.md#ports).

## Regularity

`phx.DerivativeRegularity` declares how smooth a component is in its value
arguments (`INPUT`, `PRIMAL_STATE`): a classical `continuity` order (`-1`, `k`,
or `"smooth"`), the structure of the pieces between non-smooth loci
(`"polynomial"` with a `degree_bound`, `"smooth"`, or `"none"`), identifier
conditions, and an optional support. Constructors cover the common cases:
`smooth()`, `piecewise_polynomial(...)`, `piecewise_smooth(...)`, and
`discontinuous()`. `admits_order(order)` answers what a value derivative of that
order is:

- orders within the continuity class are `SMOOTH`;
- proven degeneracy is `NONE`: polynomial pieces of degree below the order (the
  second derivative of a ReLU network with a linear head), or a discontinuous
  map without pieces;
- everything else is `ALMOST_EVERYWHERE`, and from order `continuity + 2` it
  carries the `"singular-part-ignored"` condition.

Owners admit derivatives under a `phx.RegularityPolicy`:

- proven degeneracy is always refused (`regularity-degenerate`);
- almost-everywhere derivatives need
  `RegularityPolicy(allow_almost_everywhere=True)`;
- implicit routes need classical `C^1` (`implicit-requires-c1`), so a ReLU law
  cannot enter implicit mechanics;
- undeclared regularity is refused for `MODEL` and `DISCRETIZATION` authority,
  and admitted for `SURROGATE`, `ACCELERATOR`, and `DECISION` only with
  `allow_undeclared=True` on a non-implicit route.

Admission happens at planning, before any evaluation. A PINN whose Laplacian
would differentiate a ReLU→linear network is refused when the
`FunctionalSolver` is constructed; a ReLU→tanh composite is admitted only when
the solver is built with `regularity_policy=phx.RegularityPolicy(allow_almost_everywhere=True)`,
and its recorded admission carries the almost-everywhere level and condition.

```python executable
relu_linear = phx.DerivativeRegularity.piecewise_polynomial(continuity=0, degree_bound=1)
assert relu_linear.admits_order(1)[0] is phx.GradientLevel.ALMOST_EVERYWHERE
assert relu_linear.admits_order(2)[0] is phx.GradientLevel.NONE

relu_tanh = phx.DerivativeRegularity.piecewise_smooth(continuity=0)
level, conditions = relu_tanh.admits_order(2)
assert level is phx.GradientLevel.ALMOST_EVERYWHERE
assert conditions == ("singular-part-ignored",)

relu_network = phx.nn.models.MLP(
    in_size="scalar",
    out_size="scalar",
    width_size=16,
    depth=2,
    activation=jax.nn.relu,
    key=jr.key(0),
)
space = phx.domain.Interval1d(0.0, 1.0)
condition = phx.conditions.Residual(
    "u", space.component(), lambda u: phx.operators.laplacian(u, var="x") + 2.0
)
source = phx.integration.per_step(
    phx.integration.mean_over(condition.on), phx.integration.MonteCarloPlan(64)
)
try:
    phx.solver.FunctionalSolver(
        functions={"u": space.Model("x")(relu_network)},
        terms=(phx.terms.ResidualPenalty(condition, source),),
    )
except ValueError as error:
    assert "regularity-degenerate" in str(error)
else:
    raise AssertionError("a ReLU-linear PINN Laplacian must be refused at planning")
```

See [API → Regularity](api/differentiation.md#regularity) and
[Derivative contracts](appendix/ml_differentiability.md).

## Precision

`phx.ComponentPrecisionContract` declares the dtypes a component reads, holds,
computes, accumulates, and returns (canonical dtype names), plus
`absolute_error_floor` and `relative_error_floor` on its evaluation error and
the evidence behind them (`amplification_evidence`, `cast_boundary_evidence`).
`ComponentPrecisionContract.native(dtype)` declares one dtype with undeclared
floors. `residual_floor(scale)` is `max(absolute, relative * scale)` over the
declared floors, or `None`: machine epsilon is never substituted for an
undeclared floor.

Nonlinear owners compose these contracts in
`phx.nonlinear.NonlinearPrecisionPolicy(components=...)`, which takes bound
`ComponentContract` values. Residual-defining components (`MODEL`,
`DISCRETIZATION`, `SURROGATE`) must declare a precision contract, and
`residual_floor(scale)` sums their floors. `ACCELERATOR` and `DECISION`
components are excluded, so a float32 preconditioner never raises the floor of a
float64 residual. Passed to a Newton solve as `precision=policy`, the policy
refuses a residual-defining component that computes in a coarser dtype without
a declared floor, and refuses a tolerance below the derived floor; the floor
itself is an achievable tolerance.

```python executable
class Cubic(phx.AbstractArrayModel):
    precision: phx.ComponentPrecisionContract
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, precision):
        self.precision = precision
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        return x**3

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=self.precision,
            randomness=phx.RandomnessContract("deterministic"),
        )


def computed_in(dtype, floor):
    return phx.ComponentPrecisionContract(
        input_dtype="float64",
        parameter_dtype=dtype,
        compute_dtype=dtype,
        accumulation_dtype=dtype,
        output_dtype="float64",
        absolute_error_floor=floor,
        cast_boundary_evidence=(f"float64-to-{dtype}-input",),
    )


physical = Cubic(computed_in("float64", 1e-13))
preconditioner = Cubic(computed_in("float32", 1e-6))
policy = phx.nonlinear.NonlinearPrecisionPolicy(
    components=(
        phx.bind_component(physical, phx.ComponentAuthority.MODEL).contract(),
        phx.bind_component(preconditioner, phx.ComponentAuthority.ACCELERATOR).contract(),
    )
)
assert policy.residual_floor() == 1e-13  # the accelerator does not raise it
```

See [API → Component precision floor](api/nonlinear.md#component-precision-floor).

## Randomness

`phx.RandomnessContract(mode)` declares a component's randomness:

- `"deterministic"` — none;
- `"fixed-realization"` — random structure frozen to one realization, named by
  `realization_id` once bound;
- `"resampled"` — fresh randomness on every evaluation.

`requires_inference_state=True` declares that the mode holds only in inference
state (dropout disabled, running statistics frozen), which the owner must bind.
Implicit and authoritative owners admit deterministic components and bound
fixed realizations; they refuse resampled randomness
(`resampled-randomness-not-admitted`) and undeclared randomness. Solver
objectives require every component model to be deterministic or bound to one
realization.

`phx.FrozenRealization(model, key, realization_id=...)` is the owner's explicit
realization binding: every evaluation uses the bound key (the caller's key is
ignored), and the binding declares a fixed realization. An implicit root through
a frozen realization reproduces its primal and its derivative exactly, and its
component evidence names the realization.

```python executable
class Noisy(phx.AbstractArrayModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=None):
        return jnp.tanh(x) + 0.05 * jr.normal(key, jnp.shape(x), dtype=x.dtype)

    def model_execution_contract(self):
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            randomness=phx.RandomnessContract("resampled"),
        )


draw = phx.FrozenRealization(Noisy(), jr.key(7), realization_id="noise-draw-7")
randomness = draw.model_execution_contract().randomness
assert randomness.mode == "fixed-realization"
assert randomness.realization_id == "noise-draw-7"
point = jnp.asarray([0.3, -0.2])
assert jnp.array_equal(draw(point, key=jr.key(1)), draw(point, key=jr.key(2)))
```

## Objective admission

A training objective has a scientific meaning, `phx.ObjectiveKind`
(`PHYSICAL_RESIDUAL`, `DATA_FIT`, `SOLUTION_MAP`, `ROLLOUT`, `ALGORITHMIC_WORK`,
`SUPERVISED_PROXY`), and a derivative mechanism, `phx.DerivativeRoute`
(`DIRECT`, `IMPLICIT`, `UNROLLED`, `SPECTRAL`, `RELAXED`, `EXTERNAL_ADJOINT`,
`STOPPED`). `phx.authority_admits(authority, route, kind)` decides which pairs
may train a component:

| Authority | Admitted `(route, kind)` pairs |
|---|---|
| `ACCELERATOR` | `(UNROLLED, ALGORITHMIC_WORK)`, `(DIRECT, SUPERVISED_PROXY)` |
| `DISCRETIZATION` | `(UNROLLED, ROLLOUT)`, `(DIRECT, ROLLOUT)`, `(IMPLICIT, SOLUTION_MAP)`, `(DIRECT, SUPERVISED_PROXY)`, `(DIRECT, PHYSICAL_RESIDUAL)` |
| `MODEL` | `(DIRECT, DATA_FIT)`, `(DIRECT, PHYSICAL_RESIDUAL)`, `(DIRECT, SUPERVISED_PROXY)`, `(IMPLICIT, SOLUTION_MAP)`, `(UNROLLED, ROLLOUT)`, `(EXTERNAL_ADJOINT, SOLUTION_MAP)`, `(EXTERNAL_ADJOINT, ROLLOUT)` |
| `SURROGATE` | `(DIRECT, PHYSICAL_RESIDUAL)`, `(DIRECT, DATA_FIT)`, `(UNROLLED, ROLLOUT)` |
| `DECISION` | `(DIRECT, ROLLOUT)`, `(UNROLLED, ROLLOUT)`, `(RELAXED, ROLLOUT)`, `(DIRECT, SUPERVISED_PROXY)` |

Solver objectives turn one FIXED prepared solve into a training signal for a
component that lives outside it:

| Objective | Kind | Route |
|---|---|---|
| `phx.solver.SolverObjective` | `SOLUTION_MAP` | `IMPLICIT` |
| `phx.solver.RolloutObjective` | `ROLLOUT` | `UNROLLED` |
| `phx.solver.AlgorithmicWorkObjective` | `ALGORITHMIC_WORK` | `UNROLLED` |

Each takes `(solve, bind, measure)` plus `objective_id`, optional `cases`, and an
optional `component` selector. `bind(solve, component)` binds the trained
component into the prepared solve through the owner's own binding path on every
evaluation; `measure(owner, case)` runs it. An `AlgorithmicWorkObjective` runs
exactly `work` iterations without early exit and scores
`phx.solver.algorithmic_work_loss`, the log ratio of final to initial residual
norms of the original problem (with a stopped precision-aware floor); a case
whose iteration count differs from `work` counts as failed.

Admission runs before anything is traced. It refuses a PARAMETER leaf without an
owning slot or binding, a stochastic model without a frozen realization, and a
component whose authority admits no signal from the objective. An accelerator
changes how fast a solve converges, never its answer, so a solution-map-only
objective has no admissible signal for it: instead of silently producing a zero
gradient, it is refused with a `ValueError`. The same accelerator trains through
an `AlgorithmicWorkObjective`.

`phx.solver.train_components(tree, objectives, optimizer=..., steps=..., key=...)`
trains a tree against several objectives. Each objective trains only the
authority groups that admit it and holds the others fixed, so a mixed-authority
tree needs one compatible objective per group — for example a `RolloutObjective`
for a face closure (`DISCRETIZATION`) and an `AlgorithmicWorkObjective` for a
preconditioner (`ACCELERATOR`):

```python
result = phx.solver.train_components(
    {"closure": closure, "preconditioner": preconditioner},
    (rollout, work),  # each built with component=lambda tree: tree["closure"], ...
    optimizer=optax.adam(3e-2),
    steps=12,
    key=jr.key(6),
)
closure_paths, preconditioner_paths = result.selection
```

```python executable
assert not phx.authority_admits(
    phx.ComponentAuthority.ACCELERATOR,
    phx.DerivativeRoute.IMPLICIT,
    phx.ObjectiveKind.SOLUTION_MAP,
)
assert phx.authority_admits(
    phx.ComponentAuthority.ACCELERATOR,
    phx.DerivativeRoute.UNROLLED,
    phx.ObjectiveKind.ALGORITHMIC_WORK,
)


def never_called(*_):
    raise AssertionError("a refused objective never binds or measures")


solution_map_only = phx.solver.SolverObjective(
    None, never_called, never_called, objective_id="solution-map-only"
)
accelerator = phx.bind_component(network, phx.ComponentAuthority.ACCELERATOR)
try:
    solution_map_only.evaluate(accelerator)
except ValueError as error:
    assert "no admissible training signal" in str(error)
else:
    raise AssertionError("an accelerator has no solution-map signal")
```

See [API → Solver objectives and component training](api/solver/component_training.md)
and [API → Authority and objectives](api/differentiation.md#authority-and-objectives).

## Proposals vs authoritative results

A learned component may *propose*; the native owner *decides*. The owner
re-evaluates its own residual, owns acceptance, and keeps the proposal
inspectable beside the result:

- **Step corrections.** `phx.solver.LearnedStepCorrection(model, state_shape=...,
  maximum_relative_correction=...)` maps the accepted state and the native
  candidate to an increment. Native checks (finiteness and support, declared
  conservation invariants, lower bounds, a stability bound relative to the
  native increment) admit or reject the proposal as one transaction. A rejected
  proposal leaves the native candidate unchanged and reports its reason bits
  (`phx.solver.LearnedStepCorrectionReason`); the transform never retries and
  never fails the step. Derivatives hold every accept/reject decision frozen.
- **Initial guesses and operator proposals.** A neural operator can propose the
  next state of an implicit step, or the degrees of freedom of a finite-element
  solution, and hand it to a native Newton solve as the initial guess. The
  corrector re-evaluates the native residual, so the corrected answer does not
  depend on the proposal's quality, and a proposal that is not a number never
  becomes an answer. The proposal and the corrected state stay separately
  inspectable: in a `phx.solver.SolverCaseResult` `aux` record, or as typed
  fields through `phx.discretization.DiscreteFieldFunctionView`, whose
  port-compatible difference is the correction itself.
- **Learned constitutive laws.** A law bound with
  `phx.equations.LearnedConstitutiveModel` inside an implicit root is accepted
  only by the native Newton termination; a load whose equilibrium leaves the
  learned support is never accepted, and an invalid tangent poisons the
  derivative instead of returning a plausible number.
- **Accelerators.** A learned preconditioner changes Krylov work, not the answer:
  the native solve measures the original residual, so a preconditioner whose
  action is invalid outside its training support cannot make the solve report
  success.

## External tiers

`phx.ExecutionTier` names how a model executes. The tier fixes whether the model
may run under JAX tracing; derivatives are a separate question answered by its
derivative contract (`phx.supports_derivative`):

| Tier | Execution | Host-only |
|---|---|---|
| `"native-jax"` | a Phydrax JAX model | never |
| `"functional-jax"` | `FunctionalJAXAdapter` over an external functional model | never |
| `"converted-native"` | an external model converted into native JAX | never |
| `"compiled-inference"` | a precompiled executable | as declared |
| `"host-inference"` | a host runtime outside JAX (`HostInferenceAdapter`) | always |
| `"external-adjoint"` | a provider supplying staged adjoint actions (`ExternalAdjointAction`) | as declared |

A host-only model supports neither `jit` nor `vmap`. The `EXTERNAL_ADJOINT`
derivative route requires the `"external-adjoint"` tier and provides reverse
mode only.

**Host-only inference.** `phx.export.HostInferenceAdapter(runner, input_schema,
output_schema, binding)` runs a host runtime eagerly. Inputs and outputs are
checked against `phx.interchange.ExternalTensorSpec` schemas, outputs must be
finite, and results are detached arrays. `jit`, `vmap`, `grad`, `jvp`, and `vjp`
are refused with a `TypeError` before the runtime is invoked. A host-only
model's contract offers no JAX derivative route (`STOPPED`, `EXTERNAL_ADJOINT`,
or no supported surfaces).

**Staged external adjoints.** A provider with its own adjoint subclasses
`phx.interchange.ExternalAdjointAction` and implements `_primal(inputs)`
(returning a `phx.interchange.ExternalPrimalStage` with a `realization_id`) and
`_adjoint(stage, output_cotangents)` (returning the input cotangents and the
realization they were formed at). `stage_primal(*inputs)` evaluates once and
stages the detached primal; downstream JAX code differentiates with respect to
the stage outputs; `apply_adjoint(stage, *output_cotangents)` returns `Jᵀȳ` at
exactly that realization, and upstream JAX code continues with its own VJP.
Both calls are host-only and refuse every JAX transformation before reaching
the provider. Replay checks fail closed: a stage from another action or
configuration is refused before the provider's adjoint runs, and an adjoint
formed at a different realization is refused instead of returning cotangents.

```python
x, pullback = jax.vjp(upstream, theta)
stage = provider.stage_primal(x)
outputs = tuple(jnp.asarray(value) for value in stage.outputs)
loss, output_cotangents = jax.value_and_grad(downstream, argnums=(0, 1))(*outputs)
(x_bar,) = provider.apply_adjoint(stage, *output_cotangents)
(theta_bar,) = pullback(jnp.asarray(x_bar))
```

**No silent derivative-free fallback.** `derivative_support` reports a
provider's route: `"external-adjoint"` for staged adjoints, or `"none"` with
the Phydrax derivative-free methods that can drive it through eager function
values. Those alternatives are listed, never selected on the caller's behalf;
a solver objective whose trained model does not admit the objective's route is
refused with the derivative-free consumers named (`train_components` with a
distribution-evolution optimizer, or `phx.uq.fit_eki`).

```python executable
semantic = phx.SemanticProvenance({"kind": "guide-doubling-runtime"})
host = phx.export.HostInferenceAdapter(
    lambda values: [2.0 * values[0]],
    (phx.interchange.ExternalTensorSpec("x", (3,), jnp.float64),),
    (phx.interchange.ExternalTensorSpec("y", (3,), jnp.float64),),
    phx.ArtifactBindingIdentity(
        semantic,
        phx.NumericRevision(semantic, {"scale": 2.0}),
        phx.ExecutableSignature(shapes={"x": (3,)}, dtypes={"x": jnp.float64}),
    ),
)
values = jnp.asarray([1.0, 2.0, 3.0])
assert jnp.array_equal(host(values), 2.0 * values)  # eager inference runs
try:
    jax.jit(host)(values)
except TypeError as error:
    assert "JAX transformations" in str(error)
else:
    raise AssertionError("host-only inference refuses jit")
assert host.derivative_support.route == "none"
assert "phydrax.uq.fit_eki" in host.derivative_support.alternatives
```

See [API → Host inference](api/export.md#host-inference) and
[API → Staged external adjoints](api/interchange.md#staged-external-adjoints).

## Qualification gates

Twenty-four gates qualify these contracts end to end, each through public API
only. A gate's scenarios are the pytest functions named `test_g<N>_...` in
`tests/integration/test_ml_interoperability_qualification.py`; parametrized
scenarios belong to the same gate.

|Gate|Title|Consumer-visible assertion|
|---|---|---|
|G1|Statistical closure → PDE|A fitted closure keeps its ports and binds frozen into a PDE; an explicit parameter subspace trains a coefficient subset; a mismatched port mapping is rejected before execution.|
|G2|Learned constitutive → implicit mechanics|Native Newton owns acceptance; the implicit parameter gradient matches finite differences; an invalid tangent poisons the derivative; a law without classical C1 regularity is refused.|
|G3|Learned preconditioner → Krylov|Trained through a fixed-work `AlgorithmicWorkObjective`, the preconditioner reduces Krylov work without changing answers; outside its support the solve cannot report success.|
|G4|Neural dynamics → control|The same bound model drives continuous and fixed-step discrete dynamics; a neural feedback policy's rollout gradient matches finite differences and trains; MPC sensitivity through the learned linearization matches finite differences.|
|G5|Neural operator → native corrector|Proposal and corrected state stay separately inspectable (including as field views); the native corrector re-evaluates its own residual; corrected answers do not depend on proposal quality.|
|G6|External host refusal|Host-only models run eagerly; jit, vmap, grad, jvp and vjp are refused before invocation; no derivative-free method is chosen silently.|
|G7|Stateful functional model|A rejected update commits nothing; an accepted update commits parameters and model state together and checkpoints restore both; evaluation does not mutate state.|
|G8|Plan-embedded component|A closure inside prepared finite-volume dynamics trains (directly and through a rollout objective); fixed data beside it does not.|
|G9|Representation swap|One face closure serves Cartesian, mapped and unstructured finite-volume owners and finite-element facet traces read through a field view; mismatched ports are rejected.|
|G10|Silent-zero accelerator|An ACCELERATOR under a SOLUTION_MAP-only objective is refused with `ValueError` before tracing; a fixed-work objective gives it a nonzero gradient.|
|G11|Regularity mismatch|A ReLU→linear PINN Laplacian is rejected at planning; ReLU→tanh is admitted only under an explicit almost-everywhere policy.|
|G12|In-situ rollback|A physical rejection restores parameters, model state, optimizer state, targets and cursors bitwise.|
|G13|Staged external adjoint|The staged VJP matches the all-JAX reference and finite differences; a replay mismatch is refused before the provider runs.|
|G14|Ensemble lanes|Members keep FIXED normalizers on the member lane; serial equals vmap; lane training moves parameters only.|
|G15|Precision floor|An undeclared float32 residual component is refused; a declared floor derives the achievable tolerance; accelerators do not raise the floor.|
|G16|Frozen randomness|Resampled randomness is refused on certified root maps; a frozen realization reproduces primal and derivative.|
|G17|Axis identity|Equal extents never create axis identity; raw outputs are validated only against their declaration; equal-extent ports do not share identity.|
|G18|Closure geometry|ALE closures receive exact unit normals, face measures and grid-normal velocities; Cartesian owners keep parity with the baseline.|
|G19|Differentiable MPC|The closed-loop gradient through active bounds matches finite differences and improves performance; one weak window refuses the complete derivative.|
|G20|Mixed authority|Each authority group trains only from a compatible contribution: rollout signal for DISCRETIZATION, fixed-work signal for ACCELERATOR.|
|G21|Replay derivative|DEM and reactive replay mismatches invalidate cotangents while preserving primal and replay evidence.|
|G22|Identity stability|A dynamic weight update changes the numeric revision, not the executable identity; revisions are verified on load; statically held weights change the executable signature.|
|G23|Field validity|C0 facet gradients are refused without a trace side; explicit trace sides define facet gradients; degenerate second derivatives are refused.|
|G24|Coupled training policy|`CoupledTrainingPolicy` decides whether a valid physical step survives a training rejection.|

Run the gate suite directly with pytest:

```bash
PYTHONPATH=. python -m pytest tests/integration/test_ml_interoperability_qualification.py -q
```

The qualification runner executes the same scenarios and binds each gate's
outcome into content-addressed `phydrax.qualification` records (support tuple,
quantitative criterion, campaign start, raw observation, campaign observation,
and scientific qualification evidence):

```bash
PYTHONPATH=. python tools/ml_interoperability_qualification.py --workers 8 --output report.json
```

`--gate G<N>` (repeatable) restricts the run to a subset; unselected gates are
reported as missing and the report outcome is then inconclusive. Without
`--output` the report is printed. The exit status is nonzero when any selected
gate failed or was inconclusive. An unchanged build, environment, and outcome
set reproduce a byte-identical report.

See [API → Qualification criteria and campaign causality](api/qualification.md).
