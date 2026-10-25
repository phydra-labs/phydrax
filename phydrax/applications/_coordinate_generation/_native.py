# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native fixed-chemistry conditional flow matching; no pretrained capability.

Host preparation/training/persistence are not JIT entry points. The learned
velocity and the prepared sampler's lane transition are differentiable numeric
PyTrees. The sampler is a one-shot, idempotent atomistic generation sampler:
every proposal is a pure function of its caller-derived lane key and typed
case conditions. Integration coordinate 0..1 is generative pseudotime, never
molecular time. This gauge-fixed, singular-support model exposes no coordinate
likelihood or Boltzmann weights, and its fixed atom order is an ABI, not a
permutation-equivariance claim.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import numpy.typing as npt
from jax import Array
from jax.typing import ArrayLike

from phydrax._strict import StrictModule
from phydrax.optim._adam import adam

from ..._external_resource import read_bounded_resource, ResourceLimits
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._identity import NumericRevision
from ...atomistic._conditioning import AtomisticConditioning, AtomisticConditioningSpec
from ...atomistic._generation import (
    AbstractAtomisticGenerationSink,
    AbstractPreparedAtomisticSampler,
    AtomisticConditionSchedule,
    AtomisticGeneratedProposal,
    AtomisticGenerationStreamPlan,
    AtomisticGenerationSupport,
    AtomisticPayloadSpec,
    AtomisticProposalBlock,
    AtomisticProposalStatus,
    role_keys,
)
from ...domain import DomainFunction, HyperRectangle, TimeInterval
from ...domain._evaluation import PointwiseEvaluator
from ...dynamics import ContinuousSystem, StateLayout
from ...ml.artifacts import read_ml_artifact, save_ml_artifact
from ...nn.models import MLP
from ...qualification import ReferenceArtifactManifest
from ...solver import DiffraxEvolution, FunctionalSolver
from ...terms import FlowMatchingTerm
from ...transport import EndpointCouplingSample, LinearEndpointInterpolant
from ...typing import ConvertibleToArray, PRNGKey
from ._decoder import (
    AbstractCoordinateDecoder,
    CartesianCoordinateDecoder,
    SupportsCoordinateDecoding,
)
from ._support import (
    CoordinateGeometryFilter,
    CoordinateProposalQualification,
    PreparedCoordinateSupport,
    qualify_coordinate_proposals,
    require_coordinate_conditioning,
)


def require_coordinate_rights(
    rights: Sequence[ReferenceArtifactManifest],
    *,
    training_use: bool = False,
    commercial_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> tuple[str, ...]:
    """Compose the native requested-use admissions without dropping parent restrictions."""
    if not rights or any(
        not isinstance(item, ReferenceArtifactManifest) for item in rights
    ):
        raise ValueError(
            "Explicit source/parameter/weight rights manifests are required."
        )
    return tuple(
        item.require_rights(
            training_use=training_use,
            commercial_use=commercial_use,
            redistribution=redistribution,
            export=export,
        )
        for item in rights
    )


def _condition_matrix(
    conditioning: AtomisticConditioningSpec,
    conditions: Mapping[str, ConvertibleToArray],
    support: PreparedCoordinateSupport,
    /,
) -> tuple[Array, AtomisticConditioning]:
    """Admit exactly the declared typed case fields; return them in spec order."""
    spec = require_coordinate_conditioning(conditioning, support.resources)
    if not isinstance(conditions, Mapping):
        raise TypeError("Conditions are a mapping from declared field names to values.")
    values = dict(conditions)
    missing = sorted(set(spec.names).difference(values))
    undeclared = sorted(set(values).difference(spec.names))
    if missing or undeclared:
        raise ValueError(
            "Conditions must supply exactly the declared fields; "
            f"missing {missing!r}, undeclared {undeclared!r}."
        )
    first_name = spec.names[0]
    first = values[first_name]
    if not isinstance(first, Array):
        first = np.asarray(first)
        values[first_name] = first
    shape = first.shape
    count = shape[0] if len(shape) == 1 else 0
    if count < 1:
        raise ValueError("Every condition field supplies one scalar value per case.")
    admitted = AtomisticConditioning(
        spec, values, case_count=count, atom_capacity=support.template.atom_capacity
    )
    status = admitted.case_status(jnp.tile(support.template.atom_mask[0], count))
    if bool(jnp.any(status != 0)):
        raise ValueError("Conditions must be finite and inside their declared bounds.")
    matrix = jnp.stack(tuple(admitted.value(name) for name in spec.names), axis=1)
    return matrix.astype(support.template.positions.dtype), admitted


@dataclass(frozen=True)
class CoordinateTrainingData:
    support: PreparedCoordinateSupport
    raw_positions: Array
    canonical_positions: Array
    encoded_coordinates: Array
    decoder: AbstractCoordinateDecoder
    conditions: Array
    conditioning: AtomisticConditioningSpec
    record_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    rights: tuple[ReferenceArtifactManifest, ...]
    corpus_description: str
    dataset_id: str


def prepare_coordinate_training_data(
    support: PreparedCoordinateSupport,
    positions: npt.ArrayLike,
    conditions: Mapping[str, ConvertibleToArray],
    *,
    conditioning: AtomisticConditioningSpec,
    record_ids: Iterable[str],
    source_manifest_ids: Iterable[str],
    split_group_ids: Iterable[str],
    validation_groups: Iterable[str],
    rights: Iterable[ReferenceArtifactManifest],
    corpus_description: str,
    decoder: AbstractCoordinateDecoder | None = None,
    commercial_use: bool = False,
) -> CoordinateTrainingData:
    """Admit mapped conformers with a disjoint caller-defined group split.

    ``conditions`` maps every declared field of ``conditioning`` (required
    continuous scalar case ports with explicit units or dimensionless meaning)
    to one value per record; the spec's canonical field order is the model's
    condition-input order. Group identifiers must encode the corpus's actual
    leakage policy (e.g. independent acquisition/trajectory/construct groups);
    this function can check disjointness and duplicate coordinates, not infer
    biological independence.
    """
    manifests = tuple(rights)
    admitted = require_coordinate_rights(
        manifests, training_use=True, commercial_use=commercial_use
    )
    values = np.asarray(positions)
    count = values.shape[0] if values.ndim else 0
    if (
        values.shape != (count, support.template.atom_capacity, 3)
        or not 2 <= count <= support.resources.max_records
    ):
        raise ValueError(
            "Training coordinates must have bounded shape (record, atom_capacity, 3)."
        )
    context, _ = _condition_matrix(conditioning, conditions, support)
    if context.shape[0] != count:
        raise ValueError("Conditions must be aligned to every record.")
    ids, sources, groups = (
        tuple(record_ids),
        tuple(source_manifest_ids),
        tuple(split_group_ids),
    )
    if any(
        len(items) != count or any(not value for value in items)
        for items in (ids, sources, groups)
    ):
        raise ValueError(
            "Every record needs identity, source manifest, and split-group lineage."
        )
    if len(set(ids)) != count or not set(sources) <= set(admitted):
        raise ValueError("Record IDs must be unique and every source must be admitted.")
    validation = frozenset(validation_groups)
    if not validation or not validation <= set(groups):
        raise ValueError(
            "Validation groups must name an explicit subset of corpus groups."
        )
    train = tuple(i for i, group in enumerate(groups) if group not in validation)
    heldout = tuple(i for i, group in enumerate(groups) if group in validation)
    if not train or not heldout or not corpus_description:
        raise ValueError(
            "Nonempty independent training/validation splits and corpus description are required."
        )
    canonical, gauge_valid = support.canonicalize(
        jnp.asarray(values, dtype=support.template.positions.dtype)
    )
    qualification = qualify_coordinate_proposals(support, canonical)
    if not bool(jnp.all(gauge_valid & qualification.accepted)):
        raise ValueError(
            "Training coordinates fail declared gauge/geometry/chirality qualification."
        )
    prepared_decoder = CartesianCoordinateDecoder(support) if decoder is None else decoder
    if (
        not isinstance(prepared_decoder, AbstractCoordinateDecoder)
        or prepared_decoder.support_id != support.support_id
    ):
        raise ValueError(
            "Coordinate decoder must implement the numeric ABI for this exact support."
        )
    encoding = prepared_decoder.encode(canonical)
    encoded = jnp.asarray(encoding.coordinates, dtype=canonical.dtype)
    if (
        encoded.shape != (count, prepared_decoder.coordinate_size)
        or jnp.asarray(encoding.valid).shape != (count,)
        or not bool(jnp.all(encoding.valid & jnp.all(jnp.isfinite(encoded), axis=1)))
    ):
        raise ValueError(
            "Every training conformer must encode to one finite valid representation."
        )
    coordinate_ids = tuple(
        array_tree_fingerprint(canonical[i])["sha256"] for i in range(count)
    )
    if {coordinate_ids[i] for i in train} & {coordinate_ids[i] for i in heldout}:
        raise ValueError(
            "Identical canonical conformers leak across the validation split."
        )
    dataset_id = canonical_fingerprint(
        {
            "kind": "conditional-coordinate-corpus",
            "support": support.support_id,
            "arrays": array_tree_fingerprint((values, context)),
            "representation": prepared_decoder.representation_id,
            "encoded": array_tree_fingerprint(encoded),
            "conditioning": conditioning.spec_id,
            "condition_names": conditioning.names,
            "records": ids,
            "sources": sources,
            "groups": groups,
            "validation": sorted(validation),
            "rights": admitted,
            "description": corpus_description,
        }
    )
    return CoordinateTrainingData(
        support,
        jnp.asarray(values),
        canonical,
        encoded,
        prepared_decoder,
        context.astype(canonical.dtype),
        conditioning,
        ids,
        sources,
        groups,
        train,
        heldout,
        manifests,
        corpus_description,
        dataset_id,
    )


class ConditionalCoordinateVelocity(StrictModule):
    """Dense global coordinate model with explicit chemical atom/token features.

    Fixed support/order is an ABI, not permutation equivariance. Proper-rigid
    invariance is handled by the declared anchor gauge at the data boundary.
    """

    network: MLP
    decoder: AbstractCoordinateDecoder
    masses: tuple[float, ...] = eqx.field(static=True)
    mask: tuple[bool, ...] = eqx.field(static=True)
    token_features: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    condition_names: tuple[str, ...] = eqx.field(static=True)
    conditioning_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: PreparedCoordinateSupport,
        decoder: AbstractCoordinateDecoder,
        conditioning: AtomisticConditioningSpec,
        network: MLP,
    ) -> None:
        if (
            not isinstance(decoder, AbstractCoordinateDecoder)
            or decoder.support_id != support.support_id
        ):
            raise ValueError("Model decoder must bind the exact coordinate support.")
        spec = require_coordinate_conditioning(conditioning, support.resources)
        if (
            not isinstance(network, MLP)
            or network.in_size != _network_input_size(support, decoder, len(spec.fields))
            or network.out_size != decoder.coordinate_size
        ):
            raise ValueError(
                "Coordinate network must map state, time, conditions, and token "
                "features to the decoder coordinate size."
            )
        self.decoder = decoder
        self.masses = tuple(float(v) for v in np.asarray(support.template.masses[0]))
        self.mask = tuple(bool(v) for v in np.asarray(support.template.atom_mask[0]))
        self.token_features = support.token_features
        self.support_id = support.support_id
        self.coordinate_size = decoder.coordinate_size
        self.representation_id = decoder.representation_id
        self.condition_names = spec.names
        self.conditioning_id = spec.spec_id
        self.network = network

    def center(self, value: ArrayLike) -> Array:
        """Project a one-case model state; name retained for Cartesian callers."""
        return self.decoder.project(value)

    def __call__(self, state: ArrayLike, time: ArrayLike, condition: Array) -> Array:
        features = jnp.concatenate(
            (
                self.center(state),
                jnp.asarray(time).reshape((1,)),
                condition,
                jnp.asarray(self.token_features).reshape((-1,)),
            )
        )
        return self.center(self.network(features))


def _network_input_size(
    support: PreparedCoordinateSupport,
    decoder: AbstractCoordinateDecoder,
    condition_features: int,
) -> int:
    features = support.token_features
    return (
        decoder.coordinate_size
        + 1
        + condition_features
        + len(features) * len(features[0])
    )


def _velocity_function(model: ConditionalCoordinateVelocity) -> DomainFunction:
    dimension = model.coordinate_size
    # Domains describe argument shapes; no clipping of coordinates or conditions.
    domain = HyperRectangle(
        jnp.full((dimension,), -1e12), jnp.full((dimension,), 1e12), label="x"
    )
    domain = domain @ TimeInterval(0.0, 1.0)
    domain = domain @ HyperRectangle(
        jnp.full((len(model.condition_names),), -1e12),
        jnp.full((len(model.condition_names),), 1e12),
        label="condition",
    )
    return domain.Function("x", "t", "condition")(model)


def _endpoints(
    data: CoordinateTrainingData,
    indices: tuple[int, ...],
    key: PRNGKey,
    num_pairs: int,
) -> EndpointCouplingSample:
    noise_key, index_key = jr.split(key)
    index = jnp.asarray(indices)[jr.randint(index_key, (num_pairs,), 0, len(indices))]
    # The standard Gaussian is the exact chosen flow source in representation
    # coordinates, not a molecular equilibrium distribution.
    noise = jr.normal(
        noise_key,
        (num_pairs, data.decoder.coordinate_size),
        dtype=data.encoded_coordinates.dtype,
    )
    source = jax.vmap(data.decoder.project)(noise)
    return EndpointCouplingSample(
        source=source,
        target=data.encoded_coordinates[index],
        source_indices=jnp.arange(num_pairs),
        target_indices=index,
        valid=jnp.ones(num_pairs, dtype=jnp.bool_),
        log_weights=jnp.zeros(num_pairs),
        context={"condition": data.conditions[index]},
        coupling_id=data.dataset_id,
        provenance=(
            "native-standard-Gaussian-to-fixed-representation-independent-coupling:"
            + data.decoder.representation_id
        ),
    )


@dataclass(frozen=True)
class CoordinateFitResult:
    model: ConditionalCoordinateVelocity
    support: PreparedCoordinateSupport
    conditioning: AtomisticConditioningSpec
    dataset_id: str
    rights: tuple[ReferenceArtifactManifest, ...]
    initial_training_loss: float
    final_training_loss: float
    validation_loss: float
    training_steps: int
    fit_id: str
    scientific_claim: str = (
        "trained fixed-chemistry conditional proposals; uncalibrated predictive accuracy"
    )


def fit_coordinate_model(
    data: CoordinateTrainingData,
    *,
    key: PRNGKey,
    steps: int = 200,
    pairs_per_step: int = 32,
    width: int = 64,
    depth: int = 2,
    learning_rate: float = 1e-3,
    commercial_use: bool = False,
) -> CoordinateFitResult:
    """Actually optimize the native flow-matching objective with FunctionalSolver."""
    support = data.support
    require_coordinate_rights(
        data.rights, training_use=True, commercial_use=commercial_use
    )
    for value, limit in (
        (steps, support.resources.max_training_steps),
        (pairs_per_step, support.resources.max_pairs),
        (width, support.resources.max_width),
        (depth, support.resources.max_depth),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= limit
        ):
            raise ValueError("Training configuration exceeds its finite resource policy.")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    model_key, train_key, eval_key, validation_key = jr.split(key, 4)
    model = ConditionalCoordinateVelocity(
        support,
        data.decoder,
        data.conditioning,
        MLP(
            in_size=_network_input_size(
                support, data.decoder, len(data.conditioning.fields)
            ),
            out_size=data.decoder.coordinate_size,
            width_size=width,
            depth=depth,
            key=model_key,
        ),
    )
    interpolant = LinearEndpointInterpolant((data.decoder.coordinate_size,))
    provider = lambda sample_key: _endpoints(
        data,
        data.train_indices,
        jr.fold_in(sample_key, jr.key_data(train_key)[0]),
        pairs_per_step,
    )
    term = FlowMatchingTerm("velocity", provider, interpolant, sampling_mode="resample")
    evaluation = FlowMatchingTerm(
        "velocity",
        _endpoints(data, data.train_indices, eval_key, pairs_per_step),
        interpolant,
    )
    validation = FlowMatchingTerm(
        "velocity",
        _endpoints(data, data.validation_indices, validation_key, pairs_per_step),
        interpolant,
    )
    function = _velocity_function(model)
    initial = float(evaluation.loss({"velocity": function}, key=eval_key))
    solver = FunctionalSolver(functions={"velocity": function}, terms=(term,))
    fitted = solver.solve(
        num_iter=steps,
        optim=adam(learning_rate),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    learned_function = fitted.functions["velocity"]
    learned_evaluator = learned_function.func
    # _velocity_function binds the velocity model through a PointwiseEvaluator.
    if not (isinstance(learned_evaluator, PointwiseEvaluator)):
        raise RuntimeError(
            "Internal invariant failed: isinstance(learned_evaluator, PointwiseEvaluator)."
        )
    learned = learned_evaluator.function
    if not (isinstance(learned, ConditionalCoordinateVelocity)):
        raise RuntimeError(
            "Internal invariant failed: isinstance(learned, ConditionalCoordinateVelocity)."
        )
    final = float(evaluation.loss({"velocity": learned_function}, key=eval_key))
    heldout = float(validation.loss({"velocity": learned_function}, key=validation_key))
    if not np.isfinite((initial, final, heldout)).all():
        raise FloatingPointError(
            "Coordinate training produced nonfinite loss; no model is admitted."
        )
    fit_id = canonical_fingerprint(
        {
            "kind": "native-conditional-coordinate-fit",
            "dataset": data.dataset_id,
            "representation": data.decoder.representation_id,
            "conditioning": data.conditioning.spec_id,
            "weights": array_tree_fingerprint(learned),
            "steps": steps,
            "pairs": pairs_per_step,
            "width": width,
            "depth": depth,
            "learning_rate": learning_rate,
            "key": np.asarray(jr.key_data(key)).tolist(),
        }
    )
    return CoordinateFitResult(
        learned,
        support,
        data.conditioning,
        data.dataset_id,
        data.rights,
        initial,
        final,
        heldout,
        steps,
        fit_id,
    )


class _CoordinateField(StrictModule):
    model: ConditionalCoordinateVelocity

    def __call__(self, time: ArrayLike, state: Array, condition: Array) -> Array:
        return self.model(state, time, condition)


@final
class CoordinateLaneState(StrictModule):
    """Numeric one-shot continuation of every sampler lane (leading lane axis).

    ``raw`` is the final ODE state in decoder representation coordinates;
    ``conditions`` are the consumed case conditions in the model's input order;
    ``condition_status`` holds `AtomisticConditionStatus` bits per lane.
    """

    raw: Array
    solver_valid: Array
    solver_status: Array
    conditions: Array
    condition_status: Array


_BASE_PAYLOAD = (
    "raw_coordinates",
    "solver_valid",
    "solver_status",
    "conditions",
    "condition_status",
    "canonical_positions",
    "gauge_valid",
)
_DECODER_PAYLOAD_PREFIX = "decoder"


def _decoder_layout(
    decoder: AbstractCoordinateDecoder, atom_capacity: int, dtype: np.dtype, /
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    """Named per-row leaves of the decoder evidence (lane axis excluded)."""
    probe = jax.ShapeDtypeStruct((2, decoder.coordinate_size), dtype)
    evidence = jax.eval_shape(decoder.decode, probe)
    leaves, _ = jax.tree_util.tree_flatten_with_path(evidence)
    entries = tuple(
        (
            _DECODER_PAYLOAD_PREFIX + jax.tree_util.keystr(path),
            tuple(leaf.shape[1:]),
            np.dtype(leaf.dtype).name,
        )
        for path, leaf in leaves
    )
    if (
        evidence.positions.shape != (2, atom_capacity, 3)
        or evidence.valid.shape != (2,)
        or any(leaf.shape[:1] != (2,) for _, leaf in leaves)
    ):
        raise ValueError(
            "Decoder evidence must retain one Cartesian result, validity bit and "
            "evidence row per proposal."
        )
    return entries


@final
class PreparedCoordinateSampler(AbstractPreparedAtomisticSampler):
    """One-shot, idempotent conditional flow sampler on a fixed coordinate support.

    Each lane integrates the fitted velocity with the native `DiffraxEvolution`
    from the standard Gaussian source drawn from ``role_keys(lane_keys,
    "source-noise")``; ordinals enter only through the caller's lane keys.
    The emitted block carries the decoded Cartesian proposal and, as payload,
    raw ODE coordinates, solver validity/status, consumed conditions, the
    canonical gauge view and every decoder evidence leaf. Status bits mark
    solver failure, nonfinite raw states, decoder invalidity, a degenerate
    canonical gauge and invalid conditions; nothing is repaired or resampled.
    """

    evolution: DiffraxEvolution
    model: ConditionalCoordinateVelocity
    coordinate_support: PreparedCoordinateSupport
    support: AtomisticGenerationSupport
    payload_spec: AtomisticPayloadSpec
    decoder_payload: tuple[str, ...] = eqx.field(static=True)
    fit_id: str = eqx.field(static=True)
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    max_steps: int = eqx.field(static=True)
    sampler_id: str = eqx.field(static=True)
    stateful: bool = eqx.field(static=True)
    idempotent: bool = eqx.field(static=True)

    def __init__(
        self,
        model: ConditionalCoordinateVelocity,
        coordinate_support: PreparedCoordinateSupport,
        conditioning: AtomisticConditioningSpec,
        /,
        *,
        fit_id: str,
        rtol: float,
        atol: float,
        max_steps: int,
    ) -> None:
        if not isinstance(model, ConditionalCoordinateVelocity):
            raise TypeError("model must be a ConditionalCoordinateVelocity.")
        if model.support_id != coordinate_support.support_id:
            raise ValueError("Learned model and prepared chemical support differ.")
        if model.conditioning_id != conditioning.spec_id:
            raise ValueError("Learned model and typed conditioning spec differ.")
        tolerances = (float(rtol), float(atol))
        if not np.all(np.isfinite(tolerances)) or min(tolerances) <= 0.0:
            raise ValueError("ODE tolerances must be finite and positive.")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not 1 <= max_steps <= coordinate_support.resources.max_solver_steps
        ):
            raise ValueError("ODE step capacity exceeds the finite resource policy.")
        if not isinstance(fit_id, str) or not fit_id:
            raise ValueError("fit_id must be a non-empty string.")
        support = coordinate_support.generation_support(conditioning)
        template = coordinate_support.template
        dtype = np.dtype(template.positions.dtype)
        decoder_entries = _decoder_layout(model.decoder, template.atom_capacity, dtype)
        payload = AtomisticPayloadSpec(
            (
                ("raw_coordinates", (model.coordinate_size,), dtype.name),
                ("solver_valid", (), "bool"),
                ("solver_status", (), "int32"),
                ("conditions", (len(model.condition_names),), dtype.name),
                ("condition_status", (), "int32"),
                ("canonical_positions", (template.atom_capacity, 3), dtype.name),
                ("gauge_valid", (), "bool"),
                *decoder_entries,
            )
        )
        system = ContinuousSystem(
            _CoordinateField(model),
            state_layout=StateLayout((model.coordinate_size,)),
            system_id=fit_id,
        )
        self.evolution = DiffraxEvolution(
            system, rtol=tolerances[0], atol=tolerances[1], max_steps=max_steps
        )
        self.model = model
        self.coordinate_support = coordinate_support
        self.support = support
        self.payload_spec = payload
        self.decoder_payload = tuple(name for name, _, _ in decoder_entries)
        self.fit_id = fit_id
        self.rtol, self.atol = tolerances
        self.max_steps = max_steps
        self.stateful = False
        self.idempotent = True
        self.sampler_id = canonical_fingerprint(
            {
                "kind": "native-conditional-coordinate-sampler",
                "fit": fit_id,
                "coordinate_support": coordinate_support.support_id,
                "generation_support": support.support_id,
                "representation": model.representation_id,
                "conditioning": conditioning.spec_id,
                "source": "standard-gaussian-in-representation-coordinates",
                "pseudotime": [0.0, 1.0],
                "rtol": tolerances[0],
                "atol": tolerances[1],
                "max_steps": max_steps,
                "payload": payload.spec_id,
            }
        )

    @property
    def decoder(self) -> AbstractCoordinateDecoder:
        return self.model.decoder

    def _dtype(self) -> np.dtype:
        return np.dtype(self.coordinate_support.template.positions.dtype)

    def vacant_method(self, lanes: int, /) -> CoordinateLaneState:
        dtype = self._dtype()
        return CoordinateLaneState(
            jnp.zeros((lanes, self.model.coordinate_size), dtype=dtype),
            jnp.zeros((lanes,), dtype=jnp.bool_),
            jnp.zeros((lanes,), dtype=jnp.int32),
            jnp.zeros((lanes, len(self.model.condition_names)), dtype=dtype),
            jnp.zeros((lanes,), dtype=jnp.int32),
        )

    def initialize_method(
        self,
        keys: Array,
        ordinals: Array,
        conditioning: AtomisticConditioning | None,
        selected: Array,
        /,
    ) -> CoordinateLaneState:
        # The proposal is a pure function of (lane key, conditioning); the
        # caller addresses the ordinal through the lane key.  Every lane is
        # integrated in one vmapped flow solve; `admit` keeps admitted rows.
        del ordinals, selected
        if not isinstance(conditioning, AtomisticConditioning):
            raise ValueError("Coordinate proposals require typed case conditioning.")
        values = jnp.stack(
            tuple(conditioning.value(name) for name in self.model.condition_names),
            axis=-1,
        ).astype(self._dtype())
        lanes = values.shape[0]
        mask = jnp.tile(self.coordinate_support.template.atom_mask[0], lanes)
        condition_status = conditioning.case_status(mask).astype(jnp.int32)
        noise_keys = role_keys(keys, "source-noise")
        return _solve_lanes(self, noise_keys, values, condition_status)

    def emit_method(
        self, method: CoordinateLaneState, ordinals: Array, /
    ) -> AtomisticProposalBlock:
        decoded = self.decoder.decode(method.raw)
        positions = jnp.asarray(decoded.positions)
        canonical, gauge_valid = self.coordinate_support.canonicalize(positions)
        decoder_valid = jnp.asarray(decoded.valid, dtype=jnp.bool_)
        finite = jnp.all(jnp.isfinite(method.raw), axis=-1)
        status = (
            jnp.where(
                method.solver_valid, 0, int(AtomisticProposalStatus.SAMPLER_FAILURE)
            )
            | jnp.where(finite, 0, int(AtomisticProposalStatus.NONFINITE))
            | jnp.where(decoder_valid, 0, int(AtomisticProposalStatus.DECODER_INVALID))
            | jnp.where(gauge_valid, 0, int(AtomisticProposalStatus.GEOMETRY_INVALID))
            | jnp.where(
                method.condition_status == 0,
                0,
                int(AtomisticProposalStatus.CONDITION_INVALID),
            )
        ).astype(jnp.int32)
        lanes = positions.shape[0]
        capacity = self.support.atom_capacity
        species = jnp.broadcast_to(
            jnp.asarray(self.support.fixed_species, dtype=jnp.int32), (lanes, capacity)
        )
        mask = jnp.broadcast_to(
            jnp.asarray(self.support.fixed_mask, dtype=jnp.bool_), (lanes, capacity)
        )
        return AtomisticProposalBlock(
            ordinals,
            positions,
            species,
            mask,
            None,
            status,
            (
                method.raw,
                method.solver_valid,
                method.solver_status,
                method.conditions,
                method.condition_status,
                canonical,
                gauge_valid,
                *jax.tree.leaves(decoded),
            ),
            sampler_id=self.sampler_id,
            payload_names=self.payload_spec.names,
        )

    def propose(
        self,
        keys: Array,
        ordinals: ArrayLike,
        conditioning: AtomisticConditioning,
        /,
    ) -> AtomisticProposalBlock:
        """Numeric JIT/grad boundary: one emitted proposal row per lane key."""
        ordinal = jnp.asarray(ordinals).astype(jnp.int64)
        everyone = jnp.ones(ordinal.shape, dtype=jnp.bool_)
        return self.emit_method(
            self.initialize_method(keys, ordinal, conditioning, everyone), ordinal
        )

    def decoder_evidence(
        self, leaves: Sequence[ArrayLike], /
    ) -> SupportsCoordinateDecoding:
        """Rebuild the decoder's own evidence object from its published payload leaves."""
        values = tuple(jnp.asarray(value) for value in leaves)
        if len(values) != len(self.decoder_payload) or not values:
            raise ValueError("Decoder evidence needs every declared payload leaf.")
        probe = jax.ShapeDtypeStruct(
            (values[0].shape[0], self.model.coordinate_size), self._dtype()
        )
        structure = jax.tree.structure(jax.eval_shape(self.decoder.decode, probe))
        return jax.tree.unflatten(structure, values)

    def signature_facts(self) -> dict[str, str | int]:
        return {
            "kind": "native-conditional-coordinate-flow",
            "representation": self.model.representation_id,
            "coordinate_size": self.model.coordinate_size,
            "atom_capacity": self.support.atom_capacity,
            "condition_features": len(self.model.condition_names),
            "dtype": self._dtype().name,
            "evolution": self.evolution.evolution_id,
            "rtol": repr(self.rtol),
            "atol": repr(self.atol),
            "max_steps": self.max_steps,
        }

    def numeric_revision(self) -> NumericRevision:
        return NumericRevision(
            self.sampler_id,
            {
                "network": tuple(
                    jax.tree.leaves(eqx.filter(self.model.network, eqx.is_array))
                )
            },
        )


@eqx.filter_jit
def _solve_lanes(
    sampler: PreparedCoordinateSampler,
    noise_keys: Array,
    values: Array,
    condition_status: Array,
    /,
) -> CoordinateLaneState:
    dtype = values.dtype
    size = sampler.model.coordinate_size
    # Exactly the standard Gaussian source of the fitted flow, one draw per lane.
    noise = jax.vmap(lambda key: jr.normal(key, (size,), dtype=dtype))(noise_keys)
    initial = jax.vmap(sampler.model.center)(noise)

    def one(state: Array, condition: Array) -> tuple[Array, Array, Array]:
        result = sampler.evolution.advance(state, 0.0, 1.0, condition)
        return result.final_state, result.valid, result.status

    raw, valid, status = jax.vmap(one)(initial, values)
    return CoordinateLaneState(
        raw,
        jnp.asarray(valid, dtype=jnp.bool_),
        jnp.asarray(status, dtype=jnp.int32),
        values,
        condition_status,
    )


def prepare_coordinate_sampler(
    fit: CoordinateFitResult,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_steps: int = 1024,
    commercial_use: bool = False,
    export: bool = False,
) -> PreparedCoordinateSampler:
    require_coordinate_rights(fit.rights, commercial_use=commercial_use, export=export)
    return PreparedCoordinateSampler(
        fit.model,
        fit.support,
        fit.conditioning,
        fit_id=fit.fit_id,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )


@dataclass(frozen=True)
class CoordinateProposalBatch:
    """Host materialization of one published, ordinal-contiguous proposal run.

    Every requested ordinal is present, accepted or not; ``proposal_status``
    keeps the `AtomisticProposalStatus` bits and ``filter_bits`` the stream's
    filter bits of every rejection.
    """

    raw_positions: Array
    canonical_positions: Array
    conditions: Array
    condition_names: tuple[str, ...]
    solver_valid: Array
    solver_status: Array
    qualification: CoordinateProposalQualification
    proposal_status: Array
    filter_bits: Array
    ordinals: tuple[int, ...]
    sample_ids: tuple[str, ...]
    campaign_id: str
    sampler_id: str
    generation_support_id: str
    parent_fit_id: str
    rights: tuple[ReferenceArtifactManifest, ...]
    raw_coordinates: Array
    decoder_valid: Array
    decoder_residuals: Array
    decoder_representation_id: str
    decoder_evidence: SupportsCoordinateDecoding
    confidence: None = None
    confidence_semantics: str = "uncalibrated; geometry validity is not sample confidence"
    likelihood_capability: str = (
        "unavailable on constrained rigid-gauge coordinate support"
    )
    time_semantics: str = "integration coordinate 0..1 is generative pseudotime, not molecular-dynamics time"
    symmetry_semantics: str = (
        "fixed atom order is a support ABI; no permutation-equivariance claim"
    )


def _proposal_batch(
    fit: CoordinateFitResult,
    sampler: PreparedCoordinateSampler,
    geometry: CoordinateGeometryFilter,
    campaign_id: str,
    records: Sequence[AtomisticGeneratedProposal],
    /,
) -> CoordinateProposalBatch:
    def stacked(name: str) -> Array:
        return jnp.asarray(np.stack(tuple(record.value(name) for record in records)))

    evidence = sampler.decoder_evidence(
        tuple(stacked(name) for name in sampler.decoder_payload)
    )
    status = jnp.asarray([record.status for record in records], dtype=jnp.int32)
    codes = np.stack(tuple(record.filter_values for record in records))
    return CoordinateProposalBatch(
        raw_positions=jnp.asarray(evidence.positions),
        canonical_positions=stacked("canonical_positions"),
        conditions=stacked("conditions"),
        condition_names=sampler.model.condition_names,
        solver_valid=stacked("solver_valid"),
        solver_status=stacked("solver_status"),
        qualification=geometry.qualification(codes[:, 0], status),
        proposal_status=status,
        filter_bits=jnp.asarray(
            [record.filter_bits for record in records], dtype=jnp.int32
        ),
        ordinals=tuple(record.ordinal for record in records),
        sample_ids=tuple(record.proposal_id for record in records),
        campaign_id=campaign_id,
        sampler_id=sampler.sampler_id,
        generation_support_id=sampler.support.support_id,
        parent_fit_id=fit.fit_id,
        rights=fit.rights,
        raw_coordinates=stacked("raw_coordinates"),
        decoder_valid=jnp.asarray(evidence.valid, dtype=jnp.bool_),
        decoder_residuals=jnp.asarray(evidence.residuals),
        decoder_representation_id=sampler.decoder.representation_id,
        decoder_evidence=evidence,
    )


def sample_coordinate_proposals(
    fit: CoordinateFitResult,
    key: PRNGKey,
    conditions: Mapping[str, ConvertibleToArray],
    *,
    campaign_id: str,
    start_ordinal: int,
    lanes_per_member: int,
    sink: AbstractAtomisticGenerationSink,
    commercial_use: bool = False,
    export: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-7,
    max_steps: int = 1024,
) -> CoordinateProposalBatch:
    """Run one bounded proposal stream and materialize every published proposal.

    Condition row ``i`` is ordinal ``start_ordinal + i``. Each proposal's key is
    derived from ``(key, campaign_id, support, sampler, ordinal)``, never from
    lane or chunk order, so its id and numbers do not depend on
    ``lanes_per_member`` or on how a request is split into ordinal chunks.
    Every proposal, accepted or rejected with its status/filter bits, is
    published to the explicit ``sink`` in ordinal order and returned on the
    host; nothing is repaired or resampled. Sample ids are
    `atomistic_proposal_id(campaign_id, support_id, sampler_id, ordinal)`.
    """
    sampler = prepare_coordinate_sampler(
        fit,
        commercial_use=commercial_use,
        export=export,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    resources = fit.support.resources
    if (
        isinstance(lanes_per_member, bool)
        or not isinstance(lanes_per_member, int)
        or not 1 <= lanes_per_member <= resources.max_samples
    ):
        raise ValueError("Resident proposal lanes exceed the finite resource policy.")
    if (
        isinstance(start_ordinal, bool)
        or not isinstance(start_ordinal, int)
        or start_ordinal < 0
    ):
        raise ValueError("start_ordinal must be a nonnegative integer.")
    if not isinstance(sink, AbstractAtomisticGenerationSink):
        raise TypeError("Proposals publish to an explicit generation sink.")
    context, admitted = _condition_matrix(fit.conditioning, conditions, fit.support)
    count = context.shape[0]
    if count > resources.max_samples:
        raise ValueError("Sampling count exceeds its finite resource policy.")
    geometry = CoordinateGeometryFilter(fit.support)
    schedule = AtomisticConditionSchedule(
        fit.conditioning,
        {name: admitted.value(name) for name in fit.conditioning.names},
        policy="bounded",
        first_ordinal=start_ordinal,
    )
    stream = AtomisticGenerationStreamPlan(
        sampler,
        campaign_id=campaign_id,
        lanes_per_member=lanes_per_member,
        # One resident run in the bank while the next one is admitted.
        pending_capacity=2 * lanes_per_member,
        conditions=schedule,
        filters=(geometry,),
    ).prepare()
    state = stream.start(
        key, target=count, attempt_budget=count, first_ordinal=start_ordinal
    )
    published: list[AtomisticGeneratedProposal] = []
    for _ in range(-(-count // lanes_per_member) + 2):
        if stream.finished(state):
            break
        state = stream.transition(state)
        ready = stream.ready(state)
        if not ready:
            continue
        receipt = sink.publish(ready)
        if receipt is None:
            raise RuntimeError(
                "The generation sink refused publication; no partial batch is returned."
            )
        state = stream.commit(state, receipt)
        published.extend(ready[: receipt.count])
    if not stream.finished(state) or len(published) != count:
        raise RuntimeError(
            "The bounded proposal stream did not publish every requested ordinal."
        )
    return _proposal_batch(fit, sampler, geometry, campaign_id, published)


def save_coordinate_model(
    path: str | Path,
    fit: CoordinateFitResult,
    *,
    commercial_use: bool = False,
    redistribution: bool = False,
    export: bool = False,
) -> Path:
    """Archive the trained network; the fixed support and decoder stay caller-owned.

    Uses native pickle-free ML artifacts and retains every inherited restriction.
    """
    require_coordinate_rights(
        fit.rights,
        commercial_use=commercial_use,
        redistribution=redistribution,
        export=export,
    )
    return save_ml_artifact(
        path,
        fit.model.network,
        provenance={
            "conditioning": {
                "support_id": fit.support.support_id,
                "representation_id": fit.model.representation_id,
                "coordinate_size": fit.model.coordinate_size,
                "conditions": list(fit.model.condition_names),
                "conditioning_spec_id": fit.conditioning.spec_id,
                "fields": [
                    {
                        "name": field.name,
                        "field_id": field.field_id,
                        "port_id": field.port.port_id,
                        "semantic_id": field.port.semantic_id,
                        "unit_id": None if field.unit is None else field.unit.unit_id,
                    }
                    for field in fit.conditioning.fields
                ],
            },
            "coordinates": {
                "representation_id": fit.model.representation_id,
                "length_unit": fit.support.template.scale.length_unit.to_dict(),
                "meaning": "structural-proposals-not-Boltzmann-samples",
            },
            "fit_id": fit.fit_id,
            "dataset_id": fit.dataset_id,
            "representation_id": fit.model.representation_id,
            "steps": fit.training_steps,
            "initial_training_loss": fit.initial_training_loss,
            "final_training_loss": fit.final_training_loss,
            "validation_loss": fit.validation_loss,
            "rights": [item.to_record() for item in fit.rights],
        },
        licenses=tuple(sorted({item.license_id for item in fit.rights})),
    )


def load_coordinate_model(
    path: str | PathLike[str],
    support: PreparedCoordinateSupport,
    *,
    weight_rights: ReferenceArtifactManifest,
    conditioning: AtomisticConditioningSpec,
    decoder: AbstractCoordinateDecoder | None = None,
    commercial_use: bool = False,
    export: bool = False,
) -> CoordinateFitResult:
    """Load admitted, checksum-bound weights; never fetch a checkpoint or provider.

    The archived network is rebound to the caller's prepared `support`, typed
    `conditioning` spec and `decoder` (default: the Cartesian decoder of
    `support`); all must match the support, condition ports and representation
    the network was trained on.
    """
    require_coordinate_rights(
        (weight_rights,), commercial_use=commercial_use, export=export
    )
    source = Path(path).expanduser().absolute()
    resource = read_bounded_resource(
        source.name,
        trusted_root=source.parent,
        limits=ResourceLimits(weight_rights.size_bytes, 64, 1, 0, 0),
    )
    payload = resource.data
    digest = hashlib.new(weight_rights.checksum_algorithm, payload).hexdigest()
    if digest != weight_rights.checksum or len(payload) != weight_rights.size_bytes:
        raise ValueError("Weight bytes do not match their rights manifest.")
    artifact = read_ml_artifact(source)
    metadata = artifact.manifest.provenance
    recorded = metadata["conditioning"]
    decoder_ = CartesianCoordinateDecoder(support) if decoder is None else decoder
    spec = require_coordinate_conditioning(conditioning, support.resources)
    if (
        recorded["support_id"] != support.support_id
        or recorded["representation_id"] != decoder_.representation_id
    ):
        raise ValueError(
            "Checkpoint is not the requested fixed-chemistry coordinate model."
        )
    if (
        recorded.get("conditioning_spec_id") != spec.spec_id
        or tuple(recorded["conditions"]) != spec.names
    ):
        raise ValueError(
            "Checkpoint was trained on other typed condition ports than requested."
        )
    model = ConditionalCoordinateVelocity(support, decoder_, spec, artifact.model)
    parents = tuple(
        ReferenceArtifactManifest.from_record(item) for item in metadata["rights"]
    )
    rights = (*parents, weight_rights)
    require_coordinate_rights(rights, commercial_use=commercial_use, export=export)
    return CoordinateFitResult(
        model,
        support,
        spec,
        metadata["dataset_id"],
        rights,
        metadata["initial_training_loss"],
        metadata["final_training_loss"],
        metadata["validation_loss"],
        metadata["steps"],
        metadata["fit_id"],
    )
