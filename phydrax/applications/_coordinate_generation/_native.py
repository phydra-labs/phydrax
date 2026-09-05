# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native fixed-chemistry conditional flow matching; no pretrained capability.

Host preparation/training/persistence are not JIT entry points. The learned
velocity and the prepared sampler are differentiable numeric PyTrees. Integration
coordinate 0..1 is generative pseudotime, never molecular time. This gauge-fixed,
singular-support model exposes no coordinate likelihood or Boltzmann weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...domain import HyperRectangle, TimeInterval
from ...dynamics import ContinuousSystem, StateLayout
from ...ml.artifacts import read_ml_artifact, save_ml_artifact
from ...nn.models import MLP
from ...qualification import ReferenceArtifactManifest
from ...solver import DiffraxEvolution, FunctionalSolver
from ...terms import FlowMatchingTerm
from ...transport import EndpointCouplingSample, LinearEndpointInterpolant
from ._support import PreparedCoordinateSupport, qualify_coordinate_proposals


def require_coordinate_rights(
    rights,
    *,
    training_use=False,
    commercial_use=False,
    redistribution=False,
    export=False,
):
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


@dataclass(frozen=True)
class CoordinateTrainingData:
    support: PreparedCoordinateSupport
    raw_positions: object
    canonical_positions: object
    conditions: object
    condition_names: tuple[str, ...]
    record_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    split_group_ids: tuple[str, ...]
    train_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    rights: tuple[ReferenceArtifactManifest, ...]
    corpus_description: str
    dataset_id: str


def prepare_coordinate_training_data(
    support,
    positions,
    conditions,
    *,
    condition_names,
    record_ids,
    source_manifest_ids,
    split_group_ids,
    validation_groups,
    rights,
    corpus_description,
    commercial_use=False,
):
    """Admit mapped conformers with a disjoint caller-defined group split.

    Group identifiers must encode the corpus's actual leakage policy (e.g.
    independent acquisition/trajectory/construct groups); this function can check
    disjointness and duplicate coordinates, not infer biological independence.
    """
    manifests = tuple(rights)
    admitted = require_coordinate_rights(
        manifests, training_use=True, commercial_use=commercial_use
    )
    values = np.asarray(positions)
    context = np.asarray(conditions)
    count = values.shape[0] if values.ndim else 0
    if (
        values.shape != (count, support.template.atom_capacity, 3)
        or not 2 <= count <= support.resources.max_records
    ):
        raise ValueError(
            "Training coordinates must have bounded shape (record, atom_capacity, 3)."
        )
    names = tuple(condition_names)
    if (
        not names
        or len(names) > support.resources.max_condition_features
        or len(set(names)) != len(names)
        or any(not name for name in names)
    ):
        raise ValueError(
            "Condition features require bounded, unique nonempty semantic names."
        )
    if context.shape != (count, len(names)) or not np.isfinite(context).all():
        raise ValueError("Conditions must be finite and aligned to every record.")
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
            "condition_names": names,
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
        jnp.asarray(context, dtype=canonical.dtype),
        names,
        ids,
        sources,
        groups,
        train,
        heldout,
        manifests,
        corpus_description,
        dataset_id,
    )


class ConditionalCoordinateVelocity(eqx.Module):
    """Dense global coordinate model with explicit chemical atom/token features.

    Fixed support/order is an ABI, not permutation equivariance. Proper-rigid
    invariance is handled by the declared anchor gauge at the data boundary.
    """

    network: MLP
    masses: tuple[float, ...] = eqx.field(static=True)
    mask: tuple[bool, ...] = eqx.field(static=True)
    token_features: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    condition_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, support, condition_names, *, width, depth, key):
        self.masses = tuple(float(v) for v in np.asarray(support.template.masses[0]))
        self.mask = tuple(bool(v) for v in np.asarray(support.template.atom_mask[0]))
        self.token_features = support.token_features
        self.support_id = support.support_id
        self.condition_names = tuple(condition_names)
        feature_size = len(self.token_features) * len(self.token_features[0])
        self.network = MLP(
            in_size=support.dimension + 1 + len(condition_names) + feature_size,
            out_size=support.dimension,
            width_size=width,
            depth=depth,
            key=key,
        )

    def center(self, value):
        positions = value.reshape((len(self.mask), 3))
        mask = jnp.asarray(self.mask)[:, None]
        masses = jnp.where(jnp.asarray(self.mask), jnp.asarray(self.masses), 0.0)
        clean = jnp.where(mask, positions, 0.0)
        center = jnp.sum(clean * masses[:, None], axis=0) / jnp.sum(masses)
        return jnp.where(mask, clean - center, 0.0).reshape((-1,))

    def __call__(self, state, time, condition):
        features = jnp.concatenate(
            (
                self.center(state),
                jnp.asarray(time).reshape((1,)),
                condition,
                jnp.asarray(self.token_features).reshape((-1,)),
            )
        )
        return self.center(self.network(features))


def _velocity_function(model):
    dimension = len(model.mask) * 3
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


def _endpoints(data, indices, key, num_pairs):
    noise_key, index_key = jr.split(key)
    support = data.support
    index = jnp.asarray(indices)[jr.randint(index_key, (num_pairs,), 0, len(indices))]
    # The VP owner supplies its declared Gaussian reference. Here that law is an
    # exact chosen flow source, NOT a claim that finite-time VP lost all signal.
    noise = (
        support.source_law.sample(noise_key, (num_pairs,))
        .astype(support.template.positions.dtype)
        .reshape((num_pairs, support.template.atom_capacity, 3))
    )
    source = support.center(noise).reshape((num_pairs, support.dimension))
    return EndpointCouplingSample(
        source=source,
        target=data.canonical_positions[index].reshape((num_pairs, support.dimension)),
        source_indices=jnp.arange(num_pairs),
        target_indices=index,
        valid=jnp.ones(num_pairs, dtype=bool),
        log_weights=jnp.zeros(num_pairs),
        context={"condition": data.conditions[index]},
        coupling_id=data.dataset_id,
        provenance="native-Gaussian-to-canonical-coordinate-independent-coupling",
    )


@dataclass(frozen=True)
class CoordinateFitResult:
    model: ConditionalCoordinateVelocity
    support: PreparedCoordinateSupport
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
    data,
    *,
    key,
    steps=200,
    pairs_per_step=32,
    width=64,
    depth=2,
    learning_rate=1e-3,
    commercial_use=False,
):
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
        support, data.condition_names, width=width, depth=depth, key=model_key
    )
    interpolant = LinearEndpointInterpolant((support.dimension,))
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
        optim=optax.adam(learning_rate),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    learned_function = fitted.functions["velocity"]
    learned = learned_function.func.function
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
        data.dataset_id,
        data.rights,
        initial,
        final,
        heldout,
        steps,
        fit_id,
    )


class _CoordinateField(eqx.Module):
    model: ConditionalCoordinateVelocity

    def __call__(self, time, state, condition):
        return self.model(state, time, condition)


class PreparedCoordinateSampler(eqx.Module):
    evolution: DiffraxEvolution
    model: ConditionalCoordinateVelocity
    source_law: object
    max_samples: int = eqx.field(static=True)

    def __call__(self, key, conditions):
        """Numeric JIT/grad boundary: returns every state, valid bit and status."""
        conditions = jnp.asarray(conditions)
        if conditions.ndim != 2 or conditions.shape[1] != len(self.model.condition_names):
            raise ValueError(
                "Sampling conditions must have shape (sample, condition_feature)."
            )
        count = conditions.shape[0]
        if not 1 <= count <= self.max_samples:
            raise ValueError("Sampling count exceeds its finite resource policy.")
        conditions = eqx.error_if(
            conditions,
            ~jnp.all(jnp.isfinite(conditions)),
            "Sampling conditions must be finite.",
        )
        # This is exactly the chosen standard Gaussian law used by the training source.
        noise = self.source_law.sample(key, (count,)).astype(conditions.dtype)
        initial = jax.vmap(self.model.center)(noise)

        def one(state, condition):
            result = self.evolution.advance(state, 0.0, 1.0, condition)
            return (
                result.final_state.reshape((len(self.model.mask), 3)),
                result.valid,
                result.status,
            )

        return jax.vmap(one)(initial, conditions)


def prepare_coordinate_sampler(
    fit, *, rtol=1e-5, atol=1e-7, max_steps=1024, commercial_use=False, export=False
):
    require_coordinate_rights(fit.rights, commercial_use=commercial_use, export=export)
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= fit.support.resources.max_solver_steps
    ):
        raise ValueError("ODE step capacity exceeds the finite resource policy.")
    if fit.model.support_id != fit.support.support_id:
        raise ValueError("Learned model and prepared chemical support differ.")
    system = ContinuousSystem(
        _CoordinateField(fit.model),
        state_layout=StateLayout((fit.support.dimension,)),
        system_id=fit.fit_id,
    )
    return PreparedCoordinateSampler(
        DiffraxEvolution(system, rtol=rtol, atol=atol, max_steps=max_steps),
        fit.model,
        fit.support.source_law,
        fit.support.resources.max_samples,
    )


@dataclass(frozen=True)
class CoordinateProposalBatch:
    raw_positions: object
    canonical_positions: object
    conditions: object
    solver_valid: object
    solver_status: object
    qualification: object
    sample_ids: tuple[str, ...]
    parent_fit_id: str
    rights: tuple[ReferenceArtifactManifest, ...]
    confidence: None = None
    confidence_semantics: str = "uncalibrated; geometry validity is not sample confidence"
    likelihood_capability: str = (
        "unavailable on constrained rigid-gauge coordinate support"
    )


def sample_coordinate_proposals(
    fit,
    key,
    conditions,
    *,
    commercial_use=False,
    export=False,
    rtol=1e-5,
    atol=1e-7,
    max_steps=1024,
):
    """Host all-sample materialization; raw ODE states and canonical views stay distinct."""
    sampler = prepare_coordinate_sampler(
        fit,
        commercial_use=commercial_use,
        export=export,
        rtol=rtol,
        atol=atol,
        max_steps=max_steps,
    )
    context = jnp.asarray(conditions, dtype=fit.support.template.positions.dtype)
    raw, valid, status = eqx.filter_jit(sampler)(key, context)
    canonical, gauge_valid = fit.support.canonicalize(raw)
    qualification = qualify_coordinate_proposals(
        fit.support, canonical, solver_valid=valid & gauge_valid
    )
    batch_id = canonical_fingerprint(
        {
            "fit": fit.fit_id,
            "key": np.asarray(jr.key_data(key)).tolist(),
            "conditions": array_tree_fingerprint(context),
        }
    )
    return CoordinateProposalBatch(
        raw,
        canonical,
        context,
        valid,
        status,
        qualification,
        tuple(f"{batch_id}:{i}" for i in range(context.shape[0])),
        fit.fit_id,
        fit.rights,
    )


def save_coordinate_model(
    path, fit, *, commercial_use=False, redistribution=False, export=False
):
    """Use native pickle-free ML artifacts; retain every inherited restriction."""
    require_coordinate_rights(
        fit.rights,
        commercial_use=commercial_use,
        redistribution=redistribution,
        export=export,
    )
    return save_ml_artifact(
        path,
        fit.model,
        feature_schema={
            "support_id": fit.support.support_id,
            "conditions": list(fit.model.condition_names),
        },
        target_schema={
            "coordinates": "canonical-mass-centered",
            "length_unit": fit.support.template.scale.length_unit.to_dict(),
            "meaning": "structural-proposals-not-Boltzmann-samples",
        },
        provenance={
            "fit_id": fit.fit_id,
            "dataset_id": fit.dataset_id,
            "steps": fit.training_steps,
            "initial_training_loss": fit.initial_training_loss,
            "final_training_loss": fit.final_training_loss,
            "validation_loss": fit.validation_loss,
            "rights": [item.to_record() for item in fit.rights],
        },
        licenses=tuple(sorted({item.license_id for item in fit.rights})),
    )


def load_coordinate_model(
    path, support, *, weight_rights, commercial_use=False, export=False
):
    """Load admitted, checksum-bound weights; never fetch a checkpoint or provider."""
    require_coordinate_rights(
        (weight_rights,), commercial_use=commercial_use, export=export
    )
    payload = Path(path).read_bytes()
    digest = hashlib.new(weight_rights.checksum_algorithm, payload).hexdigest()
    if digest != weight_rights.checksum or len(payload) != weight_rights.size_bytes:
        raise ValueError("Weight bytes do not match their rights manifest.")
    artifact = read_ml_artifact(path)
    model = artifact.model
    if (
        not isinstance(model, ConditionalCoordinateVelocity)
        or model.support_id != support.support_id
    ):
        raise ValueError(
            "Checkpoint is not the requested fixed-chemistry coordinate model."
        )
    metadata = artifact.manifest.provenance
    parents = tuple(
        ReferenceArtifactManifest.from_record(item) for item in metadata["rights"]
    )
    rights = (*parents, weight_rights)
    require_coordinate_rights(rights, commercial_use=commercial_use, export=export)
    return CoordinateFitResult(
        model,
        support,
        metadata["dataset_id"],
        rights,
        metadata["initial_training_loss"],
        metadata["final_training_loss"],
        metadata["validation_loss"],
        metadata["steps"],
        metadata["fit_id"],
    )
