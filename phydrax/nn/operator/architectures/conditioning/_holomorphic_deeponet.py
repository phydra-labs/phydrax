#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._holomorphic import HolomorphicJet
from phydrax._holomorphic_linear import HolomorphicLinearFrame
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.equations.trefftz._holomorphic_constraints import (
    HolomorphicAffineCoefficientMap,
    PreparedHolomorphicConstraintOperator,
)
from phydrax.nn._keys import EvalKey
from phydrax.nn.operator.data import FunctionSamples, OperatorBatch
from phydrax.nn.operator.engine import AbstractOperatorModel

from ._deeponet import (
    AbstractBasisTrunk,
    AbstractBranchEncoder,
    DeepONet,
)


HolomorphicTrunkMode = Literal["unconstrained", "fixed-target", "variable-target"]


class ConditionalHolomorphicMapCertificate(StrictModule, NonTrainableState):
    """Construction evidence for query-holomorphic conditional maps."""

    query_complex_input_size: int = eqx.field(static=True)
    complex_output_size: int = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    trunk_mode: HolomorphicTrunkMode = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    constraint_operator_id: str | None = eqx.field(static=True)
    coefficient_layout: str = eqx.field(static=True)
    bias_mode: str = eqx.field(static=True)
    branch_names: tuple[str, ...] = eqx.field(static=True)
    branch_fusion: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        query_complex_input_size: int,
        complex_output_size: int,
        latent_size: int,
        maximum_derivative_order: int,
        trunk_mode: HolomorphicTrunkMode,
        frame_id: str,
        constraint_operator_id: str | None,
        coefficient_layout: str,
        bias_mode: str,
        branch_names: tuple[str, ...],
        branch_fusion: str,
    ):
        input_size = int(query_complex_input_size)
        output_size = int(complex_output_size)
        latent = int(latent_size)
        derivative_order = int(maximum_derivative_order)
        frame_id_ = str(frame_id)
        operator_id = (
            None if constraint_operator_id is None else str(constraint_operator_id)
        )
        layout = str(coefficient_layout)
        bias = str(bias_mode)
        branches = tuple(str(value) for value in branch_names)
        fusion = str(branch_fusion)
        if min(input_size, output_size, latent) <= 0 or derivative_order < 0:
            raise ValueError(
                "Conditional holomorphic certificate dimensions are invalid."
            )
        if trunk_mode not in ("unconstrained", "fixed-target", "variable-target"):
            raise ValueError("Unknown conditional holomorphic trunk mode.")
        if not frame_id_ or not layout or not bias or not branches or not fusion:
            raise ValueError("Conditional holomorphic identifiers must be nonempty.")
        if trunk_mode == "unconstrained" and operator_id is not None:
            raise ValueError("Unconstrained holomorphic trunks cannot bind constraints.")
        if trunk_mode != "unconstrained" and not operator_id:
            raise ValueError(
                "Constrained holomorphic trunks require an operator identity."
            )
        self.query_complex_input_size = input_size
        self.complex_output_size = output_size
        self.latent_size = latent
        self.maximum_derivative_order = derivative_order
        self.trunk_mode = trunk_mode
        self.frame_id = frame_id_
        self.constraint_operator_id = operator_id
        self.coefficient_layout = layout
        self.bias_mode = bias
        self.branch_names = branches
        self.branch_fusion = fusion
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "conditional-holomorphic-map-certificate",
                "query_complex_input_size": input_size,
                "complex_output_size": output_size,
                "latent_size": latent,
                "maximum_derivative_order": derivative_order,
                "trunk_mode": trunk_mode,
                "frame_id": frame_id_,
                "constraint_operator_id": operator_id,
                "coefficient_layout": layout,
                "bias_mode": bias,
                "branch_names": list(branches),
                "branch_fusion": fusion,
            }
        )


class TargetAugmentedBranchEncoder(AbstractBranchEncoder):
    """Deterministic source targets concatenated with learned nullspace coordinates."""

    free_encoder: AbstractBranchEncoder
    target_indices: tuple[int, ...] = eqx.field(static=True)
    target_count: int = eqx.field(static=True)
    free_size: int = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)
    encoder_id: str = eqx.field(static=True)

    def __init__(
        self,
        free_encoder: AbstractBranchEncoder,
        target_indices: tuple[int, ...],
        /,
    ):
        if not isinstance(free_encoder, AbstractBranchEncoder):
            raise TypeError("free_encoder must be AbstractBranchEncoder.")
        indices = tuple(int(value) for value in target_indices)
        if (
            not indices
            or any(value < 0 for value in indices)
            or len(set(indices)) != len(indices)
        ):
            raise ValueError("target_indices must be unique nonnegative indices.")
        self.free_encoder = free_encoder
        self.target_indices = indices
        self.target_count = len(indices)
        self.free_size = int(free_encoder.latent_size)
        self.latent_size = self.target_count + self.free_size
        self.encoder_id = canonical_fingerprint(
            {
                "kind": "target-augmented-branch-encoder",
                "target_indices": list(indices),
                "free_size": self.free_size,
            }
        )

    def __call__(
        self,
        samples: FunctionSamples,
        /,
        *,
        case_ndim: int,
        key: EvalKey = None,
    ) -> Array:
        if samples.values is None:
            raise ValueError("Target-augmented encoding requires source values.")
        values = jnp.asarray(samples.values)
        if jnp.iscomplexobj(values):
            raise TypeError("Hard boundary targets must be real.")
        case_shape = tuple(int(size) for size in values.shape[:case_ndim])
        flat = values.reshape(case_shape + (-1,))
        if max(self.target_indices) >= int(flat.shape[-1]):
            raise ValueError("target_indices exceed the flattened source values.")
        targets = flat[..., jnp.asarray(self.target_indices)]
        free = self.free_encoder(samples, case_ndim=case_ndim, key=key)
        if free.shape != case_shape + (self.free_size,):
            raise ValueError("Free branch encoder returned an invalid shape.")
        return jnp.concatenate((targets, free), axis=-1)


class HolomorphicBasisTrunk(AbstractBasisTrunk, NonTrainableState):
    """Continuous certified holomorphic frame consumed by the shared DeepONet."""

    frame: Any
    constraint_operator: PreparedHolomorphicConstraintOperator | None
    coefficient_map: HolomorphicAffineCoefficientMap | None
    mode: HolomorphicTrunkMode = eqx.field(static=True)
    latent_size: int = eqx.field(static=True)
    out_size: int | Literal["scalar"] = eqx.field(static=True)
    coordinate_dimension: int = eqx.field(static=True)
    trunk_id: str = eqx.field(static=True)

    def __init__(
        self,
        frame: HolomorphicLinearFrame,
        /,
        *,
        constraint_operator: PreparedHolomorphicConstraintOperator | None = None,
        coefficient_map: HolomorphicAffineCoefficientMap | None = None,
    ):
        if not isinstance(frame, HolomorphicLinearFrame):
            raise TypeError("frame must implement HolomorphicLinearFrame.")
        if constraint_operator is not None and not isinstance(
            constraint_operator, PreparedHolomorphicConstraintOperator
        ):
            raise TypeError(
                "constraint_operator must be PreparedHolomorphicConstraintOperator or None."
            )
        if coefficient_map is not None and not isinstance(
            coefficient_map, HolomorphicAffineCoefficientMap
        ):
            raise TypeError(
                "coefficient_map must be HolomorphicAffineCoefficientMap or None."
            )
        if coefficient_map is not None:
            if constraint_operator is not None:
                raise ValueError(
                    "Pass either coefficient_map or constraint_operator, not both."
                )
            constraint_operator = coefficient_map.operator
        certificate = frame.linear_frame_certificate()
        if constraint_operator is not None:
            constrained_frame = constraint_operator.plan.frame.linear_frame_certificate()
            if constrained_frame.frame_id != certificate.frame_id:
                raise ValueError(
                    "Constraint operator and holomorphic trunk frame differ."
                )
        if coefficient_map is not None:
            mode: HolomorphicTrunkMode = "fixed-target"
            latent = coefficient_map.nullity
            if latent <= 0:
                raise ValueError("Fixed-target DeepONet trunk requires free coordinates.")
        elif constraint_operator is not None:
            mode = "variable-target"
            latent = (
                constraint_operator.target_count + constraint_operator.evidence.nullity
            )
        else:
            mode = "unconstrained"
            latent = certificate.real_coefficient_count
        self.frame = frame
        self.constraint_operator = constraint_operator
        self.coefficient_map = coefficient_map
        self.mode = mode
        self.latent_size = latent
        self.out_size = (
            "scalar"
            if certificate.complex_output_size == 1
            else certificate.complex_output_size
        )
        self.coordinate_dimension = 2 * certificate.complex_input_size
        self.trunk_id = canonical_fingerprint(
            {
                "kind": "holomorphic-deeponet-basis-trunk",
                "frame": certificate.frame_id,
                "mode": mode,
                "constraint_operator": (
                    None
                    if constraint_operator is None
                    else constraint_operator.prepared_id
                ),
                "coefficient_map": (
                    None if coefficient_map is None else coefficient_map.map_id
                ),
            }
        )

    @property
    def requires_fixed_query(self) -> bool:
        return False

    def _complex_coordinates(self, values: Array, /) -> Array:
        certificate = self.frame.linear_frame_certificate()
        dimension = certificate.complex_input_size
        if values.shape[-1:] != (2 * dimension,):
            raise ValueError(
                f"Holomorphic query coordinates must end with {2 * dimension} entries."
            )
        return values[..., :dimension] + 1j * values[..., dimension:]

    def _transform_basis(self, basis: Array, /) -> Array:
        if self.mode == "unconstrained":
            return basis
        operator = self.constraint_operator
        assert operator is not None
        nullspace = basis @ operator.nullspace_basis
        if self.mode == "fixed-target":
            return nullspace
        lift = basis @ operator.right_inverse
        return jnp.concatenate((lift, nullspace), axis=-1)

    def basis_derivative(
        self,
        coordinates: ArrayLike,
        multi_index: tuple[int, ...],
        /,
    ) -> Array:
        basis = self.frame.basis_derivative(coordinates, multi_index)
        return self._transform_basis(basis)

    def offset_derivative(
        self,
        coordinates: ArrayLike,
        multi_index: tuple[int, ...],
        /,
    ) -> Array:
        certificate = self.frame.linear_frame_certificate()
        if self.mode != "fixed-target":
            return jnp.zeros((certificate.complex_output_size,), dtype=jnp.complex128)
        coefficient_map = self.coefficient_map
        assert coefficient_map is not None
        basis = self.frame.basis_derivative(coordinates, multi_index)
        return basis @ coefficient_map.particular_coefficients

    def evaluate(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        key: EvalKey = None,
    ) -> Array:
        del key
        coordinates = query.coordinates_array(case_shape=case_shape)
        complex_coordinates = self._complex_coordinates(coordinates)
        dimension = self.frame.linear_frame_certificate().complex_input_size
        flat = complex_coordinates.reshape((-1, dimension))
        basis = jax.vmap(lambda value: self.basis_derivative(value, (0,) * dimension))(
            flat
        )
        output_size = self.frame.linear_frame_certificate().complex_output_size
        return basis.reshape(
            case_shape + query.sample_shape + (output_size, self.latent_size)
        )

    def evaluate_offset(
        self,
        query: FunctionSamples,
        /,
        *,
        case_shape: tuple[int, ...] = (),
        key: EvalKey = None,
    ) -> Array:
        del key
        coordinates = query.coordinates_array(case_shape=case_shape)
        complex_coordinates = self._complex_coordinates(coordinates)
        dimension = self.frame.linear_frame_certificate().complex_input_size
        flat = complex_coordinates.reshape((-1, dimension))
        offset = jax.vmap(lambda value: self.offset_derivative(value, (0,) * dimension))(
            flat
        )
        output_size = self.frame.linear_frame_certificate().complex_output_size
        return offset.reshape(case_shape + query.sample_shape + (output_size,))


class ConditionalHolomorphicDeepONet(AbstractOperatorModel):
    """DeepONet whose continuous query decoder is holomorphic by construction."""

    operator_architecture = "DeepONet"

    operator: DeepONet
    trunk: HolomorphicBasisTrunk
    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    _certificate: ConditionalHolomorphicMapCertificate

    def __init__(self, operator: DeepONet, /):
        if not isinstance(operator, DeepONet):
            raise TypeError("operator must be DeepONet.")
        if not isinstance(operator.trunk, HolomorphicBasisTrunk):
            raise TypeError(
                "Conditional holomorphic DeepONet requires HolomorphicBasisTrunk."
            )
        trunk = operator.trunk
        if operator.coord_dim != trunk.coordinate_dimension:
            raise ValueError(
                "DeepONet coordinate dimension and holomorphic trunk differ."
            )
        if trunk.mode != "unconstrained" and operator.bias is not None:
            raise ValueError(
                "Constrained holomorphic DeepONet cannot use a free decoder bias."
            )
        if (
            operator.latent_size != trunk.latent_size
            or operator.out_size != trunk.out_size
        ):
            raise ValueError("DeepONet and holomorphic trunk sizes differ.")
        frame = trunk.frame.linear_frame_certificate()
        operator_id = (
            None
            if trunk.constraint_operator is None
            else trunk.constraint_operator.prepared_id
        )
        self.operator = operator
        self.trunk = trunk
        self.in_size = operator.in_size
        self.out_size = operator.out_size
        self._certificate = ConditionalHolomorphicMapCertificate(
            query_complex_input_size=frame.complex_input_size,
            complex_output_size=frame.complex_output_size,
            latent_size=trunk.latent_size,
            maximum_derivative_order=frame.maximum_derivative_order,
            trunk_mode=trunk.mode,
            frame_id=frame.frame_id,
            constraint_operator_id=operator_id,
            coefficient_layout=(
                "full-real-frame"
                if trunk.mode == "unconstrained"
                else "nullspace"
                if trunk.mode == "fixed-target"
                else "target-plus-nullspace"
            ),
            bias_mode="constant" if operator.bias is not None else "none",
            branch_names=tuple(operator.branches),
            branch_fusion=operator.fusion,
        )

    @property
    def operator_contract(self):
        return self.operator.operator_contract

    def conditional_holomorphic_certificate(self) -> ConditionalHolomorphicMapCertificate:
        return self._certificate

    def __call__(self, value: Any, /, *, key: EvalKey = None) -> Array:
        return self.operator(value, key=key)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        return self.operator.__call_operator_batch__(batch, key=key)

    def query_jet(
        self,
        batch: OperatorBatch,
        coordinate: ArrayLike,
        order: int,
        /,
        *,
        key: EvalKey = None,
    ) -> HolomorphicJet:
        order_ = int(order)
        certificate = self._certificate
        if certificate.query_complex_input_size != 1:
            raise ValueError("Scalar conditional jets require one complex query input.")
        if order_ < 0 or order_ > certificate.maximum_derivative_order:
            raise ValueError("Requested conditional holomorphic jet is unavailable.")
        coefficients = self.operator.encode_sources(batch, key=key)
        scalar = jnp.asarray(coordinate).reshape(())

        def derivative(current: int) -> Array:
            basis = self.trunk.basis_derivative(scalar, (current,))
            result = jnp.sum(coefficients[..., None, :] * basis, axis=-1)
            result = result + self.trunk.offset_derivative(scalar, (current,))
            if current == 0 and self.operator.bias is not None:
                result = result + self.operator.bias
            return result

        value = derivative(0)
        derivatives = tuple(derivative(current) for current in range(1, order_ + 1))
        if self.out_size == "scalar":
            value = value[..., 0]
            derivatives = tuple(item[..., 0] for item in derivatives)
        return HolomorphicJet(value, derivatives)


class ConditionalHarmonicOperator2D(AbstractOperatorModel):
    """Real harmonic field operator from one query-holomorphic complex potential."""

    operator_architecture = "DeepONet"

    potential: ConditionalHolomorphicDeepONet
    in_size: int | Literal["scalar"]
    out_size: Literal["scalar"]

    def __init__(self, potential: ConditionalHolomorphicDeepONet, /):
        if not isinstance(potential, ConditionalHolomorphicDeepONet):
            raise TypeError("potential must be ConditionalHolomorphicDeepONet.")
        if potential.conditional_holomorphic_certificate().complex_output_size != 1:
            raise ValueError("Conditional harmonic operator requires one complex output.")
        self.potential = potential
        self.in_size = potential.in_size
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return self.potential.operator_contract

    def __call__(self, value: Any, /, *, key: EvalKey = None) -> Array:
        return jnp.real(self.potential(value, key=key))

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        return jnp.real(self.potential.__call_operator_batch__(batch, key=key))


__all__ = [
    "ConditionalHarmonicOperator2D",
    "ConditionalHolomorphicDeepONet",
    "ConditionalHolomorphicMapCertificate",
    "HolomorphicBasisTrunk",
    "HolomorphicTrunkMode",
    "TargetAugmentedBranchEncoder",
]
