#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Mapping

from ._domain import Domain, JointFactor


class ProductDomain(Domain):
    """Ordered independent product of atomic joint factors.

    A factor may own several intrinsically coupled coordinates. Products flatten
    nested products but never split a factor merely because several public labels
    are bound to it.
    """

    _factors: tuple[JointFactor, ...]

    def __init__(self, *domains: Domain):
        if not domains:
            raise ValueError("ProductDomain requires at least one domain factor.")

        incoming: list[JointFactor] = []
        for domain in domains:
            if not isinstance(domain, Domain):
                raise TypeError(f"Unsupported domain type {type(domain).__name__}.")
            incoming.extend(domain.joint_factors)

        factors: list[JointFactor] = []
        for candidate in incoming:
            candidate_labels = set(candidate.labels)
            duplicate = False
            for existing in factors:
                overlap = candidate_labels.intersection(existing.labels)
                if not overlap:
                    continue
                if candidate.labels == existing.labels and candidate.same_support(existing):
                    duplicate = True
                    break
                raise ValueError(
                    "Label collision between joint domain factors "
                    f"{existing.labels} and {candidate.labels}. Relabel a complete "
                    "factor explicitly before joining."
                )
            if not duplicate:
                factors.append(candidate)
        self._factors = tuple(factors)

    @property
    def factors(self) -> tuple[JointFactor, ...]:
        """Atomic joint factors in canonical product order."""
        return self._factors

    @property
    def joint_factors(self) -> tuple[JointFactor, ...]:
        return self._factors

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for factor in self._factors for label in factor.labels)

    def same_support(self, other: object, /) -> bool:
        if not isinstance(other, Domain):
            return False
        other_factors = other.joint_factors
        if self.labels != other.labels or len(self.factors) != len(other_factors):
            return False
        return all(
            left.same_support(right)
            for left, right in zip(self.factors, other_factors, strict=True)
        )

    def relabel(
        self,
        labels: str | Mapping[str, str],
        /,
    ) -> Domain:
        if isinstance(labels, str):
            if len(self.labels) != 1:
                raise ValueError(
                    "Relabeling a product with multiple coordinates requires a mapping."
                )
            mapping = {self.labels[0]: labels}
        else:
            mapping = dict(labels)
        unknown = tuple(label for label in mapping if label not in self.labels)
        if unknown:
            raise KeyError(
                f"Relabel mapping contains unknown labels {unknown}; "
                f"expected a subset of {self.labels}."
            )
        replacement = tuple(mapping.get(label, label) for label in self.labels)
        if len(set(replacement)) != len(replacement):
            raise ValueError(f"Relabeling produced duplicate labels {replacement}.")
        factors = tuple(factor.relabel(mapping) for factor in self.factors)
        if all(new is old for new, old in zip(factors, self.factors, strict=True)):
            return self
        if len(factors) == 1:
            return factors[0]
        return ProductDomain(*factors)

    def boundary(self):
        """Return the additive product-boundary decomposition."""
        from ._base import AbstractGeometry
        from ._components import Boundary, ComponentSum, FixedEnd, FixedStart
        from ._scalar import AbstractScalarDomain

        terms = []
        for factor in self.factors:
            if len(factor.labels) != 1:
                raise TypeError(
                    "boundary() requires a boundary decomposition provider for coupled "
                    f"factor {factor.labels}."
                )
            label = factor.labels[0]
            if isinstance(factor, AbstractGeometry):
                terms.append(self.component({label: Boundary()}))
                continue
            if isinstance(factor, AbstractScalarDomain):
                terms.append(self.component({label: FixedStart()}))
                terms.append(self.component({label: FixedEnd()}))
                continue
            raise TypeError(
                "boundary() is not defined for factor type "
                f"{type(factor).__name__}."
            )
        return ComponentSum(tuple(terms))
