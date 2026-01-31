#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._domain import _AbstractDomain, _AbstractUnaryDomain


class ProductDomain(_AbstractDomain):
    r"""A labeled Cartesian product of unary domains.

    Given unary domains $\Omega_1,\dots,\Omega_k$ with labels $\ell_1,\dots,\ell_k$,
    a `ProductDomain` represents the product

    $$
    \Omega = \Omega_{\ell_1}\times\cdots\times\Omega_{\ell_k}.
    $$

    Labels must be unique unless colliding factors are structurally equivalent, in
    which case they are de-duplicated. This allows joining domains that share the
    same label (e.g. two identical `TimeInterval`s) without ambiguity.
    """

    _factors: tuple[_AbstractUnaryDomain, ...]

    def __init__(self, *domains: _AbstractDomain):
        """Create a product domain from unary domains (or nested product domains)."""
        factors: list[_AbstractUnaryDomain] = []
        for dom in domains:
            if isinstance(dom, ProductDomain):
                factors.extend(dom.factors)
            elif isinstance(dom, _AbstractUnaryDomain):
                factors.append(dom)
            else:
                raise TypeError(f"Unsupported domain type {type(dom).__name__}.")

        labels = [f.label for f in factors]
        unique = set(labels)
        if len(unique) != len(labels):
            # Raise an error unless all colliding factors are structurally equivalent.
            seen: dict[str, _AbstractUnaryDomain] = {}
            for f in factors:
                if f.label not in seen:
                    seen[f.label] = f
                    continue
                if not seen[f.label].equivalent(f):
                    raise ValueError(
                        f"Label collision for {f.label!r} when joining domains; relabel explicitly."
                    )
            # De-duplicate equivalent colliding factors (keep first occurrence).
            dedup: dict[str, _AbstractUnaryDomain] = {}
            for f in factors:
                dedup.setdefault(f.label, f)
            factors = list(dedup.values())

        self._factors = tuple(factors)

    @property
    def factors(self) -> tuple[_AbstractUnaryDomain, ...]:
        """Return the unary factors of the product domain."""
        return self._factors

    @property
    def labels(self) -> tuple[str, ...]:
        r"""Return the label tuple $(\ell_1,\ldots,\ell_k)$ for this product domain."""
        return tuple(f.label for f in self._factors)

    def factor(self, label: str, /) -> _AbstractUnaryDomain:
        """Return the unary factor corresponding to `label`."""
        for f in self._factors:
            if f.label == label:
                return f
        raise KeyError(f"Label {label!r} not in domain {self.labels}.")

    def equivalent(self, other: object, /) -> bool:
        """Return whether `other` is structurally equivalent to this domain."""
        if not isinstance(other, ProductDomain):
            return False
        if self.labels != other.labels:
            return False
        for a, b in zip(self.factors, other.factors, strict=True):
            if not a.equivalent(b):
                return False
        return True

    def boundary(self):
        r"""Return the boundary as a `DomainComponentUnion`.

        This constructs a union over boundary terms for each factor:

        - For geometry factors, one term with that label set to `Boundary()`.
        - For scalar factors (e.g. time), two terms corresponding to `FixedStart()`
          and `FixedEnd()`.

        This mirrors the common decomposition of $\partial(\Omega_x\times[t_0,t_1])$
        into spatial boundary and initial/final time slices.
        """
        from ._base import _AbstractGeometry
        from ._components import Boundary, DomainComponentUnion, FixedEnd, FixedStart
        from ._domain import RelabeledDomain
        from ._scalar import _AbstractScalarDomain

        terms = []
        for lbl in self.labels:
            factor = self.factor(lbl)
            if isinstance(factor, RelabeledDomain):
                factor = factor.base

            if isinstance(factor, _AbstractGeometry):
                terms.append(self.component({lbl: Boundary()}))
                continue

            if isinstance(factor, _AbstractScalarDomain):
                terms.append(self.component({lbl: FixedStart()}))
                terms.append(self.component({lbl: FixedEnd()}))
                continue

            raise TypeError(
                f"boundary() is not defined for unary domain type {type(factor).__name__}."
            )

        return DomainComponentUnion(tuple(terms))
