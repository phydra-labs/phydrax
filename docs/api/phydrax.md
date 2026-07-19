# Phydrax

Top-level package namespace. Most functionality lives in subpackages:

- `phydrax.domain`: domains, geometry, sampling, and domain functions
- `phydrax.data_utils`: CSV loading and array scaling helpers
- `phydrax.operators`: differential/integral operators on `DomainFunction`s
- `phydrax.constraints`: residual and data penalty terms
- `phydrax.objectives`: raw signed scalar objectives and integral functionals
- `phydrax.nn`: neural network components and structured models
- `phydrax.solver`: loss assembly and training utilities
- `phydrax.export`: deployment helpers for learned inference functions
