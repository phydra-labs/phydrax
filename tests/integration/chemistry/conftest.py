"""Optional molecular CUDA-provider admission, separate from numerical qualification."""

from collections.abc import Iterator
from importlib.metadata import distribution, PackageNotFoundError

import jax
import pytest

from phydrax import config
from phydrax._cuda_native import load_cuda_component
from phydrax._execution_device import device_platform


@pytest.fixture
def cuest_runtime() -> Iterator[None]:
    """Skip absent optional dependencies/hardware; never hide an invalid install."""
    for package in ("phydrax-cuest", "phydrax-cuda13"):
        try:
            distribution(package)
        except PackageNotFoundError:
            pytest.skip(f"Optional {package} distribution is not installed")
    devices = tuple(
        device for device in jax.local_devices() if device_platform(device) == "cuda"
    )
    if not devices:
        pytest.skip("Real CUDA hardware is unavailable to the selected JAX backend")
    # Once distributions and a CUDA device exist, a missing/corrupt component,
    # dependency mismatch, explicit bad path, or other loader refusal is a failure.
    load_cuda_component("cuest")
    with (
        config.context(config.NumericsConfig(enable_x64=True)),
        jax.default_device(devices[0]),
    ):
        yield
