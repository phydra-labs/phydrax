#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from jax.nn import initializers


_initializer_dict = {
    "lecun_normal": initializers.lecun_normal(in_axis=1, out_axis=0),
    "lecun_uniform": initializers.lecun_uniform(in_axis=1, out_axis=0),
    "he_normal": initializers.he_normal(in_axis=1, out_axis=0),
    "he_uniform": initializers.he_uniform(in_axis=1, out_axis=0),
    "glorot_normal": initializers.glorot_normal(in_axis=1, out_axis=0),
    "glorot_uniform": initializers.glorot_uniform(in_axis=1, out_axis=0),
    "orthogonal": initializers.orthogonal(column_axis=1),
}

_initializer_dict.update(
    {
        "kaiming_normal": _initializer_dict["he_normal"],
        "kaiming_uniform": _initializer_dict["he_uniform"],
        "xavier_normal": _initializer_dict["glorot_normal"],
        "xavier_uniform": _initializer_dict["glorot_uniform"],
    }
)
