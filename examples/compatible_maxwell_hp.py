#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import phydrax as phx


def run():
    complex_ = phx.discretization.fem.TensorDeRhamComplex(4, 3)
    return {
        "gradient_shape": complex_.gradient.shape,
        "curl_shape": complex_.curl.shape,
        "divergence_shape": complex_.divergence.shape,
        "curl_gradient_defect": float(complex_.grad_curl_defect),
        "divergence_curl_defect": float(complex_.curl_div_defect),
    }


if __name__ == "__main__":
    print(run())
