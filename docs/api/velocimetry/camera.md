# Cameras and reconstruction

Camera projection, refraction, calibration, rigs, and uncertainty-aware triangulation.
`CameraModel.refractive_stack` consumes
`phydrax.optics.geometric.PlanarRefractiveStack`; refraction is owned by the optics
domain rather than duplicated inside the camera package.


::: phydrax.velocimetry.camera
    options:
      show_root_heading: true
      members_order: source
