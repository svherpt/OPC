def get_source_points(config):
    """
    Returns a list of (fx, fy, weight) source points in frequency space.
    fx, fy are spatial frequency offsets [1/nm].
    """

    illumination_type = config.get("illumination_type", "conventional")

    wavelength_nm = config.get("wavelength_nm", 193.0)
    numerical_aperture = config.get("numerical_aperture", 1.35)

    # Maximum spatial frequency supported by the projection system
    k = numerical_aperture / wavelength_nm

    if illumination_type == "conventional":
        # Single on-axis source
        return [
            (0.0, 0.0, 1.0)
        ]

    elif illumination_type == "dipole_x":
        # Two off-axis sources along x
        r = 0.6 * k
        return [
            ( r, 0.0, 0.5),
            (-r, 0.0, 0.5)
        ]

    elif illumination_type == "dipole_y":
        # Two off-axis sources along y
        r = 0.6 * k
        return [
            (0.0,  r, 0.5),
            (0.0, -r, 0.5)
        ]

    elif illumination_type == "quadrupole":
        # Four symmetric off-axis sources
        r = 0.6 * k
        return [
            ( r,  0.0, 0.25),
            (-r,  0.0, 0.25),
            ( 0.0,  r, 0.25),
            ( 0.0, -r, 0.25)
        ]

    elif illumination_type == "annular":
        # Simple discrete annular illumination (4-point ring)
        r = 0.7 * k
        return [
            ( r,  0.0, 0.25),
            (-r,  0.0, 0.25),
            ( 0.0,  r, 0.25),
            ( 0.0, -r, 0.25)
        ]

    else:
        raise ValueError(f"Unknown illumination_type: {illumination_type}")
