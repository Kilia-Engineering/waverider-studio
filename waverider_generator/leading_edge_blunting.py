"""
Leading Edge Blunting Module
============================
Provides multiple approaches to blunt the sharp leading edge of waverider geometries.

Approach A (Primary): CAD-level fillet using CadQuery/OCC
Approach B: Point-level modification before surface creation
Approach C (Fallback): Boolean cut + lofted replacement

Convention:
    x -> streamwise direction
    y -> transverse direction (vertical)
    z -> spanwise direction
    Origin at the waverider tip
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def blunt_leading_edge_points(waverider, radius):
    """
    Approach B: Point-level leading edge blunting.

    Modifies the leading edge points and nearby surface stream points
    by replacing the sharp apex with a circular arc profile of the
    given radius. This must be called BEFORE surface/CAD creation.

    Parameters
    ----------
    waverider : waverider object
        The generated waverider object (from generator.py).
        Must have upper_surface_streams and lower_surface_streams populated.
    radius : float
        Blunting radius in meters.

    Returns
    -------
    modified_upper_streams : list of np.ndarray
        Modified upper surface streams with blunted leading edge.
    modified_lower_streams : list of np.ndarray
        Modified lower surface streams with blunted leading edge.
    blunted_le : np.ndarray
        The new (blunted) leading edge points.
    """
    if radius <= 0:
        return (list(waverider.upper_surface_streams),
                list(waverider.lower_surface_streams),
                np.vstack([s[0] for s in waverider.upper_surface_streams]))

    us_streams = waverider.upper_surface_streams
    ls_streams = waverider.lower_surface_streams

    modified_upper = []
    modified_lower = []
    blunted_le_points = []

    for i in range(len(us_streams)):
        us = us_streams[i].copy()
        ls = ls_streams[i].copy()

        le_upper = us[0]  # leading edge point from upper surface
        le_lower = ls[0]  # leading edge point from lower surface

        # Compute the local tangent directions at leading edge
        # Upper surface tangent (pointing downstream from LE)
        if us.shape[0] >= 2:
            t_upper = us[1] - us[0]
            t_upper_norm = np.linalg.norm(t_upper)
            if t_upper_norm > 1e-12:
                t_upper = t_upper / t_upper_norm
            else:
                t_upper = np.array([1.0, 0.0, 0.0])
        else:
            t_upper = np.array([1.0, 0.0, 0.0])

        # Lower surface tangent (pointing downstream from LE)
        if ls.shape[0] >= 2:
            t_lower = ls[1] - ls[0]
            t_lower_norm = np.linalg.norm(t_lower)
            if t_lower_norm > 1e-12:
                t_lower = t_lower / t_lower_norm
            else:
                t_lower = np.array([1.0, 0.0, 0.0])
        else:
            t_lower = np.array([1.0, 0.0, 0.0])

        # The bisector direction (where the arc center sits)
        bisector = t_upper + t_lower
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm > 1e-12:
            bisector = bisector / bisector_norm
        else:
            # Tangents are opposite → use the normal to the plane
            bisector = np.array([1.0, 0.0, 0.0])

        # Half-angle between upper and lower tangents
        cos_half = np.clip(np.dot(t_upper, t_lower), -1, 1)
        half_angle = np.arccos(cos_half) / 2.0

        if half_angle < 1e-6:
            # Nearly flat LE, no blunting needed
            modified_upper.append(us)
            modified_lower.append(ls)
            blunted_le_points.append(le_upper)
            continue

        # Distance from original LE to arc center along bisector
        d_center = radius / np.sin(half_angle)

        # Arc center position
        arc_center = le_upper + d_center * bisector

        # The tangent points where the arc meets each surface
        # Project from center onto each tangent line
        # Tangent point on upper: center - dot(center-le, t_upper)*t_upper projected
        tp_upper = arc_center - np.dot(arc_center - le_upper, t_upper) * t_upper
        # Correct: tangent point is at distance = radius from center, along the
        # perpendicular from center to the tangent line
        proj_upper = le_upper + np.dot(arc_center - le_upper, t_upper) * t_upper
        tp_upper = proj_upper

        proj_lower = le_lower + np.dot(arc_center - le_lower, t_lower) * t_lower
        tp_lower = proj_lower

        # Generate arc points between tp_upper and tp_lower
        n_arc = 8  # number of arc discretization points
        # Vectors from center to tangent points
        v_upper = tp_upper - arc_center
        v_lower = tp_lower - arc_center

        # Normalize
        v_upper_norm = np.linalg.norm(v_upper)
        v_lower_norm = np.linalg.norm(v_lower)
        if v_upper_norm > 1e-12:
            v_upper_hat = v_upper / v_upper_norm
        else:
            v_upper_hat = -t_upper
        if v_lower_norm > 1e-12:
            v_lower_hat = v_lower / v_lower_norm
        else:
            v_lower_hat = -t_lower

        # Angle between the two radii
        cos_arc = np.clip(np.dot(v_upper_hat, v_lower_hat), -1, 1)
        arc_angle = np.arccos(cos_arc)

        # Generate arc using Rodrigues rotation or slerp
        arc_points = []
        for k in range(n_arc + 1):
            frac = k / n_arc
            # Spherical linear interpolation
            if arc_angle > 1e-6:
                w1 = np.sin((1 - frac) * arc_angle) / np.sin(arc_angle)
                w2 = np.sin(frac * arc_angle) / np.sin(arc_angle)
            else:
                w1 = 1 - frac
                w2 = frac
            direction = w1 * v_upper_hat + w2 * v_lower_hat
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-12:
                direction = direction / dir_norm
            point = arc_center + radius * direction
            arc_points.append(point)

        arc_points = np.array(arc_points)

        # New blunted leading edge point is the midpoint of the arc
        blunted_le = arc_points[n_arc // 2]
        blunted_le_points.append(blunted_le)

        # Modify upper surface: replace first point with upper tangent point,
        # then prepend the upper half of the arc
        upper_arc = arc_points[:n_arc // 2 + 1][::-1]  # from mid to upper
        us[0] = tp_upper
        modified_upper.append(np.vstack([upper_arc, us]))

        # Modify lower surface: replace first point with lower tangent point,
        # then prepend the lower half of the arc
        lower_arc = arc_points[n_arc // 2:]  # from mid to lower
        ls[0] = tp_lower
        modified_lower.append(np.vstack([lower_arc, ls]))

    blunted_le = np.array(blunted_le_points)
    return modified_upper, modified_lower, blunted_le


def fillet_leading_edge(solid, radius, tolerance=0.01):
    """
    Approach A (Primary): Apply CAD-level fillet to the leading edge.

    Uses CadQuery/OpenCascade to fillet the sharp edges where
    upper and lower surfaces meet at the leading edge.

    Parameters
    ----------
    solid : cq.Solid or cq.Workplane
        The waverider solid geometry (from to_CAD).
    radius : float
        Fillet radius in meters.
    tolerance : float
        Tolerance for identifying leading edge edges.

    Returns
    -------
    filleted : cq.Solid or cq.Workplane
        The filleted waverider solid.
    """
    import cadquery as cq

    try:
        # If it's a Workplane, extract the solid
        if hasattr(solid, 'val'):
            the_solid = solid.val()
        elif hasattr(solid, 'objects') and len(solid.objects) > 0:
            the_solid = solid.objects[0]
        else:
            the_solid = solid

        # Find leading edge edges
        # The LE is at the front of the waverider (minimum x values)
        # Strategy: find edges where the minimum x coordinate is small
        # (near the nose) or edges that are shared between upper and lower faces

        all_edges = the_solid.Edges()
        if not all_edges:
            logger.warning("No edges found in solid, returning unchanged")
            return solid

        # Get bounding box to understand scale
        bb = the_solid.BoundingBox()
        x_min = bb.xmin
        x_max = bb.xmax
        length = x_max - x_min

        # Leading edge edges: those whose midpoint x < some threshold
        # and that are at the front boundary between upper and lower surfaces
        le_edges = []
        # The LE runs from tip (x=0) to the wingtip (x=length, z=width)
        # We identify LE edges by checking: midpoint is at the boundary
        # where y is near the maximum y for that x position (upper surface)
        # and the edge connects upper and lower surfaces

        for edge in all_edges:
            # Get edge midpoint
            mid = edge.Center()
            # Get edge endpoints
            vertices = edge.Vertices()
            if len(vertices) < 2:
                continue

            v1 = vertices[0].Center()
            v2 = vertices[1].Center()

            # LE edges typically have both endpoints with similar z coordinates
            # and span the front of the vehicle
            # For a waverider, the LE is where the upper surface (y ~= 0 near sym)
            # meets the lower surface

            # Heuristic: LE edges have endpoints at the front (low x relative to length)
            # and the edge connects two faces with different normal y-components
            avg_x = (v1.x + v2.x) / 2
            min_y = min(v1.y, v2.y)

            # Check if edge is near the leading edge region
            # The LE connects tip to wingtip along the front
            # Simple approach: find edges where both vertices are close
            # to the LE curve (roughly where upper and lower surfaces meet)

            # Get adjacent faces
            try:
                faces = the_solid.facesIntersectedByLine(
                    (mid.x, mid.y, mid.z),
                    (mid.x, mid.y + 0.001, mid.z)
                )
            except Exception:
                faces = []

            # Alternative: check if edge length is significant and
            # the edge runs roughly in the x-z plane
            edge_vec = np.array([v2.x - v1.x, v2.y - v1.y, v2.z - v1.z])
            edge_len = np.linalg.norm(edge_vec)

            if edge_len < tolerance * length:
                continue  # skip tiny edges

            # LE edges: primarily in x-z plane (small y component relative to length)
            # and the y-component change should be small
            y_range = abs(v2.y - v1.y)
            xz_range = np.sqrt((v2.x - v1.x)**2 + (v2.z - v1.z)**2)

            if xz_range > 0 and y_range / xz_range < 0.3:
                # This edge is mostly in the x-z plane
                # Check if it's at the leading edge (front boundary)
                # The LE has the property that x increases with z
                if v1.x < x_max * 0.99 or v2.x < x_max * 0.99:
                    # Not a trailing edge
                    le_edges.append(edge)

        if not le_edges:
            logger.warning("Could not identify leading edge edges for filleting. "
                           "Trying to fillet all front edges.")
            # Fallback: try all edges with small x midpoints
            for edge in all_edges:
                mid = edge.Center()
                if mid.x < x_min + 0.5 * length:
                    le_edges.append(edge)

        if not le_edges:
            raise RuntimeError("No leading edge edges found for filleting")

        # Apply fillet
        logger.info(f"Applying fillet with radius={radius} to {len(le_edges)} edges")
        filleted = cq.Workplane("XY").newObject([the_solid])
        filleted = filleted.newObject([the_solid.fillet(radius, le_edges)])

        return filleted

    except Exception as e:
        logger.error(f"Fillet approach failed: {e}")
        raise


def loft_blunted_leading_edge(waverider, solid, radius, n_sections=20):
    """
    Approach C (Fallback): Boolean cut + lofted replacement.

    Creates a blunted leading edge by:
    1. Cutting away a thin strip at the leading edge using a boolean operation
    2. Replacing it with a lofted surface having a circular arc cross-section

    Parameters
    ----------
    waverider : waverider object
        The generated waverider (for geometry reference).
    solid : cq.Solid or cq.Workplane
        The original sharp waverider solid.
    radius : float
        Blunting radius in meters.
    n_sections : int
        Number of cross-sections along the LE for the loft.

    Returns
    -------
    blunted_solid : cq.Solid or cq.Workplane
        The waverider with blunted leading edge.
    """
    import cadquery as cq

    try:
        if hasattr(solid, 'val'):
            the_solid = solid.val()
        elif hasattr(solid, 'objects') and len(solid.objects) > 0:
            the_solid = solid.objects[0]
        else:
            the_solid = solid

        # Extract leading edge from waverider
        us_streams = waverider.upper_surface_streams
        le_points = np.vstack([s[0] for s in us_streams])

        # Get the tangent directions at the leading edge
        # Upper surface tangent at each LE point
        upper_tangents = []
        lower_tangents = []
        for i in range(len(us_streams)):
            us = us_streams[i]
            ls = waverider.lower_surface_streams[i]
            if us.shape[0] >= 2:
                t_u = us[1] - us[0]
                t_u = t_u / (np.linalg.norm(t_u) + 1e-12)
            else:
                t_u = np.array([1, 0, 0], dtype=float)
            if ls.shape[0] >= 2:
                t_l = ls[1] - ls[0]
                t_l = t_l / (np.linalg.norm(t_l) + 1e-12)
            else:
                t_l = np.array([1, 0, 0], dtype=float)
            upper_tangents.append(t_u)
            lower_tangents.append(t_l)

        # Create a cutting box that removes the nose region
        # The cut depth is related to the blunting radius
        bb = the_solid.BoundingBox()

        # Cut depth: how far back from LE we remove material
        # For a circular arc of radius r, the depth is approximately r
        cut_depth = radius * 2.0

        # Build the lofted replacement nose piece
        # For each LE section, create a circular arc cross-section
        loft_wires = []
        arc_centers = []
        for i in range(len(le_points)):
            le_pt = le_points[i]
            t_u = upper_tangents[i]
            t_l = lower_tangents[i]

            # Bisector direction
            bisector = t_u + t_l
            b_norm = np.linalg.norm(bisector)
            if b_norm > 1e-12:
                bisector = bisector / b_norm
            else:
                bisector = np.array([1, 0, 0], dtype=float)

            # Half-angle
            cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
            half_angle = np.arccos(cos_half) / 2.0

            if half_angle < 1e-6:
                continue

            # Arc center
            d_center = radius / np.sin(half_angle)
            center = le_pt + d_center * bisector
            arc_centers.append(center)

            # Tangent points
            tp_upper = le_pt + np.dot(center - le_pt, t_u) * t_u
            tp_lower = le_pt + np.dot(center - le_pt, t_l) * t_l

            # Arc mid-point (the new blunted LE point)
            v_up = tp_upper - center
            v_lo = tp_lower - center
            v_up_hat = v_up / (np.linalg.norm(v_up) + 1e-12)
            v_lo_hat = v_lo / (np.linalg.norm(v_lo) + 1e-12)
            v_mid = v_up_hat + v_lo_hat
            v_mid = v_mid / (np.linalg.norm(v_mid) + 1e-12)
            arc_mid = center + radius * v_mid

            # Create a 3-point arc wire at this cross-section
            try:
                wire = (cq.Workplane("XY")
                        .moveTo(tp_upper[1], tp_upper[2])
                        .threePointArc(
                            (arc_mid[1], arc_mid[2]),
                            (tp_lower[1], tp_lower[2])
                        )
                        .val())

                # Transform wire to correct 3D position
                # The wire is in the Y-Z plane, need to move to x position
                wire = wire.moved(cq.Location(cq.Vector(tp_upper[0], 0, 0)))
                loft_wires.append(wire)
            except Exception as e:
                logger.warning(f"Failed to create arc wire at section {i}: {e}")
                continue

        if len(loft_wires) < 2:
            raise RuntimeError("Not enough valid cross-sections for lofted blunting")

        # Create cutting solid: a box that covers the LE region
        x_min = min(le_points[:, 0]) - cut_depth
        x_max = max(le_points[:, 0]) + cut_depth
        y_min = bb.ymin - 0.01
        y_max = bb.ymax + 0.01
        z_min = min(le_points[:, 2]) - 0.01
        z_max = max(le_points[:, 2]) + 0.01

        # Build cut box - a thin strip along the leading edge
        # We need a more sophisticated cutting surface that follows the LE
        # For now, use a swept cut along the leading edge curve

        # Boolean cut the nose off
        cut_box = (cq.Workplane("XY")
                   .transformed(offset=cq.Vector(
                       (x_min + x_max) / 2,
                       (y_min + y_max) / 2,
                       (z_min + z_max) / 2))
                   .box(x_max - x_min, y_max - y_min, z_max - z_min))

        cut_result = (cq.Workplane("XY")
                      .newObject([the_solid])
                      .cut(cut_box))

        # Loft the arc sections to create the blunted nose
        nose_solid = cq.Workplane("XY").newObject(loft_wires).loft(ruled=True)

        # Union the cut body with the lofted nose
        blunted = cut_result.union(nose_solid)

        return blunted

    except Exception as e:
        logger.error(f"Loft approach failed: {e}")
        raise


def apply_blunting(waverider, solid=None, radius=0.0, method='auto'):
    """
    Main entry point for leading edge blunting.

    Tries the specified method, with automatic fallback:
    - 'auto': Tries fillet first (A), then loft (C) if fillet fails
    - 'fillet': Only Approach A
    - 'loft': Only Approach C
    - 'points': Only Approach B (returns modified streams, not a solid)

    Parameters
    ----------
    waverider : waverider object
        The generated waverider object.
    solid : cq.Solid or cq.Workplane, optional
        The CAD solid (required for 'fillet', 'loft', 'auto').
    radius : float
        Blunting radius in meters.
    method : str
        Blunting method: 'auto', 'fillet', 'loft', or 'points'.

    Returns
    -------
    result : depends on method
        For 'fillet'/'loft'/'auto': the blunted cq.Solid/Workplane
        For 'points': tuple of (modified_upper, modified_lower, blunted_le)
    method_used : str
        Which method was actually used.
    """
    if radius <= 0:
        if method == 'points':
            return blunt_leading_edge_points(waverider, 0), 'points'
        return solid, 'none'

    if method == 'points':
        result = blunt_leading_edge_points(waverider, radius)
        return result, 'points'

    if method == 'fillet':
        if solid is None:
            raise ValueError("solid is required for fillet method")
        result = fillet_leading_edge(solid, radius)
        return result, 'fillet'

    if method == 'loft':
        if solid is None:
            raise ValueError("solid is required for loft method")
        result = loft_blunted_leading_edge(waverider, solid, radius)
        return result, 'loft'

    if method == 'auto':
        if solid is None:
            raise ValueError("solid is required for auto method")

        # Try fillet first (Approach A)
        try:
            result = fillet_leading_edge(solid, radius)
            logger.info("Blunting succeeded with fillet approach (A)")
            return result, 'fillet'
        except Exception as e:
            logger.warning(f"Fillet approach failed: {e}, trying loft approach")

        # Fallback to loft (Approach C)
        try:
            result = loft_blunted_leading_edge(waverider, solid, radius)
            logger.info("Blunting succeeded with loft approach (C)")
            return result, 'loft'
        except Exception as e:
            logger.warning(f"Loft approach also failed: {e}, trying point-level approach")

        # Last resort: point-level (Approach B) — returns stream data, not a solid
        try:
            result = blunt_leading_edge_points(waverider, radius)
            logger.info("Blunting succeeded with point-level approach (B)")
            return result, 'points'
        except Exception as e:
            logger.error(f"All blunting approaches failed: {e}")
            raise RuntimeError(
                f"All blunting approaches failed.\n"
                f"Fillet: {e}\nLoft: {e}\nPoints: {e}"
            )

    raise ValueError(f"Unknown method: {method}. Use 'auto', 'fillet', 'loft', or 'points'.")


def compute_blunted_le_preview(waverider, radius, n_points=50):
    """
    Compute a blunted leading edge curve for 3D preview visualization.

    This is a lightweight function that only computes the blunted LE
    geometry for display purposes without modifying the CAD model.

    Parameters
    ----------
    waverider : waverider object
        The generated waverider.
    radius : float
        Blunting radius in meters.
    n_points : int
        Number of points per arc section.

    Returns
    -------
    blunted_curve : np.ndarray
        (N, 3) array of blunted leading edge points.
    original_curve : np.ndarray
        (N, 3) array of original leading edge points.
    """
    us_streams = waverider.upper_surface_streams
    ls_streams = waverider.lower_surface_streams

    original_le = np.vstack([s[0] for s in us_streams])

    if radius <= 0:
        return original_le, original_le

    blunted_points = []

    for i in range(len(us_streams)):
        us = us_streams[i]
        ls = ls_streams[i]

        le_pt = us[0]

        # Get tangent directions
        if us.shape[0] >= 2:
            t_u = us[1] - us[0]
            n = np.linalg.norm(t_u)
            t_u = t_u / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)
        else:
            t_u = np.array([1, 0, 0], dtype=float)

        if ls.shape[0] >= 2:
            t_l = ls[1] - ls[0]
            n = np.linalg.norm(t_l)
            t_l = t_l / n if n > 1e-12 else np.array([1, 0, 0], dtype=float)
        else:
            t_l = np.array([1, 0, 0], dtype=float)

        # Bisector
        bisector = t_u + t_l
        b_norm = np.linalg.norm(bisector)
        if b_norm > 1e-12:
            bisector = bisector / b_norm
        else:
            bisector = np.array([1, 0, 0], dtype=float)

        # Half-angle
        cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
        half_angle = np.arccos(cos_half) / 2.0

        if half_angle < 1e-6:
            blunted_points.append(le_pt)
            continue

        # Arc center and midpoint
        d_center = radius / np.sin(half_angle)
        center = le_pt + d_center * bisector

        # Tangent points
        tp_upper = le_pt + np.dot(center - le_pt, t_u) * t_u
        tp_lower = le_pt + np.dot(center - le_pt, t_l) * t_l

        # Arc midpoint (the blunted LE point)
        v_up = tp_upper - center
        v_lo = tp_lower - center
        v_up_hat = v_up / (np.linalg.norm(v_up) + 1e-12)
        v_lo_hat = v_lo / (np.linalg.norm(v_lo) + 1e-12)
        v_mid = v_up_hat + v_lo_hat
        v_mid_norm = np.linalg.norm(v_mid)
        if v_mid_norm > 1e-12:
            v_mid = v_mid / v_mid_norm
        arc_mid = center + radius * v_mid

        blunted_points.append(arc_mid)

    blunted_curve = np.array(blunted_points)
    return blunted_curve, original_le
