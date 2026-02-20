from waverider_generator.generator import waverider
import cadquery as cq
from cadquery import exporters
import numpy as np
import logging
logger = logging.getLogger(__name__)


def _enforce_min_thickness(us_streams, ls_streams, min_thickness, include_le=False):
    """
    Enforce a minimum thickness between upper and lower surface streams.

    At each corresponding point pair, if the vertical (Y) distance between
    upper and lower surface is less than min_thickness, both surfaces are
    offset symmetrically about their midpoint to achieve the minimum.

    Parameters
    ----------
    us_streams : list of ndarray
        Upper surface streams, each shape (n_pts, 3).
    ls_streams : list of ndarray
        Lower surface streams, each shape (n_pts, 3).
    min_thickness : float
        Minimum allowed thickness in meters (same units as geometry).
    include_le : bool
        If True, also enforce thickness at j=0 (leading edge points).
        Use when the LE will be replaced by Bezier blunting curves.

    Returns
    -------
    us_out, ls_out : list of ndarray
        Deep-copied streams with thickness enforced.
    """
    us_out = [s.copy() for s in us_streams]
    ls_out = [s.copy() for s in ls_streams]

    n_streams = min(len(us_out), len(ls_out))
    j_start = 0 if include_le else 1
    for i in range(n_streams):
        n_pts = min(us_out[i].shape[0], ls_out[i].shape[0])
        for j in range(j_start, n_pts):
            y_upper = us_out[i][j, 1]
            y_lower = ls_out[i][j, 1]
            thickness = y_upper - y_lower  # upper is above lower (more positive Y)
            if thickness < min_thickness:
                mid_y = (y_upper + y_lower) / 2.0
                us_out[i][j, 1] = mid_y + min_thickness / 2.0
                ls_out[i][j, 1] = mid_y - min_thickness / 2.0

    print(f"[MinThickness] Enforced min_thickness={min_thickness:.6f}m "
          f"across {n_streams} stream pairs (include_le={include_le})")
    return us_out, ls_out


def enforce_min_thickness_arrays(upper, lower, min_thickness, include_le=False):
    """
    Enforce minimum thickness on 3D surface arrays (n_le, n_stream, 3).

    Used by the cone-derived waverider tab which stores surfaces as arrays
    rather than stream lists.

    Parameters
    ----------
    upper, lower : ndarray, shape (n_le, n_stream, 3)
        Upper and lower surface point arrays.
    min_thickness : float
        Minimum allowed thickness in meters.
    include_le : bool
        If True, also enforce thickness at j=0 (leading edge points).
        Use when the LE will be replaced by Bezier blunting curves.

    Returns
    -------
    upper_out, lower_out : ndarray
        Copies with thickness enforced.
    """
    upper_out = upper.copy()
    lower_out = lower.copy()
    n_le, n_stream = upper_out.shape[0], upper_out.shape[1]
    j_start = 0 if include_le else 1
    for i in range(n_le):
        for j in range(j_start, n_stream):
            y_up = upper_out[i, j, 1]
            y_lo = lower_out[i, j, 1]
            thickness = y_up - y_lo
            if thickness < min_thickness:
                mid_y = (y_up + y_lo) / 2.0
                upper_out[i, j, 1] = mid_y + min_thickness / 2.0
                lower_out[i, j, 1] = mid_y - min_thickness / 2.0
    print(f"[MinThickness] Enforced min_thickness={min_thickness:.6f}m "
          f"on {n_le}x{n_stream} surface arrays (include_le={include_le})")
    return upper_out, lower_out


def _sew_faces_to_solid(faces, tolerance=1e-3):
    """
    Sew faces into a solid using BRepBuilderAPI_Sewing.

    Unlike cq.Shell.makeShell which requires topologically connected faces
    (shared edges), sewing merges faces whose edges are geometrically close
    but have different OCC topology. This is essential when surfaces are built
    independently via interpPlate — each surface gets its own B-spline edges
    even when the boundary points are identical.

    Parameters
    ----------
    faces : list
        CadQuery Face objects or OCC TopoDS_Face objects.
    tolerance : float
        Sewing tolerance in model units. Edges within this distance
        are merged into shared topology.

    Returns
    -------
    cq.Solid
    """
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    from OCP.ShapeFix import ShapeFix_Solid

    sewer = BRepBuilderAPI_Sewing(tolerance)
    for face in faces:
        if hasattr(face, 'wrapped'):
            sewer.Add(face.wrapped)
        else:
            sewer.Add(face)
    sewer.Perform()

    sewn_shape = sewer.SewedShape()

    # Extract shell from the sewn shape
    explorer = TopExp_Explorer(sewn_shape, TopAbs_SHELL)
    if not explorer.More():
        raise RuntimeError(
            f"Sewing produced no shell from {len(faces)} faces "
            f"(tolerance={tolerance:.1e})")

    shell = TopoDS.Shell_s(explorer.Current())

    # Build solid from shell
    fixer = ShapeFix_Solid()
    solid_shape = fixer.SolidFromShell(shell)

    print(f"[Sewing] {len(faces)} faces → solid OK (tol={tolerance:.1e})")
    return cq.Solid(solid_shape)


def to_CAD(waverider:waverider,sides : str,export: bool,filename: str,**kwargs):

    if "scale" in kwargs:
        scale=kwargs["scale"]
        if not (isinstance(scale, (int,float)) and scale >0):
            raise ValueError("scale must be a float or int greater than 0")
    else:
        scale=1.0 # SI units (meters)

    # Leading edge blunting parameters
    blunting_radius = kwargs.get("blunting_radius", 0.0)
    blunting_method = kwargs.get("blunting_method", "auto")

    # Minimum thickness parameter (0 = disabled)
    min_thickness = kwargs.get("min_thickness", 0.0)

    # extract streams from waverider object
    us_streams=waverider.upper_surface_streams
    ls_streams=waverider.lower_surface_streams

    # Apply minimum thickness enforcement if requested
    if min_thickness > 0:
        us_streams, ls_streams = _enforce_min_thickness(
            us_streams, ls_streams, min_thickness)

    # Determine if we use the pre-blunted path
    use_pre_blunted = (blunting_radius > 0 and blunting_method == "pre_blunted")

    # Sweep-scaled radius option
    sweep_scaled = kwargs.get("sweep_scaled", False)

    if use_pre_blunted:
        # ===== PRE-BLUNTED PATH: 4-face solid with G2 Bezier LE embedded =====
        from waverider_generator.leading_edge_blunting import compute_pre_blunted_streams
        print(f"[PreBlunted G2] Computing G2 Bezier blunted geometry "
              f"(r={blunting_radius:.4f}m, sweep_scaled={sweep_scaled})")
        blunt_data = compute_pre_blunted_streams(
            us_streams, ls_streams, blunting_radius,
            sweep_scaled=sweep_scaled)

        # Modified streams have Bezier points embedded, starting at blunt_tip
        us_streams = blunt_data['modified_upper']
        ls_streams = blunt_data['modified_lower']
        # Shared LE boundary = blunt tip points
        le = blunt_data['blunted_le']

        # Fall through to the standard 4-face solid builder below
        # (same code as the original path, using modified streams + blunted LE)

    if not use_pre_blunted:
        # compute LE from original streams
        le = np.vstack([x[0] for x in us_streams])

    # ===== SHARED 4-FACE SOLID BUILDER =====
    # Both pre-blunted and original paths use us_streams, ls_streams, le

    # compute TE
    te_upper_surface = np.vstack([x[-1] for x in us_streams])
    te_lower_surface = np.vstack([x[-1] for x in ls_streams])

    # interior points for upper surface
    us_points = []
    for i in range(len(us_streams)):
        for j in range(1, us_streams[i].shape[0] - 1):
            us_points.append(tuple(us_streams[i][j]))

    # interior points for lower surface
    ls_points = []
    for i in range(len(ls_streams)):
        for j in range(1, ls_streams[i].shape[0] - 1):
            ls_points.append(tuple(ls_streams[i][j]))

    # create boundaries
    us_sym_start_y = float(us_streams[0][0, 1])
    us_sym_end_y = float(us_streams[0][-1, 1])
    ls_sym_start_y = float(ls_streams[0][0, 1])
    ls_sym_end_y = float(ls_streams[0][-1, 1])

    # Nose X position (may differ from 0 for pre-blunted)
    nose_x_upper = float(us_streams[0][0, 0])
    nose_x_lower = float(ls_streams[0][0, 0])

    points_upper_surface = [(nose_x_upper, us_sym_start_y, 0),
                            (waverider.length, us_sym_end_y, 0)]
    points_lower_surface = [(nose_x_lower, ls_sym_start_y, 0),
                            (waverider.length, ls_sym_end_y, 0)]
    workplane = cq.Workplane("XY")
    edge_wire_te_upper_surface = workplane.moveTo(
        points_upper_surface[0][0], points_upper_surface[0][1])
    edge_wire_te_lower_surface = workplane.moveTo(
        points_lower_surface[0][0], points_lower_surface[0][1])

    for point in points_upper_surface[1:]:
        edge_wire_te_upper_surface = edge_wire_te_upper_surface.lineTo(point[0], point[1])
    for point in points_lower_surface[1:]:
        edge_wire_te_lower_surface = edge_wire_te_lower_surface.lineTo(point[0], point[1])

    # add the LE and TE splines
    edge_wire_te_upper_surface = edge_wire_te_upper_surface.add(
        cq.Workplane("XY").spline([tuple(x) for x in le]))
    edge_wire_te_lower_surface = edge_wire_te_lower_surface.add(
        cq.Workplane("XY").spline([tuple(x) for x in le]))
    edge_wire_te_upper_surface = edge_wire_te_upper_surface.add(
        cq.Workplane("XY").spline([tuple(x) for x in te_upper_surface]))
    edge_wire_te_lower_surface = edge_wire_te_lower_surface.add(
        cq.Workplane("XY").spline([tuple(x) for x in te_lower_surface]))

    # create surfaces
    upper_surface = cq.Workplane("XY").interpPlate(
        edge_wire_te_upper_surface, us_points, 0)
    lower_surface = cq.Workplane("XY").interpPlate(
        edge_wire_te_lower_surface, ls_points, 0)

    # back face
    e1 = cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in te_lower_surface])
    e2 = cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in te_upper_surface])
    sym_edge = np.vstack(((waverider.length, us_sym_end_y, 0),
                          (waverider.length, ls_sym_end_y, 0)))
    v1 = cq.Vector(*sym_edge[0])
    v2 = cq.Vector(*sym_edge[1])
    e3 = cq.Edge.makeLine(v1, v2)
    back = cq.Face.makeFromWires(cq.Wire.assembleEdges([e1, e2, e3]))

    # symmetry face
    v_nose_upper = cq.Vector(nose_x_upper, us_sym_start_y, 0)
    v_nose_lower = cq.Vector(nose_x_lower, ls_sym_start_y, 0)
    v_te_upper = cq.Vector(waverider.length, us_sym_end_y, 0)
    v_te_lower = cq.Vector(waverider.length, ls_sym_end_y, 0)
    e4 = cq.Edge.makeLine(v_nose_upper, v_te_upper)
    e5 = cq.Edge.makeLine(v_nose_lower, v_te_lower)
    sym_edges = [e3, e4, e5]
    if abs(us_sym_start_y - ls_sym_start_y) > 1e-8:
        e6 = cq.Edge.makeLine(v_nose_lower, v_nose_upper)
        sym_edges.append(e6)
    sym = cq.Face.makeFromWires(cq.Wire.assembleEdges(sym_edges))

    # create solid (4-face, sewn)
    left_side = _sew_faces_to_solid(
        [upper_surface.objects[0], lower_surface.objects[0], back, sym]).scale(scale)

    # Apply post-solid fillet only for non-pre-blunted path with legacy methods
    if not use_pre_blunted and blunting_radius > 0:
        le_scaled = le * scale
        left_side = _apply_le_fillet(left_side, blunting_radius * scale, le_scaled)

    right_side = left_side.mirror(mirrorPlane='XY')

    if sides=="left":
        if export==True:
            cq.exporters.export(left_side, filename)
        return left_side

    elif sides=="right":
        if export==True:
            cq.exporters.export(right_side, filename)
        return right_side

    elif sides=="both":

        waverider_solid = (
        cq.Workplane("XY")
        .newObject([right_side])
        .union(left_side)
        )
        if export==True:
            cq.exporters.export(waverider_solid, filename)
        return waverider_solid

    else:
        return ValueError("sides is either 'left', 'right' or 'both'")


def _build_le_face(arc_sections, tp_upper_curve, tp_lower_curve):
    """
    Build the leading edge face as a lofted surface through circular arc
    cross-sections, ensuring C1 continuity with upper/lower surfaces.

    Each arc cross-section is a 3-point arc wire from tp_upper to tp_lower
    passing through the arc midpoint. Lofting through these wires with
    BRepOffsetAPI_ThruSections produces a smooth surface whose tangent at
    the tp_upper/tp_lower boundaries matches the circular arc tangent,
    which by construction equals the upper/lower surface tangent direction.

    Parameters
    ----------
    arc_sections : list of ndarray (n_arc+1, 3)
        Circular arc points at each span station, from tp_upper to tp_lower.
    tp_upper_curve : ndarray (n_stations, 3)
        Upper tangent points at each span station.
    tp_lower_curve : ndarray (n_stations, 3)
        Lower tangent points at each span station.

    Returns
    -------
    le_face : OCC Face or None
        The lofted LE face, or None if construction fails.
    """
    from OCP.BRepOffsetAPI import BRepOffsetAPI_ThruSections
    from OCP.TopoDS import TopoDS

    n_stations = len(arc_sections)

    # Collect valid arc wires (skip degenerate stations at nose)
    wires = []
    n_skipped_span = 0
    n_skipped_collinear = 0
    n_arc_wires = 0
    n_line_wires = 0
    for i in range(n_stations):
        arc = arc_sections[i]
        tp_u = tp_upper_curve[i]
        tp_l = tp_lower_curve[i]

        # Skip degenerate stations where tangent points are identical
        span = np.linalg.norm(tp_u - tp_l)
        if span < 1e-10:
            n_skipped_span += 1
            continue

        # Arc midpoint (middle of the arc array)
        mid_idx = len(arc) // 2
        arc_mid = arc[mid_idx]

        # Check that the 3 points are not collinear
        v1 = arc_mid - tp_u
        v2 = tp_l - tp_u
        cross = np.linalg.norm(np.cross(v1, v2))
        if cross < 1e-10:
            # Collinear — use a line instead of arc
            try:
                edge = cq.Edge.makeLine(
                    cq.Vector(*tp_u), cq.Vector(*tp_l))
                wire = cq.Wire.assembleEdges([edge])
                wires.append(wire)
                n_line_wires += 1
            except Exception:
                n_skipped_collinear += 1
                continue
        else:
            try:
                edge = cq.Edge.makeThreePointArc(
                    cq.Vector(*tp_u),
                    cq.Vector(*arc_mid),
                    cq.Vector(*tp_l))
                wire = cq.Wire.assembleEdges([edge])
                wires.append(wire)
                n_arc_wires += 1
            except Exception as e:
                logger.warning(f"Arc wire failed at station {i}: {e}")
                # Fall back to line
                try:
                    edge = cq.Edge.makeLine(
                        cq.Vector(*tp_u), cq.Vector(*tp_l))
                    wire = cq.Wire.assembleEdges([edge])
                    wires.append(wire)
                    n_line_wires += 1
                except Exception:
                    continue

    print(f"[PreBlunted] LE face wire stats: {n_stations} stations, "
          f"{n_skipped_span} skipped(span), {n_arc_wires} arcs, "
          f"{n_line_wires} lines, {len(wires)} total wires")

    if len(wires) < 2:
        logger.error(f"_build_le_face: only {len(wires)} valid wires, need >=2")
        return None

    try:
        # Build lofted surface through arc wires
        # isSolid=False → we want a shell/face, not a solid
        builder = BRepOffsetAPI_ThruSections(False, True)  # isSolid=False, isRuled=True initially
        for wire in wires:
            builder.AddWire(wire.wrapped)
        builder.Build()
        shape = builder.Shape()

        # Extract the face(s) from the shape
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        faces = []
        while explorer.More():
            face = TopoDS.Face_s(explorer.Current())
            faces.append(cq.Face(face))
            explorer.Next()

        if not faces:
            logger.error("_build_le_face: ThruSections produced no faces")
            return None

        print(f"[PreBlunted] LE face built: {len(wires)} arc wires → "
              f"{len(faces)} face(s)")
        # Return the first (and typically only) face
        return faces[0]

    except Exception as e:
        logger.error(f"_build_le_face ThruSections failed: {e}")

        # Fallback: try interpPlate with arc interior points
        try:
            print("[PreBlunted] Falling back to interpPlate for LE face")
            # Boundary: tp_upper spline + tp_lower spline + nose/wingtip closure
            boundary = cq.Workplane("XY").spline(
                [tuple(x) for x in tp_upper_curve if np.linalg.norm(x - tp_upper_curve[0]) > 1e-8 or True])
            boundary = boundary.add(cq.Workplane("XY").spline(
                [tuple(x) for x in tp_lower_curve]))
            # Wingtip closure
            wt_edge = cq.Edge.makeLine(
                cq.Vector(*tp_upper_curve[-1]),
                cq.Vector(*tp_lower_curve[-1]))
            boundary = boundary.add(cq.Workplane("XY").newObject([wt_edge]))
            # Nose closure
            nose_dist = np.linalg.norm(tp_upper_curve[0] - tp_lower_curve[0])
            if nose_dist > 1e-8:
                nose_edge = cq.Edge.makeLine(
                    cq.Vector(*tp_lower_curve[0]),
                    cq.Vector(*tp_upper_curve[0]))
                boundary = boundary.add(cq.Workplane("XY").newObject([nose_edge]))

            # Interior points: arc midpoints + intermediate arc points
            interior = []
            for i in range(n_stations):
                arc = arc_sections[i]
                if np.linalg.norm(tp_upper_curve[i] - tp_lower_curve[i]) < 1e-8:
                    continue
                # Add several arc points as interior guidance
                for k in range(1, len(arc) - 1):
                    interior.append(tuple(arc[k]))

            le_face_wp = cq.Workplane("XY").interpPlate(boundary, interior, 0)
            print(f"[PreBlunted] LE face built via interpPlate fallback "
                  f"({len(interior)} interior points)")
            return le_face_wp.val()

        except Exception as e2:
            logger.error(f"_build_le_face interpPlate fallback failed: {e2}")
            return None


def _apply_le_fillet(solid, radius, le_points, nose_cap=False):
    """
    Apply leading edge blunting to a waverider solid.

    For OC waverider: just fillet the LE edges (nose_cap=False).
    For cone-derived: optionally apply nose cap after LE fillet (nose_cap=True).
    """
    all_edges = solid.Edges()
    bb = solid.BoundingBox()
    x_min = bb.xmin
    x_max = bb.xmax
    tol = max((x_max - x_min) * 0.01, 1e-4)

    print(f"[Blunting] Solid bounding box: x=[{bb.xmin:.4f}, {bb.xmax:.4f}], "
          f"y=[{bb.ymin:.4f}, {bb.ymax:.4f}], z=[{bb.zmin:.4f}, {bb.zmax:.4f}]")
    print(f"[Blunting] Solid has {len(all_edges)} edges, radius={radius:.5f}m")

    le_edges = []
    for i, edge in enumerate(all_edges):
        vertices = edge.Vertices()
        if len(vertices) < 2:
            print(f"  Edge {i}: <2 vertices, skipping")
            continue

        v1 = vertices[0].Center()
        v2 = vertices[-1].Center()
        p1 = np.array([v1.x, v1.y, v1.z])
        p2 = np.array([v2.x, v2.y, v2.z])

        on_sym = abs(p1[2]) < tol and abs(p2[2]) < tol
        on_back = abs(p1[0] - x_max) < tol and abs(p2[0] - x_max) < tol
        has_z = abs(p1[2]) > tol or abs(p2[2]) > tol
        at_tip_1 = abs(p1[0] - x_min) < tol and abs(p1[2]) < tol
        at_tip_2 = abs(p2[0] - x_min) < tol and abs(p2[2]) < tol
        has_tip = at_tip_1 or at_tip_2

        label = "?"
        if on_sym and not on_back:
            label = "symmetry (nose)" if has_tip else "symmetry"
        elif on_back:
            label = "back"
        elif has_z and has_tip:
            label = "LEADING EDGE"
            le_edges.append(edge)
        elif has_z:
            label = "trailing edge"

        print(f"  Edge {i}: ({p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f})->"
              f"({p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f})  [{label}]")

    print(f"[Blunting] {len(le_edges)} LE edge(s)")

    if not le_edges:
        print("[Blunting] No LE edges found — exporting sharp LE")
        return solid

    # LE-only fillet
    current = None
    le_r_used = 0
    for factor in [1.0, 0.75, 0.5, 0.25, 0.1]:
        r = radius * factor
        try:
            current = solid.fillet(r, le_edges)
            le_r_used = r
            print(f"[Blunting] LE fillet OK (r={r:.6f}m, factor={factor})")
            break
        except Exception as e:
            print(f"[Blunting] LE fillet failed (r={r:.6f}): {e}")

    if current is None:
        print("[Blunting] All LE fillet attempts failed — exporting sharp LE")
        if nose_cap:
            # For cone-derived: try nose cap on the original solid
            capped = _cap_nose(solid, radius)
            if capped is not None:
                return capped
        return solid

    # Optionally apply nose cap (cone-derived only)
    if nose_cap:
        capped = _cap_nose(current, le_r_used)
        if capped is not None:
            return capped

    return current


def _cap_nose(solid, radius):
    """
    Replace the sharp nose tip with a smooth rounded cap.

    1. Cut the solid at x = x_min + cut_dist to remove the sharp tip
    2. Find the new edges created on the cut face
    3. Fillet those edges to create a smooth, rounded nose cap

    The fillet on clean planar-intersection edges produces G2-continuous
    blending with the original waverider surfaces.
    """
    bb = solid.BoundingBox()
    x_min = bb.xmin
    x_max = bb.xmax
    length = x_max - x_min

    # Try increasing cut distances — farther from tip = larger cross-section
    # = more room for the fillet to succeed
    for cut_mult in [6, 10, 15, 20]:
        cut_dist = min(radius * cut_mult, length * 0.08)
        cut_x = x_min + cut_dist

        # Create a box that covers everything with x < cut_x
        y_span = bb.ymax - bb.ymin
        z_span = bb.zmax - bb.zmin
        margin = max(y_span, z_span, length * 0.1) * 3
        eps = length * 0.0001

        try:
            cutter = cq.Solid.makeBox(
                cut_dist + eps,
                margin,
                margin,
                pnt=cq.Vector(x_min - eps, bb.ymin - margin / 3, bb.zmin - margin / 3)
            )
            trimmed = solid.cut(cutter)
        except Exception as e:
            print(f"[NoseCap] Boolean cut failed (cut_x={cut_x:.4f}): {e}")
            continue

        # Find edges on the cut face: both vertices at x ≈ cut_x
        cut_tol = cut_dist * 0.15
        cut_edges = []
        for edge in trimmed.Edges():
            verts = edge.Vertices()
            if len(verts) < 2:
                continue
            c1 = verts[0].Center()
            c2 = verts[-1].Center()
            if abs(c1.x - cut_x) < cut_tol and abs(c2.x - cut_x) < cut_tol:
                cut_edges.append(edge)

        if not cut_edges:
            print(f"[NoseCap] No edges found at cut plane x={cut_x:.4f}")
            continue

        print(f"[NoseCap] Cut at x={cut_x:.4f}: found {len(cut_edges)} edge(s), "
              f"trying fillet...")

        # Fillet the cut edges — use radius up to a fraction of cut_dist
        # so the fillet fits within the cross-section
        max_fillet_r = cut_dist * 0.45
        for frac in [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]:
            r = min(radius * frac, max_fillet_r)
            if r < 1e-6:
                continue
            try:
                result = trimmed.fillet(r, cut_edges)
                print(f"[NoseCap] Nose cap OK (r={r:.6f}, cut_x={cut_x:.4f})")
                return result
            except Exception as e:
                print(f"[NoseCap] Fillet failed (r={r:.6f}): {e}")

    print("[NoseCap] All nose cap attempts failed — keeping LE-only fillet")
    return None
