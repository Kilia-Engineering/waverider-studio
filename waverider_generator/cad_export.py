from waverider_generator.generator import waverider
import cadquery as cq
from cadquery import exporters
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

    # extract streams from waverider object
    us_streams=waverider.upper_surface_streams
    ls_streams=waverider.lower_surface_streams

    # compute LE
    le = np.vstack([x[0] for x in us_streams])

    # compute TE upper surface
    te_upper_surface=np.vstack([x[-1] for x in us_streams])

    # compute TE lower surface
    te_lower_surface=np.vstack([x[-1] for x in ls_streams])

    # add interior points for upper surface
    us_points=[]
    for i in range(len(us_streams)):
        for j in range(1, us_streams[i].shape[0]-1):
            us_points.append(tuple(us_streams[i][j]))

    # add interior points for lower surface
    ls_points=[]
    for i in range(len(ls_streams)):
        for j in range(1, ls_streams[i].shape[0]-1):
            ls_points.append(tuple(ls_streams[i][j]))

    # create boundaries
    # define points to create boundary with symmetry
    points_upper_surface = [(0, 0, 0), (waverider.length, 0, 0)]
    points_lower_surface=[(0,0,0),(waverider.length,ls_streams[0][-1,1],0)]
    # create a workplane and draw lines between points
    workplane = cq.Workplane("XY")
    edge_wire_te_upper_surface = workplane.moveTo(points_upper_surface[0][0], points_upper_surface[0][1])
    edge_wire_te_lower_surface=workplane.moveTo(points_lower_surface[0][0], points_lower_surface[0][1])

    for point in points_upper_surface[1:]:
        edge_wire_te_upper_surface = edge_wire_te_upper_surface.lineTo(point[0], point[1])
    for point in points_lower_surface[1:]:
        edge_wire_te_lower_surface = edge_wire_te_lower_surface.lineTo(point[0], point[1])

    # add the le and te
    edge_wire_te_upper_surface = edge_wire_te_upper_surface.add(cq.Workplane("XY").spline([tuple(x) for x in le]))
    edge_wire_te_lower_surface = edge_wire_te_lower_surface.add(cq.Workplane("XY").spline([tuple(x) for x in le]))
    edge_wire_te_upper_surface = edge_wire_te_upper_surface.add(cq.Workplane("XY").spline([tuple(x) for x in te_upper_surface]))
    edge_wire_te_lower_surface = edge_wire_te_lower_surface.add(cq.Workplane("XY").spline([tuple(x) for x in te_lower_surface]))

    # create upper surface
    upper_surface= cq.Workplane("XY").interpPlate(edge_wire_te_upper_surface, us_points, 0)

    # create lower surface
    lower_surface= cq.Workplane("XY").interpPlate(edge_wire_te_lower_surface, ls_points, 0)

    # add back as a plane
    e1 =cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in te_lower_surface])
    e2=cq.Edge.makeSpline([cq.Vector(tuple(x)) for x in te_upper_surface])
    sym_edge=np.vstack(((waverider.length, 0, 0),(waverider.length,ls_streams[0][-1,1],0)))
    v1 = cq.Vector(*sym_edge[0])
    v2 = cq.Vector(*sym_edge[1])
    e3 = cq.Edge.makeLine(v1, v2)
    back = cq.Face.makeFromWires(cq.Wire.assembleEdges([e1, e2,e3]))

    # add symmetry plane as face
    length_edge=np.vstack(((0, 0, 0),(waverider.length,0,0)))
    v1 = cq.Vector(*length_edge[0])
    v2 = cq.Vector(*length_edge[1])
    e4 = cq.Edge.makeLine(v1, v2)
    shockwave_edge=np.vstack(((0, 0, 0),(waverider.length,ls_streams[0][-1,1],0)))
    v1 = cq.Vector(*shockwave_edge[0])
    v2 = cq.Vector(*shockwave_edge[1])
    e5 = cq.Edge.makeLine(v1, v2)
    sym=cq.Face.makeFromWires(cq.Wire.assembleEdges([e3, e4,e5]))

    # create solid
    # by convention, +ve z is left so this produces the left side
    left_side= cq.Solid.makeSolid(cq.Shell.makeShell([upper_surface.objects[0], lower_surface.objects[0], back,sym])).scale(scale)

    # Apply leading edge blunting via post-solid fillet
    if blunting_radius > 0:
        le_scaled = le * scale
        left_side = _apply_le_fillet(left_side, blunting_radius * scale, le_scaled)

    right_side= left_side.mirror(mirrorPlane='XY')

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


def _apply_le_fillet(solid, radius, le_points):
    """
    Apply leading edge blunting + nose cap to a waverider solid.

    Strategy:
    1. Fillet the LE edge only (constant radius, no nose edges)
    2. Cut off the sharp nose tip at a station near the tip
    3. Fillet the edges created by the cut to form a smooth nose cap

    This avoids the broken OCP variable-fillet API and the topology
    issues that prevent two-step filleting of nose edges.
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

    # Step 1: LE-only fillet
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
        return solid

    # Step 2: Nose cap — cut the nose tip off and fillet the cut edges
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
