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
    Apply fillet to leading edge and nose of a waverider solid.

    Step 1: Fillet the LE edge (constant radius)
    Step 2: Fillet the symmetry edges near the nose to create a
            rounded "duck nose" cap

    Identifies edges by geometry-based classification:
    - Symmetry edges: both vertices z ≈ 0
    - Back edges: both vertices x ≈ x_max
    - LE edge: z-extent + one vertex at tip (x_min, z≈0)
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
    sym_nose_edges = []
    for i, edge in enumerate(all_edges):
        vertices = edge.Vertices()
        if len(vertices) < 2:
            print(f"  Edge {i}: <2 vertices, skipping")
            continue

        v1 = vertices[0].Center()
        v2 = vertices[-1].Center()
        p1 = np.array([v1.x, v1.y, v1.z])
        p2 = np.array([v2.x, v2.y, v2.z])

        # Classify edge by vertex positions
        on_sym = abs(p1[2]) < tol and abs(p2[2]) < tol
        on_back = abs(p1[0] - x_max) < tol and abs(p2[0] - x_max) < tol
        has_z = abs(p1[2]) > tol or abs(p2[2]) > tol
        at_tip_1 = abs(p1[0] - x_min) < tol and abs(p1[2]) < tol
        at_tip_2 = abs(p2[0] - x_min) < tol and abs(p2[2]) < tol
        has_tip = at_tip_1 or at_tip_2

        label = "?"
        if on_sym and not on_back:
            label = "symmetry"
            # Symmetry edges touching the nose tip — candidates for nose rounding
            if has_tip:
                label = "symmetry (nose)"
                sym_nose_edges.append(edge)
        elif on_back:
            label = "back"
        elif has_z and has_tip:
            label = "LEADING EDGE"
            le_edges.append(edge)
        elif has_z:
            label = "trailing edge"

        print(f"  Edge {i}: ({p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f})->"
              f"({p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f})  [{label}]")

    print(f"[Blunting] {len(le_edges)} LE edge(s), {len(sym_nose_edges)} nose edge(s)")

    if not le_edges:
        print("[Blunting] No LE edges found — exporting sharp LE")
        return solid

    # Step 1: Fillet the LE edge with constant radius
    current = solid
    le_filleted = False
    for factor in [1.0, 0.75, 0.5, 0.25, 0.1]:
        r = radius * factor
        try:
            current = current.fillet(r, le_edges)
            print(f"[Blunting] LE fillet OK (r={r:.6f}m, factor={factor})")
            le_filleted = True
            break
        except Exception as e:
            print(f"[Blunting] LE fillet failed (r={r:.6f}): {e}")

    if not le_filleted:
        print("[Blunting] All LE fillet attempts failed — exporting sharp LE")
        return solid

    # Step 2: Round the nose by filleting symmetry edges near the tip
    if sym_nose_edges:
        current = _round_nose(current, radius, x_min, x_max, tol)

    return current


def _round_nose(solid, radius, x_min, x_max, tol):
    """
    Round the nose tip by applying a variable-radius fillet to the
    symmetry-plane edges near the nose.

    After the LE fillet, the solid has new edges. We re-identify the
    symmetry edges that touch the nose region, then apply a variable
    fillet: full radius at the nose vertex, tapering to zero toward the back.
    """
    try:
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
        from OCP.BRep import BRep_Tool
    except ImportError:
        try:
            from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
            from OCC.Core.BRep import BRep_Tool
        except ImportError:
            print("[Nose] OCC API not available for nose rounding")
            return solid

    # Re-identify symmetry edges in the (now filleted) solid
    all_edges = solid.Edges()
    nose_edges = []

    for edge in all_edges:
        vertices = edge.Vertices()
        if len(vertices) < 2:
            continue
        v1 = vertices[0].Center()
        v2 = vertices[-1].Center()
        p1 = np.array([v1.x, v1.y, v1.z])
        p2 = np.array([v2.x, v2.y, v2.z])

        # Symmetry edge: both z ≈ 0, not on back face
        on_sym = abs(p1[2]) < tol and abs(p2[2]) < tol
        on_back = abs(p1[0] - x_max) < tol and abs(p2[0] - x_max) < tol
        at_tip_1 = abs(p1[0] - x_min) < tol and abs(p1[2]) < tol
        at_tip_2 = abs(p2[0] - x_min) < tol and abs(p2[2]) < tol
        has_tip = at_tip_1 or at_tip_2

        if on_sym and has_tip and not on_back:
            nose_edges.append(edge)

    if not nose_edges:
        print("[Nose] No symmetry nose edges found after LE fillet")
        return solid

    print(f"[Nose] Found {len(nose_edges)} symmetry edge(s) at nose, applying variable fillet")

    # Apply variable-radius fillet: full at nose, taper to zero toward back
    nose_radius = radius  # same as LE radius for smooth transition
    for factor in [1.0, 0.75, 0.5, 0.25]:
        r_nose = nose_radius * factor
        r_back = max(nose_radius * 0.01 * factor, 1e-6)

        try:
            fillet_builder = BRepFilletAPI_MakeFillet(solid.wrapped)

            for edge in nose_edges:
                # Determine which end is the nose (x ≈ x_min)
                curve_data = BRep_Tool.Curve_s(edge.wrapped)
                curve = curve_data[0]
                u_first = curve_data[1]
                u_last = curve_data[2]

                p_first = curve.Value(u_first)
                p_last = curve.Value(u_last)

                d_first = abs(p_first.X() - x_min)
                d_last = abs(p_last.X() - x_min)

                if d_first < d_last:
                    # u_first = nose, u_last = toward back
                    fillet_builder.Add(r_nose, r_back, edge.wrapped)
                else:
                    # u_last = nose, u_first = toward back
                    fillet_builder.Add(r_back, r_nose, edge.wrapped)

            fillet_builder.Build()
            if fillet_builder.IsDone():
                result = cq.Solid(fillet_builder.Shape())
                print(f"[Nose] Nose rounding OK (r_nose={r_nose:.6f}, r_back={r_back:.6f})")
                return result
            else:
                print(f"[Nose] Fillet not done (factor={factor})")
        except Exception as e:
            print(f"[Nose] Variable fillet failed (factor={factor}): {e}")

    # Fallback: constant-radius fillet on nose edges with small radius
    print("[Nose] Variable nose fillet failed, trying constant radius...")
    for factor in [0.5, 0.25, 0.1]:
        r = radius * factor
        try:
            result = solid.fillet(r, nose_edges)
            print(f"[Nose] Constant nose fillet OK (r={r:.6f})")
            return result
        except Exception as e:
            print(f"[Nose] Constant fillet failed (r={r:.6f}): {e}")

    print("[Nose] All nose fillet attempts failed — keeping LE-only fillet")
    return solid
