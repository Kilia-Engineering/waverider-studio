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
    Apply variable-radius fillet to leading edge of a waverider solid.

    Uses OCC's BRepFilletAPI_MakeFillet with linearly varying radius:
    full radius at the wingtip, tapering to near-zero at the nose tip.
    This preserves the nose shape while blunting the outer LE.

    Identifies LE edges by geometry-based classification:
    - Exclude edges on the symmetry plane (both vertices z ≈ 0)
    - Exclude edges on the back face (both vertices x ≈ x_max)
    - Remaining edge(s) with one vertex near the tip are LE
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

        # Classify edge by vertex positions
        on_sym = abs(p1[2]) < tol and abs(p2[2]) < tol
        on_back = abs(p1[0] - x_max) < tol and abs(p2[0] - x_max) < tol
        has_z = abs(p1[2]) > tol or abs(p2[2]) > tol
        at_tip_1 = abs(p1[0] - x_min) < tol and abs(p1[2]) < tol
        at_tip_2 = abs(p2[0] - x_min) < tol and abs(p2[2]) < tol
        has_tip = at_tip_1 or at_tip_2

        label = "?"
        if on_sym:
            label = "symmetry"
        elif on_back:
            label = "back"
        elif has_z and has_tip:
            label = "LEADING EDGE"
            le_edges.append(edge)
        elif has_z:
            label = "trailing edge"

        print(f"  Edge {i}: ({p1[0]:.4f},{p1[1]:.4f},{p1[2]:.4f})->"
              f"({p2[0]:.4f},{p2[1]:.4f},{p2[2]:.4f})  [{label}]")

    print(f"[Blunting] {len(le_edges)} LE edge(s) identified")

    if not le_edges:
        print("[Blunting] No LE edges found — exporting sharp LE")
        return solid

    # Try variable-radius fillet first (preserves nose)
    result = _try_variable_fillet(solid, radius, le_edges, x_min, tol)
    if result is not None:
        return result

    # Fallback: constant-radius fillet with decreasing radius
    print("[Blunting] Variable fillet failed, trying constant radius...")
    for factor in [0.5, 0.25, 0.1]:
        r = radius * factor
        try:
            result = solid.fillet(r, le_edges)
            print(f"[Blunting] Constant fillet OK (r={r:.6f}m, factor={factor})")
            return result
        except Exception as e:
            print(f"[Blunting] Constant fillet failed (r={r:.6f}): {e}")

    print("[Blunting] All fillet attempts failed — exporting sharp LE")
    return solid


def _try_variable_fillet(solid, radius, le_edges, x_min, tol):
    """
    Apply variable-radius fillet via OCC BRepFilletAPI_MakeFillet.

    Uses a radius profile that is constant (full radius) along most of the
    LE, then quickly tapers to near-zero in the last ~15% near the nose tip.
    This preserves the duck-nose shape while blunting the rest of the LE.
    """
    try:
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
        from OCP.BRep import BRep_Tool
        from OCP.TColgp import TColgp_Array1OfPnt2d
        from OCP.gp import gp_Pnt2d
        has_profile_api = True
    except ImportError:
        has_profile_api = False

    if not has_profile_api:
        try:
            from OCC.Core.BRepFilletAPI import BRepFilletAPI_MakeFillet
            from OCC.Core.BRep import BRep_Tool
            from OCC.Core.TColgp import TColgp_Array1OfPnt2d
            from OCC.Core.gp import gp_Pnt2d
            has_profile_api = True
        except ImportError:
            print("[Blunting] OCC fillet API not available for variable radius")
            return None

    nose_fraction = 0.02  # 2% of requested radius at the very tip
    taper_zone = 0.15     # taper in the last 15% near the nose

    for factor in [1.0, 0.75, 0.5, 0.25]:
        r_full = radius * factor
        r_nose = max(radius * nose_fraction * factor, 1e-6)

        try:
            fillet_builder = BRepFilletAPI_MakeFillet(solid.wrapped)

            for edge in le_edges:
                # Get edge curve and parametric range
                curve_data = BRep_Tool.Curve_s(edge.wrapped)
                curve = curve_data[0]
                u_first = curve_data[1]
                u_last = curve_data[2]
                u_range = u_last - u_first

                # Determine which parametric end is the nose
                p_first = curve.Value(u_first)
                p_last = curve.Value(u_last)
                # Distance to nose = distance to (x_min, *, z≈0)
                d_first = abs(p_first.X() - x_min) + abs(p_first.Z())
                d_last = abs(p_last.X() - x_min) + abs(p_last.Z())
                first_is_nose = d_first < d_last

                # Build radius profile: constant full, quick taper at nose
                profile = TColgp_Array1OfPnt2d(1, 4)
                if first_is_nose:
                    # u_first = nose, u_last = wingtip
                    u_taper = u_first + taper_zone * u_range
                    profile.SetValue(1, gp_Pnt2d(u_first, r_nose))
                    profile.SetValue(2, gp_Pnt2d(u_taper, r_full))
                    profile.SetValue(3, gp_Pnt2d(u_first + 0.5 * u_range, r_full))
                    profile.SetValue(4, gp_Pnt2d(u_last, r_full))
                    print(f"[Blunting] Profile: nose@u={u_first:.3f}(r={r_nose:.6f}) "
                          f"→ full@u={u_taper:.3f}(r={r_full:.6f}) → wingtip@u={u_last:.3f}")
                else:
                    # u_first = wingtip, u_last = nose
                    u_taper = u_last - taper_zone * u_range
                    profile.SetValue(1, gp_Pnt2d(u_first, r_full))
                    profile.SetValue(2, gp_Pnt2d(u_first + 0.5 * u_range, r_full))
                    profile.SetValue(3, gp_Pnt2d(u_taper, r_full))
                    profile.SetValue(4, gp_Pnt2d(u_last, r_nose))
                    print(f"[Blunting] Profile: wingtip@u={u_first:.3f}(r={r_full:.6f}) "
                          f"→ full@u={u_taper:.3f} → nose@u={u_last:.3f}(r={r_nose:.6f})")

                fillet_builder.Add(profile, edge.wrapped)

            fillet_builder.Build()
            if fillet_builder.IsDone():
                result = cq.Solid(fillet_builder.Shape())
                print(f"[Blunting] Variable fillet OK (full={r_full:.6f}, nose={r_nose:.6f})")
                return result
            else:
                print(f"[Blunting] Variable fillet not done (factor={factor})")
        except Exception as e:
            print(f"[Blunting] Variable fillet failed (factor={factor}): {e}")

    # Fallback: try simple two-point variable fillet (R1, R2)
    print("[Blunting] Profile fillet failed, trying simple two-point variable...")
    for factor in [1.0, 0.75, 0.5, 0.25]:
        r_full = radius * factor
        r_nose = max(radius * nose_fraction * factor, 1e-6)

        try:
            fillet_builder = BRepFilletAPI_MakeFillet(solid.wrapped)
            for edge in le_edges:
                curve_data = BRep_Tool.Curve_s(edge.wrapped)
                curve = curve_data[0]
                p_first = curve.Value(curve_data[1])
                p_last = curve.Value(curve_data[2])
                d_first = abs(p_first.X() - x_min) + abs(p_first.Z())
                d_last = abs(p_last.X() - x_min) + abs(p_last.Z())

                if d_first < d_last:
                    fillet_builder.Add(r_nose, r_full, edge.wrapped)
                else:
                    fillet_builder.Add(r_full, r_nose, edge.wrapped)

            fillet_builder.Build()
            if fillet_builder.IsDone():
                result = cq.Solid(fillet_builder.Shape())
                print(f"[Blunting] Two-point variable fillet OK (full={r_full:.6f}, nose={r_nose:.6f})")
                return result
        except Exception as e:
            print(f"[Blunting] Two-point fillet failed (factor={factor}): {e}")

    return None
