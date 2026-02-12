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

    # Apply LE blunting BEFORE solid creation by moving LE points to the tangent
    # points of a blunting circle. This preserves stream structure (no extra points)
    # so interpPlate/makeShell work correctly.
    blunting_method_used = 'none'
    if blunting_radius > 0 and blunting_method in ('auto', 'points'):
        us_streams = [s.copy() for s in us_streams]
        ls_streams = [s.copy() for s in ls_streams]
        n_blunted = 0
        for i in range(len(us_streams)):
            le_pt = us_streams[i][0].copy()
            # Upper surface tangent (downstream from LE)
            if us_streams[i].shape[0] >= 2:
                t_u = us_streams[i][1] - us_streams[i][0]
                t_u_norm = np.linalg.norm(t_u)
                t_u = t_u / t_u_norm if t_u_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            else:
                t_u = np.array([1.0, 0.0, 0.0])
            # Lower surface tangent
            if ls_streams[i].shape[0] >= 2:
                t_l = ls_streams[i][1] - ls_streams[i][0]
                t_l_norm = np.linalg.norm(t_l)
                t_l = t_l / t_l_norm if t_l_norm > 1e-12 else np.array([1.0, 0.0, 0.0])
            else:
                t_l = np.array([1.0, 0.0, 0.0])

            bisector = t_u + t_l
            b_norm = np.linalg.norm(bisector)
            if b_norm < 1e-12:
                continue
            bisector = bisector / b_norm

            cos_half = np.clip(np.dot(t_u, t_l), -1, 1)
            half_angle = np.arccos(cos_half) / 2.0
            if half_angle < 1e-6:
                continue

            d_center = blunting_radius / np.sin(half_angle)
            center = le_pt + d_center * bisector

            # Tangent points where the blunting arc meets each surface
            tp_upper = le_pt + np.dot(center - le_pt, t_u) * t_u
            tp_lower = le_pt + np.dot(center - le_pt, t_l) * t_l

            us_streams[i][0] = tp_upper
            ls_streams[i][0] = tp_lower
            n_blunted += 1

        blunting_method_used = 'points'
        print(f"[Blunting] LE blunting applied to {n_blunted}/{len(us_streams)} streams (r={blunting_radius:.4f} m)")

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

    # Post-solid blunting for explicitly requested 'fillet' or 'loft' methods only.
    # (For 'auto'/'points', blunting was already applied above before solid creation.)
    if blunting_radius > 0 and blunting_method_used == 'none':
        from waverider_generator.leading_edge_blunting import apply_blunting
        le_scaled = le * scale
        try:
            result, blunting_method_used = apply_blunting(
                waverider=waverider,
                solid=left_side,
                radius=blunting_radius * scale,
                method=blunting_method,
                le_points=le_scaled
            )
            if blunting_method_used == 'points':
                print("[Blunting] Point-level method not applicable after solid creation, skipping")
                blunting_method_used = 'skipped'
            else:
                if hasattr(result, 'val'):
                    left_side = result.val()
                elif hasattr(result, 'objects') and len(result.objects) > 0:
                    left_side = result.objects[0]
                else:
                    left_side = result
                print(f"[Blunting] LE blunting applied using method: {blunting_method_used}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Blunting] LE blunting failed ({e}), exporting with sharp LE")
            blunting_method_used = 'failed'

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




