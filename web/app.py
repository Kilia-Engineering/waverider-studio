"""
Flask backend for the Waverider Web Designer.
Wraps waverider_generator to provide a REST API for generating
waverider geometries and exporting CAD files.
"""

import sys
import os
import json
import tempfile
import traceback
import numpy as np

# Add parent directory to path so we can import waverider_generator
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify, send_file, render_template
from waverider_generator.generator import waverider
from waverider_generator.cad_export import to_CAD

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# ─── Beta lookup table (from the GUI) ───────────────────────────────────────

BETA_TABLE = {
    2.0: (35.0, 40.0, 45.0),
    2.5: (30.0, 34.0, 38.0),
    3.0: (25.5, 26.5, 28.0),
    3.5: (22.0, 23.5, 25.0),
    4.0: (20.0, 21.0, 22.0),
    4.5: (18.5, 19.5, 20.5),
    5.0: (17.0, 18.0, 19.0),
    5.5: (16.0, 17.0, 18.0),
    6.0: (15.0, 16.0, 17.0),
    7.0: (13.5, 14.5, 15.5),
    8.0: (12.5, 13.5, 14.5),
    10.0: (11.0, 12.0, 13.0),
    12.0: (10.0, 11.0, 12.0),
    15.0: (9.0, 10.0, 11.0),
}


def get_recommended_beta(M):
    """Get recommended shock angle for a given Mach number via interpolation."""
    mach_values = sorted(BETA_TABLE.keys())

    if M <= mach_values[0]:
        low, mid, high = BETA_TABLE[mach_values[0]]
    elif M >= mach_values[-1]:
        low, mid, high = BETA_TABLE[mach_values[-1]]
    else:
        for i in range(len(mach_values) - 1):
            if mach_values[i] <= M <= mach_values[i + 1]:
                M1, M2 = mach_values[i], mach_values[i + 1]
                t = (M - M1) / (M2 - M1)
                low1, mid1, high1 = BETA_TABLE[M1]
                low2, mid2, high2 = BETA_TABLE[M2]
                low = low1 + t * (low2 - low1)
                mid = mid1 + t * (mid2 - mid1)
                high = high1 + t * (high2 - high1)
                break

    return {'low': round(low, 1), 'mid': round(mid, 1), 'high': round(high, 1)}


def compute_volume(wr):
    """Compute waverider internal volume using trapezoidal integration over cross-sections."""
    try:
        # numpy 2.x renamed trapz to trapezoid
        try:
            from numpy import trapezoid as _trapz
        except ImportError:
            from numpy import trapz as _trapz

        upper_streams = wr.upper_surface_streams
        lower_streams = wr.lower_surface_streams
        n_streams = min(len(upper_streams), len(lower_streams))

        # Integrate cross-sectional area along x (streamwise direction)
        n_x = 50
        x_vals = np.linspace(0, wr.length, n_x)
        areas = []

        for xi in x_vals:
            z_at_x = []
            dy_at_x = []
            for si in range(n_streams):
                us = upper_streams[si]
                ls = lower_streams[si]
                if us.shape[0] < 2 or ls.shape[0] < 2:
                    continue
                if xi < us[0, 0] or xi > us[-1, 0]:
                    continue
                y_u = float(np.interp(xi, us[:, 0], us[:, 1]))
                y_l = float(np.interp(xi, ls[:, 0], ls[:, 1]))
                z = us[0, 2]
                dy = y_u - y_l
                if dy > 0:
                    z_at_x.append(z)
                    dy_at_x.append(dy)
            area_at_x = 0.0
            if len(z_at_x) >= 2:
                z_arr = np.array(z_at_x)
                dy_arr = np.array(dy_at_x)
                idx = np.argsort(z_arr)
                area_at_x = float(_trapz(dy_arr[idx], z_arr[idx]))
            areas.append(area_at_x)

        volume = float(_trapz(areas, x_vals)) * 2  # *2 for both halves
        return round(volume, 6)
    except Exception:
        return 0.0


def build_mesh_data(wr):
    """Convert waverider geometry into triangulated mesh data for Three.js."""
    vertices = []
    faces = []
    vertex_map = {}

    def add_vertex(x, y, z):
        """Add vertex, return index. Deduplicates close vertices."""
        key = (round(x, 8), round(y, 8), round(z, 8))
        if key not in vertex_map:
            vertex_map[key] = len(vertices)
            vertices.append([float(x), float(y), float(z)])
        return vertex_map[key]

    def add_surface_triangles(surface_x, surface_y, surface_z, flip=False):
        """Triangulate a structured grid surface."""
        ny, nx = surface_x.shape
        for i in range(ny - 1):
            for j in range(nx - 1):
                v00 = add_vertex(surface_x[i, j], surface_y[i, j], surface_z[i, j])
                v10 = add_vertex(surface_x[i+1, j], surface_y[i+1, j], surface_z[i+1, j])
                v01 = add_vertex(surface_x[i, j+1], surface_y[i, j+1], surface_z[i, j+1])
                v11 = add_vertex(surface_x[i+1, j+1], surface_y[i+1, j+1], surface_z[i+1, j+1])

                if flip:
                    faces.append([v00, v10, v11])
                    faces.append([v00, v11, v01])
                else:
                    faces.append([v00, v11, v10])
                    faces.append([v00, v01, v11])

    def add_streams_surface(streams, flip=False):
        """Triangulate a surface defined by streamlines of variable length."""
        for s in range(len(streams) - 1):
            s1 = streams[s]
            s2 = streams[s + 1]

            if s1.shape[0] < 2 or s2.shape[0] < 2:
                continue

            # Resample both streams to same number of points
            n = min(s1.shape[0], s2.shape[0])
            t1 = np.linspace(0, 1, s1.shape[0])
            t2 = np.linspace(0, 1, s2.shape[0])
            t = np.linspace(0, 1, n)

            pts1 = np.column_stack([
                np.interp(t, t1, s1[:, 0]),
                np.interp(t, t1, s1[:, 1]),
                np.interp(t, t1, s1[:, 2])
            ])
            pts2 = np.column_stack([
                np.interp(t, t2, s2[:, 0]),
                np.interp(t, t2, s2[:, 1]),
                np.interp(t, t2, s2[:, 2])
            ])

            for j in range(n - 1):
                v00 = add_vertex(pts1[j, 0], pts1[j, 1], pts1[j, 2])
                v10 = add_vertex(pts2[j, 0], pts2[j, 1], pts2[j, 2])
                v01 = add_vertex(pts1[j+1, 0], pts1[j+1, 1], pts1[j+1, 2])
                v11 = add_vertex(pts2[j+1, 0], pts2[j+1, 1], pts2[j+1, 2])

                if v00 != v10 and v00 != v11 and v10 != v11:
                    if flip:
                        faces.append([v00, v10, v11])
                    else:
                        faces.append([v00, v11, v10])
                if v00 != v01 and v00 != v11 and v01 != v11:
                    if flip:
                        faces.append([v00, v11, v01])
                    else:
                        faces.append([v00, v01, v11])

    # Upper surface (structured grid) - one half
    add_surface_triangles(wr.upper_surface_x, wr.upper_surface_y, wr.upper_surface_z, flip=False)

    # Lower surface (from streamlines) - one half
    add_streams_surface(wr.lower_surface_streams, flip=True)

    # Mirror for the other half (z -> -z)
    n_verts_half = len(vertices)
    n_faces_half = len(faces)

    # Add mirrored vertices
    for i in range(n_verts_half):
        v = vertices[i]
        add_vertex(v[0], v[1], -v[2])

    # Add mirrored faces (flip winding order)
    for i in range(n_faces_half):
        f = faces[i]
        # Map mirrored vertex indices
        mv0 = add_vertex(vertices[f[0]][0], vertices[f[0]][1], -vertices[f[0]][2])
        mv1 = add_vertex(vertices[f[1]][0], vertices[f[1]][1], -vertices[f[1]][2])
        mv2 = add_vertex(vertices[f[2]][0], vertices[f[2]][1], -vertices[f[2]][2])
        faces.append([mv0, mv2, mv1])  # flip winding

    # Leading edge data
    le = wr.leading_edge.tolist()
    # Mirror leading edge
    le_mirrored = [[p[0], p[1], -p[2]] for p in reversed(le)]

    return {
        'vertices': vertices,
        'faces': faces,
        'leading_edge': le,
        'leading_edge_mirrored': le_mirrored,
    }


# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/beta-hint', methods=['GET'])
def beta_hint():
    """Get recommended beta for a Mach number."""
    try:
        M = float(request.args.get('mach', 6.0))
        beta_min = float(np.degrees(np.arcsin(1.0 / M)))
        recommended = get_recommended_beta(M)
        return jsonify({
            'beta_min': round(beta_min, 1),
            'recommended': recommended,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/validate', methods=['POST'])
def validate_params():
    """Validate design parameters and return constraint info."""
    try:
        data = request.get_json()
        X1 = float(data.get('X1', 0.11))
        X2 = float(data.get('X2', 0.63))
        width = float(data.get('width', 3.0))
        height = float(data.get('height', 1.0))

        # Compute max X2 for given X1, width, height
        max_X2 = (7.0 / 64.0) * (width / height)**4 * (1 - X1)**4 * 0.99
        max_X2 = min(max_X2, 1.0)

        valid = X2 < max_X2

        return jsonify({
            'valid': valid,
            'max_X2': round(max_X2, 4),
            'current_X2': X2,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate waverider geometry and return mesh data."""
    try:
        data = request.get_json()

        M_inf = float(data.get('mach', 6.0))
        beta = float(data.get('beta', 12.0))
        height = float(data.get('height', 1.0))
        width = float(data.get('width', 3.0))
        X1 = float(data.get('X1', 0.11))
        X2 = float(data.get('X2', 0.63))
        X3 = float(data.get('X3', 0.0))
        X4 = float(data.get('X4', 0.46))
        n_planes = int(data.get('n_planes', 40))
        n_streamwise = int(data.get('n_streamwise', 30))
        delta_streamwise = float(data.get('delta_streamwise', 0.1))
        n_upper_surface = int(data.get('n_upper_surface', 200))
        n_shockwave = int(data.get('n_shockwave', 200))
        match_shockwave = bool(data.get('match_shockwave', False))

        # Validate beta > mach angle
        mach_angle = np.degrees(np.arcsin(1.0 / M_inf))
        if beta <= mach_angle:
            return jsonify({'error': f'Shock angle ({beta:.1f}°) must be greater than Mach angle ({mach_angle:.1f}°)'}), 400

        wr = waverider(
            M_inf=M_inf,
            beta=beta,
            height=height,
            width=width,
            dp=[X1, X2, X3, X4],
            n_upper_surface=n_upper_surface,
            n_shockwave=n_shockwave,
            n_planes=n_planes,
            n_streamwise=n_streamwise,
            delta_streamwise=delta_streamwise,
            match_shockwave=match_shockwave,
        )

        mesh = build_mesh_data(wr)
        volume = compute_volume(wr)

        return jsonify({
            'success': True,
            'mesh': mesh,
            'info': {
                'length': round(wr.length, 4),
                'height': height,
                'width': width,
                'volume': volume,
                'mach': M_inf,
                'beta': beta,
                'deflection_angle': round(wr.theta, 4),
                'method': 'Shadow (Shockwave-Matched)' if match_shockwave else 'Osculating Cones',
            }
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Generation failed: {str(e)}'}), 500


@app.route('/api/export/step', methods=['POST'])
def export_step():
    """Generate and return a STEP file."""
    try:
        data = request.get_json()
        wr = _create_waverider(data)

        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
            tmppath = f.name

        to_CAD(wr, sides='both', export=True, filename=tmppath, scale=1000)

        return send_file(
            tmppath,
            mimetype='application/step',
            as_attachment=True,
            download_name='waverider.step'
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'STEP export failed: {str(e)}'}), 500


@app.route('/api/export/stl', methods=['POST'])
def export_stl():
    """Generate and return an STL file."""
    try:
        data = request.get_json()
        wr = _create_waverider(data)

        min_size = float(data.get('mesh_min_size', 5.0))
        max_size = float(data.get('mesh_max_size', 50.0))

        # First export STEP, then mesh with gmsh
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
            step_path = f.name
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
            stl_path = f.name

        to_CAD(wr, sides='both', export=True, filename=step_path, scale=1000)

        # Use gmsh to generate STL mesh
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.importShapes(step_path)
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay

        gmsh.model.mesh.generate(2)
        gmsh.write(stl_path)
        gmsh.finalize()

        # Cleanup temp step file
        os.unlink(step_path)

        return send_file(
            stl_path,
            mimetype='application/sla',
            as_attachment=True,
            download_name='waverider.stl'
        )
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'STL export failed: {str(e)}'}), 500


def _create_waverider(data):
    """Create a waverider object from request data."""
    return waverider(
        M_inf=float(data.get('mach', 6.0)),
        beta=float(data.get('beta', 12.0)),
        height=float(data.get('height', 1.0)),
        width=float(data.get('width', 3.0)),
        dp=[
            float(data.get('X1', 0.11)),
            float(data.get('X2', 0.63)),
            float(data.get('X3', 0.0)),
            float(data.get('X4', 0.46)),
        ],
        n_upper_surface=int(data.get('n_upper_surface', 200)),
        n_shockwave=int(data.get('n_shockwave', 200)),
        n_planes=int(data.get('n_planes', 40)),
        n_streamwise=int(data.get('n_streamwise', 30)),
        delta_streamwise=float(data.get('delta_streamwise', 0.1)),
        match_shockwave=bool(data.get('match_shockwave', False)),
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
