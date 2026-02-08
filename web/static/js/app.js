/* ═══════════════════════════════════════════════════════════════════════════
   Waverider Web Designer – Frontend Application
   ═══════════════════════════════════════════════════════════════════════════ */

// ─── State ──────────────────────────────────────────────────────────────────
let scene, camera, renderer, controls;
let waveriderGroup = null;
let isGenerating = false;
let currentMeshData = null;
let currentParams = null;

// Display toggles
let showUpper = true;
let showLower = true;
let showLeadingEdge = true;
let showWireframe = false;

// ─── Three.js Setup ─────────────────────────────────────────────────────────
function initThreeJS() {
    const container = document.getElementById('viewer-canvas');

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);

    // Camera
    camera = new THREE.PerspectiveCamera(
        45,
        container.clientWidth / container.clientHeight,
        0.01,
        1000
    );
    camera.position.set(8, 4, 6);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Controls
    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404060, 0.6);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight1.position.set(5, 8, 5);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0x4488ff, 0.3);
    dirLight2.position.set(-5, 2, -3);
    scene.add(dirLight2);

    // Grid helper
    const grid = new THREE.GridHelper(20, 20, 0x1a2a44, 0x111d30);
    grid.position.y = -2;
    scene.add(grid);

    // Axis helper
    const axes = new THREE.AxesHelper(2);
    axes.position.set(-1, -2, -1);
    scene.add(axes);

    // Resize
    window.addEventListener('resize', onResize);

    // Animation loop
    animate();
}

function onResize() {
    const container = document.getElementById('viewer-canvas');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// ─── Mesh Display ───────────────────────────────────────────────────────────
function displayWaverider(meshData) {
    // Remove old waverider
    if (waveriderGroup) {
        scene.remove(waveriderGroup);
        waveriderGroup.traverse(child => {
            if (child.geometry) child.geometry.dispose();
            if (child.material) {
                if (Array.isArray(child.material)) {
                    child.material.forEach(m => m.dispose());
                } else {
                    child.material.dispose();
                }
            }
        });
    }

    waveriderGroup = new THREE.Group();
    const vertices = meshData.vertices;
    const faces = meshData.faces;

    // Determine which faces belong to upper vs lower surface
    // We'll separate by normal direction: upper surface normals point "up" (positive y)
    // Actually, let's build the full mesh and then separate visually

    // Build geometry
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(faces.length * 3 * 3);

    for (let i = 0; i < faces.length; i++) {
        const [a, b, c] = faces[i];
        positions[i * 9 + 0] = vertices[a][0];
        positions[i * 9 + 1] = vertices[a][1];
        positions[i * 9 + 2] = vertices[a][2];
        positions[i * 9 + 3] = vertices[b][0];
        positions[i * 9 + 4] = vertices[b][1];
        positions[i * 9 + 5] = vertices[b][2];
        positions[i * 9 + 6] = vertices[c][0];
        positions[i * 9 + 7] = vertices[c][1];
        positions[i * 9 + 8] = vertices[c][2];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.computeVertexNormals();

    // Separate into upper and lower by face normal Y component
    const upperPositions = [];
    const lowerPositions = [];
    const normalAttr = geometry.getAttribute('normal');

    for (let i = 0; i < faces.length; i++) {
        // Average normal Y for this face
        const ny = (normalAttr.getY(i * 3) + normalAttr.getY(i * 3 + 1) + normalAttr.getY(i * 3 + 2)) / 3;

        const target = ny > 0 ? upperPositions : lowerPositions;
        for (let j = 0; j < 9; j++) {
            target.push(positions[i * 9 + j]);
        }
    }

    // Upper surface mesh
    if (upperPositions.length > 0) {
        const upperGeom = new THREE.BufferGeometry();
        upperGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(upperPositions), 3));
        upperGeom.computeVertexNormals();

        const upperMat = new THREE.MeshPhongMaterial({
            color: 0x06b6d4,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
            shininess: 60,
            flatShading: false,
        });
        const upperWireMat = new THREE.MeshPhongMaterial({
            color: 0x06b6d4,
            wireframe: true,
            transparent: true,
            opacity: 0.6,
        });

        const upperMesh = new THREE.Mesh(upperGeom, showWireframe ? upperWireMat : upperMat);
        upperMesh.name = 'upper';
        upperMesh.visible = showUpper;
        waveriderGroup.add(upperMesh);

        // Keep wireframe version for toggling
        const upperWire = new THREE.Mesh(upperGeom.clone(), upperWireMat);
        upperWire.name = 'upper_wire';
        upperWire.visible = showUpper && showWireframe;
        waveriderGroup.add(upperWire);
    }

    // Lower surface mesh
    if (lowerPositions.length > 0) {
        const lowerGeom = new THREE.BufferGeometry();
        lowerGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lowerPositions), 3));
        lowerGeom.computeVertexNormals();

        const lowerMat = new THREE.MeshPhongMaterial({
            color: 0xf97316,
            transparent: true,
            opacity: 0.85,
            side: THREE.DoubleSide,
            shininess: 60,
            flatShading: false,
        });
        const lowerWireMat = new THREE.MeshPhongMaterial({
            color: 0xf97316,
            wireframe: true,
            transparent: true,
            opacity: 0.6,
        });

        const lowerMesh = new THREE.Mesh(lowerGeom, showWireframe ? lowerWireMat : lowerMat);
        lowerMesh.name = 'lower';
        lowerMesh.visible = showLower;
        waveriderGroup.add(lowerMesh);

        const lowerWire = new THREE.Mesh(lowerGeom.clone(), lowerWireMat);
        lowerWire.name = 'lower_wire';
        lowerWire.visible = showLower && showWireframe;
        waveriderGroup.add(lowerWire);
    }

    // Leading edge lines
    if (meshData.leading_edge && meshData.leading_edge.length > 1) {
        const leGeom = new THREE.BufferGeometry();
        const lePoints = [];
        meshData.leading_edge.forEach(p => lePoints.push(new THREE.Vector3(p[0], p[1], p[2])));
        if (meshData.leading_edge_mirrored) {
            meshData.leading_edge_mirrored.forEach(p => lePoints.push(new THREE.Vector3(p[0], p[1], p[2])));
        }
        leGeom.setFromPoints(lePoints);

        const leMat = new THREE.LineBasicMaterial({ color: 0x22c55e, linewidth: 2 });
        const leLine = new THREE.Line(leGeom, leMat);
        leLine.name = 'leading_edge';
        leLine.visible = showLeadingEdge;
        waveriderGroup.add(leLine);
    }

    scene.add(waveriderGroup);

    // Center camera on the waverider
    fitCameraToObject();

    // Remove empty state
    const emptyState = document.querySelector('.empty-state');
    if (emptyState) emptyState.style.display = 'none';
}

function fitCameraToObject() {
    if (!waveriderGroup) return;

    const box = new THREE.Box3().setFromObject(waveriderGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    controls.target.copy(center);
    camera.position.set(
        center.x + maxDim * 1.5,
        center.y + maxDim * 0.8,
        center.z + maxDim * 1.2
    );
    camera.lookAt(center);
    controls.update();
}

function updateVisibility() {
    if (!waveriderGroup) return;

    waveriderGroup.traverse(child => {
        switch (child.name) {
            case 'upper':
                child.visible = showUpper && !showWireframe;
                break;
            case 'upper_wire':
                child.visible = showUpper && showWireframe;
                break;
            case 'lower':
                child.visible = showLower && !showWireframe;
                break;
            case 'lower_wire':
                child.visible = showLower && showWireframe;
                break;
            case 'leading_edge':
                child.visible = showLeadingEdge;
                break;
        }
    });
}

// ─── API Calls ──────────────────────────────────────────────────────────────
function getParams() {
    return {
        mach: parseFloat(document.getElementById('param-mach').value),
        beta: parseFloat(document.getElementById('param-beta').value),
        height: parseFloat(document.getElementById('param-height').value),
        width: parseFloat(document.getElementById('param-width').value),
        X1: parseFloat(document.getElementById('param-x1').value),
        X2: parseFloat(document.getElementById('param-x2').value),
        X3: parseFloat(document.getElementById('param-x3').value),
        X4: parseFloat(document.getElementById('param-x4').value),
        n_planes: parseInt(document.getElementById('param-nplanes').value),
        n_streamwise: parseInt(document.getElementById('param-nstreamwise').value),
        delta_streamwise: parseFloat(document.getElementById('param-delta').value),
        n_upper_surface: parseInt(document.getElementById('param-nuppersurface').value),
        n_shockwave: parseInt(document.getElementById('param-nshockwave').value),
        match_shockwave: document.getElementById('param-method-shadow').classList.contains('active'),
    };
}

async function generateWaverider() {
    if (isGenerating) return;
    isGenerating = true;

    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    setStatus('Generating waverider geometry...', 'loading');

    try {
        const params = getParams();
        currentParams = params;

        const resp = await fetch('/api/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        const data = await resp.json();

        if (!resp.ok) {
            throw new Error(data.error || 'Generation failed');
        }

        currentMeshData = data.mesh;
        displayWaverider(data.mesh);
        updateInfoPanel(data.info);
        setStatus('Waverider generated successfully', 'success');

        // Enable export buttons
        document.getElementById('btn-export-step').disabled = false;
        document.getElementById('btn-export-stl').disabled = false;

    } catch (err) {
        setStatus('Error: ' + err.message, 'error');
    } finally {
        isGenerating = false;
        btn.disabled = false;
    }
}

async function exportFile(format) {
    if (!currentParams) {
        setStatus('Generate a waverider first', 'error');
        return;
    }

    const btn = document.getElementById(`btn-export-${format}`);
    btn.disabled = true;
    setStatus(`Generating ${format.toUpperCase()} file...`, 'loading');

    try {
        const params = { ...currentParams };
        if (format === 'stl') {
            params.mesh_min_size = parseFloat(document.getElementById('param-meshmin').value);
            params.mesh_max_size = parseFloat(document.getElementById('param-meshmax').value);
        }

        const resp = await fetch(`/api/export/${format}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });

        if (!resp.ok) {
            const errData = await resp.json();
            throw new Error(errData.error || `Export failed`);
        }

        // Download file
        const blob = await resp.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `waverider.${format}`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();

        setStatus(`${format.toUpperCase()} file downloaded`, 'success');

    } catch (err) {
        setStatus('Export error: ' + err.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

async function fetchBetaHint() {
    const mach = parseFloat(document.getElementById('param-mach').value);
    if (isNaN(mach) || mach < 1.1) return;

    try {
        const resp = await fetch(`/api/beta-hint?mach=${mach}`);
        const data = await resp.json();

        const hint = document.getElementById('beta-hint');
        hint.textContent = `\u03B2 range: ${data.beta_min}\u00B0 (Mach angle) | Recommended: ${data.recommended.low}\u00B0 \u2013 ${data.recommended.high}\u00B0 (mid: ${data.recommended.mid}\u00B0)`;
    } catch (err) {
        // Silently ignore
    }
}

async function autoBeta() {
    const mach = parseFloat(document.getElementById('param-mach').value);
    if (isNaN(mach) || mach < 1.1) return;

    try {
        const resp = await fetch(`/api/beta-hint?mach=${mach}`);
        const data = await resp.json();
        document.getElementById('param-beta').value = data.recommended.mid;
        validateConstraints();
    } catch (err) {
        // Silently ignore
    }
}

async function validateConstraints() {
    const X1 = parseFloat(document.getElementById('param-x1').value);
    const X2 = parseFloat(document.getElementById('param-x2').value);
    const width = parseFloat(document.getElementById('param-width').value);
    const height = parseFloat(document.getElementById('param-height').value);

    try {
        const resp = await fetch('/api/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ X1, X2, width, height }),
        });
        const data = await resp.json();

        const badge = document.getElementById('constraint-badge');
        if (data.valid) {
            badge.className = 'constraint-badge valid';
            badge.innerHTML = `&#10003; Valid &mdash; Max X2 &asymp; ${data.max_X2.toFixed(3)} (current: ${data.current_X2.toFixed(3)})`;
        } else {
            badge.className = 'constraint-badge invalid';
            badge.innerHTML = `&#9888; X2 too large! Max X2 &asymp; ${data.max_X2.toFixed(3)} for current X1=${X1.toFixed(3)}`;
        }
    } catch (err) {
        // Silently ignore
    }
}

// ─── UI Helpers ─────────────────────────────────────────────────────────────
function setStatus(message, type) {
    const bar = document.getElementById('status-bar');
    bar.className = 'status-bar ' + (type || '');
    bar.innerHTML = (type === 'loading' ? '<span class="spinner"></span>' : '') + message;
}

function updateInfoPanel(info) {
    const panel = document.getElementById('info-panel');
    panel.classList.add('visible');

    document.getElementById('info-length').textContent = info.length.toFixed(3) + ' m';
    document.getElementById('info-volume').textContent = info.volume.toFixed(4) + ' m\u00B3';
    document.getElementById('info-method').textContent = info.method;
    document.getElementById('info-deflection').textContent = info.deflection_angle.toFixed(2) + '\u00B0';
}

// ─── Slider sync ────────────────────────────────────────────────────────────
function syncSlider(sliderId, displayId) {
    const slider = document.getElementById(sliderId);
    const display = document.getElementById(displayId);
    display.textContent = parseFloat(slider.value).toFixed(2);
    slider.addEventListener('input', () => {
        display.textContent = parseFloat(slider.value).toFixed(2);
        validateConstraints();
    });
}

// ─── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();

    // Sync sliders
    syncSlider('param-x1', 'display-x1');
    syncSlider('param-x2', 'display-x2');
    syncSlider('param-x3', 'display-x3');
    syncSlider('param-x4', 'display-x4');

    // Method tabs
    document.getElementById('param-method-osculating').addEventListener('click', () => {
        document.getElementById('param-method-osculating').classList.add('active');
        document.getElementById('param-method-shadow').classList.remove('active');
    });
    document.getElementById('param-method-shadow').addEventListener('click', () => {
        document.getElementById('param-method-shadow').classList.add('active');
        document.getElementById('param-method-osculating').classList.remove('active');
    });

    // Generate button
    document.getElementById('btn-generate').addEventListener('click', generateWaverider);

    // Export buttons
    document.getElementById('btn-export-step').addEventListener('click', () => exportFile('step'));
    document.getElementById('btn-export-stl').addEventListener('click', () => exportFile('stl'));

    // Auto beta
    document.getElementById('btn-auto-beta').addEventListener('click', autoBeta);

    // Mach change -> update beta hint
    document.getElementById('param-mach').addEventListener('change', fetchBetaHint);
    document.getElementById('param-mach').addEventListener('input', fetchBetaHint);

    // Constraint validation on relevant param changes
    ['param-x1', 'param-x2', 'param-width', 'param-height'].forEach(id => {
        document.getElementById(id).addEventListener('change', validateConstraints);
        document.getElementById(id).addEventListener('input', validateConstraints);
    });

    // Viewer toggle buttons
    document.getElementById('btn-toggle-upper').addEventListener('click', function () {
        showUpper = !showUpper;
        this.classList.toggle('active', showUpper);
        updateVisibility();
    });
    document.getElementById('btn-toggle-lower').addEventListener('click', function () {
        showLower = !showLower;
        this.classList.toggle('active', showLower);
        updateVisibility();
    });
    document.getElementById('btn-toggle-le').addEventListener('click', function () {
        showLeadingEdge = !showLeadingEdge;
        this.classList.toggle('active', showLeadingEdge);
        updateVisibility();
    });
    document.getElementById('btn-toggle-wireframe').addEventListener('click', function () {
        showWireframe = !showWireframe;
        this.classList.toggle('active', showWireframe);
        updateVisibility();
    });
    document.getElementById('btn-reset-view').addEventListener('click', fitCameraToObject);

    // Mesh preset buttons
    document.querySelectorAll('.mesh-preset-btn').forEach(btn => {
        btn.addEventListener('click', function () {
            document.querySelectorAll('.mesh-preset-btn').forEach(b => b.classList.remove('active'));
            this.classList.add('active');
            document.getElementById('param-meshmin').value = this.dataset.min;
            document.getElementById('param-meshmax').value = this.dataset.max;
        });
    });

    // Collapsible sections
    document.querySelectorAll('.section-header').forEach(header => {
        header.addEventListener('click', () => {
            header.closest('.section').classList.toggle('collapsed');
        });
    });

    // Initial setup
    fetchBetaHint();
    validateConstraints();
    setStatus('Ready. Configure parameters and click Generate.', '');
});
