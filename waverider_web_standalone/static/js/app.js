/* ═══════════════════════════════════════════════════════════════════════════
   Waverider Web Designer — Frontend  (v6)
   ═══════════════════════════════════════════════════════════════════════════ */
console.log('app.js v6 loaded');

let scene, camera, renderer, controls;
let waveriderGroup = null;
let axesHelper = null, gridHelper = null;
let isGenerating = false;
let currentParams = null;
let currentMethod = 'osc'; // 'osc' or 'shadow'

let showUpper = true, showLower = true, showLE = true, showCG = true, showWire = false, showAxes = true;

/* ─── Three.js ───────────────────────────────────────────────────────────── */

function initViewer() {
  const el = document.getElementById('viewer-canvas');
  let w = el.clientWidth, h = el.clientHeight;
  console.log('initViewer: container size', w, 'x', h);
  if (w === 0 || h === 0) { setTimeout(initViewer, 100); return; }

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1000);
  camera.position.set(8, 4, 6);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;

  scene.add(new THREE.AmbientLight(0x404060, 0.6));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.8);
  d1.position.set(5, 8, 5); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x4488ff, 0.3);
  d2.position.set(-5, 2, -3); scene.add(d2);

  gridHelper = new THREE.GridHelper(20, 20, 0x1a1a1a, 0x111111);
  gridHelper.position.y = -2; scene.add(gridHelper);
  axesHelper = new THREE.AxesHelper(1.5);
  axesHelper.position.set(-1, -2, -1); scene.add(axesHelper);

  window.addEventListener('resize', onResize);
  animate();
}

function onResize() {
  const el = document.getElementById('viewer-canvas');
  if (!renderer) return;
  camera.aspect = el.clientWidth / el.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(el.clientWidth, el.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  if (controls) controls.update();
  if (renderer && scene && camera) renderer.render(scene, camera);
}

/* ─── Mesh Rendering ─────────────────────────────────────────────────────── */

function displayWaverider(data) {
  if (waveriderGroup) {
    scene.remove(waveriderGroup);
    waveriderGroup.traverse(c => {
      if (c.geometry) c.geometry.dispose();
      if (c.material) (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose());
    });
  }

  waveriderGroup = new THREE.Group();
  const { vertices, faces } = data;
  console.log('Waverider mesh: ' + vertices.length + ' vertices, ' + faces.length + ' faces');

  // Build full geometry to compute vertex normals for upper/lower classification
  const positions = new Float32Array(vertices.length * 3);
  for (let i = 0; i < vertices.length; i++) {
    positions[i*3] = vertices[i][0]; positions[i*3+1] = vertices[i][1]; positions[i*3+2] = vertices[i][2];
  }
  const allIndices = [];
  for (let i = 0; i < faces.length; i++) allIndices.push(faces[i][0], faces[i][1], faces[i][2]);
  const fullGeo = new THREE.BufferGeometry();
  fullGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  fullGeo.setIndex(allIndices);
  fullGeo.computeVertexNormals();
  const normals = fullGeo.getAttribute('normal');

  // Classify faces: compute face normal from vertex normals average
  const upperFaces = [], lowerFaces = [];
  for (let i = 0; i < faces.length; i++) {
    const avgNy = (normals.getY(faces[i][0]) + normals.getY(faces[i][1]) + normals.getY(faces[i][2])) / 3;
    if (avgNy > 0) lowerFaces.push(faces[i]);
    else upperFaces.push(faces[i]);
  }

  // Helper to build a mesh from a subset of faces
  function buildSurface(surfaceFaces, name, color) {
    const remap = {}, newVerts = [], newIdx = [];
    for (const f of surfaceFaces) {
      for (const vi of f) {
        if (remap[vi] === undefined) { remap[vi] = newVerts.length; newVerts.push(vertices[vi]); }
      }
      newIdx.push(remap[f[0]], remap[f[1]], remap[f[2]]);
    }
    const pos = new Float32Array(newVerts.length * 3);
    const col = new Float32Array(newVerts.length * 3);
    for (let i = 0; i < newVerts.length; i++) {
      pos[i*3] = newVerts[i][0]; pos[i*3+1] = newVerts[i][1]; pos[i*3+2] = newVerts[i][2];
      col[i*3] = color[0]; col[i*3+1] = color[1]; col[i*3+2] = color[2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geo.setIndex(newIdx);
    geo.computeVertexNormals();

    const solid = new THREE.Mesh(geo, new THREE.MeshPhongMaterial({ vertexColors: true, side: THREE.DoubleSide, shininess: 60, transparent: true, opacity: 0.85 }));
    solid.name = name;
    waveriderGroup.add(solid);

    const wire = new THREE.Mesh(geo.clone(), new THREE.MeshBasicMaterial({ vertexColors: true, wireframe: true, transparent: true, opacity: 0.5 }));
    wire.name = name + '_wire'; wire.visible = false;
    waveriderGroup.add(wire);
  }

  buildSurface(upperFaces, 'upper', [0.96, 0.62, 0.04]);
  buildSurface(lowerFaces, 'lower', [0.23, 0.51, 0.96]);
  fullGeo.dispose();

  // Leading edge
  if (data.leading_edge && data.leading_edge.length > 1) {
    const pts = [];
    if (data.leading_edge_mirrored) data.leading_edge_mirrored.forEach(p => pts.push(new THREE.Vector3(p[0], p[1], p[2])));
    const startIdx = data.leading_edge_mirrored ? 1 : 0;
    for (let i = startIdx; i < data.leading_edge.length; i++) {
      const p = data.leading_edge[i]; pts.push(new THREE.Vector3(p[0], p[1], p[2]));
    }
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 }));
    line.name = 'leading_edge';
    waveriderGroup.add(line);
  }

  // CG marker (green star)
  if (data.cg && data.cg.length === 3) {
    const cgGroup = new THREE.Group();
    cgGroup.name = 'cg_marker';

    // Sphere
    const sgeo = new THREE.SphereGeometry(0.04, 16, 16);
    const smat = new THREE.MeshBasicMaterial({ color: 0x10b981 });
    const sphere = new THREE.Mesh(sgeo, smat);
    sphere.position.set(data.cg[0], data.cg[1], data.cg[2]);
    cgGroup.add(sphere);

    // Cross lines through CG for visibility
    const cLen = 0.08;
    const cgPos = new THREE.Vector3(data.cg[0], data.cg[1], data.cg[2]);
    [new THREE.Vector3(cLen,0,0), new THREE.Vector3(0,cLen,0), new THREE.Vector3(0,0,cLen)].forEach(dir => {
      const pts = [cgPos.clone().sub(dir), cgPos.clone().add(dir)];
      const lg = new THREE.BufferGeometry().setFromPoints(pts);
      cgGroup.add(new THREE.Line(lg, new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 })));
    });

    waveriderGroup.add(cgGroup);
    console.log('CG at', data.cg);
  }

  scene.add(waveriderGroup);
  updateVisibility();
  fitCamera();
  document.querySelector('.empty-state').style.display = 'none';
  document.getElementById('viewer-legend').classList.add('visible');
}

function fitCamera() {
  if (!waveriderGroup) return;
  const box = new THREE.Box3().setFromObject(waveriderGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  if (maxDim === 0) return;
  controls.target.copy(center);
  camera.position.set(center.x + maxDim * 1.5, center.y + maxDim * 0.8, center.z + maxDim * 1.2);
  camera.lookAt(center);
  controls.update();
}

function updateVisibility() {
  if (!waveriderGroup) return;
  waveriderGroup.traverse(c => {
    if (c.name === 'upper')        c.visible = showUpper && !showWire;
    if (c.name === 'upper_wire')   c.visible = showUpper && showWire;
    if (c.name === 'lower')        c.visible = showLower && !showWire;
    if (c.name === 'lower_wire')   c.visible = showLower && showWire;
    if (c.name === 'leading_edge') c.visible = showLE;
    if (c.name === 'cg_marker')    c.visible = showCG;
  });
  if (axesHelper) axesHelper.visible = showAxes;
  if (gridHelper) gridHelper.visible = showAxes;
}

/* ─── Method switching ────────────────────────────────────────────────────── */

function switchMethod(method) {
  currentMethod = method;
  document.getElementById('panel-osc').style.display = method === 'osc' ? '' : 'none';
  document.getElementById('panel-shadow').style.display = method === 'shadow' ? '' : 'none';

  document.getElementById('method-osc').classList.toggle('active', method === 'osc');
  document.getElementById('method-shadow').classList.toggle('active', method === 'shadow');
}

/* ─── API ────────────────────────────────────────────────────────────────── */

function getParams() {
  const base = {
    mach: parseFloat(document.getElementById('p-mach').value),
    beta: parseFloat(document.getElementById('p-beta').value),
  };

  if (currentMethod === 'shadow') {
    return {
      ...base,
      method: 'shadow',
      poly_order: parseInt(document.getElementById('p-poly-order').value),
      A3: parseFloat(document.getElementById('p-a3').value),
      A2: parseFloat(document.getElementById('p-a2').value),
      A0: parseFloat(document.getElementById('p-a0').value),
      n_le: parseInt(document.getElementById('p-nle').value),
      n_streamwise: parseInt(document.getElementById('p-sw-shadow').value),
      length: parseFloat(document.getElementById('p-length').value),
    };
  }

  return {
    ...base,
    method: 'osc',
    height: parseFloat(document.getElementById('p-height').value),
    width: parseFloat(document.getElementById('p-width').value),
    match_shockwave: document.getElementById('p-match-shock').checked,
    X1: parseFloat(document.getElementById('p-x1').value),
    X2: parseFloat(document.getElementById('p-x2').value),
    X3: parseFloat(document.getElementById('p-x3').value),
    X4: parseFloat(document.getElementById('p-x4').value),
    n_planes: parseInt(document.getElementById('p-nplanes').value),
    n_streamwise: parseInt(document.getElementById('p-nstream').value),
    delta_streamwise: parseFloat(document.getElementById('p-delta').value),
    n_upper_surface: parseInt(document.getElementById('p-nupper').value),
    n_shockwave: parseInt(document.getElementById('p-nshock').value),
  };
}

async function generate() {
  if (isGenerating) return;
  isGenerating = true;
  const btn = document.getElementById('btn-generate');
  btn.disabled = true;
  setStatus('Generating waverider geometry\u2026', 'loading');

  try {
    const params = getParams();
    currentParams = params;
    const resp = await fetch('/api/generate', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(params) });
    const d = await resp.json();
    if (!resp.ok) throw new Error(d.error || 'Generation failed');
    displayWaverider(d.mesh);
    updateInfo(d.info);
    setStatus('Waverider generated successfully', 'success');
    document.getElementById('btn-step').disabled = false;
    document.getElementById('btn-stl').disabled = false;
  } catch (e) {
    setStatus('Error: ' + e.message, 'error');
  } finally {
    isGenerating = false;
    btn.disabled = false;
  }
}

async function exportFile(fmt) {
  if (!currentParams) return setStatus('Generate a waverider first', 'error');
  const btn = document.getElementById('btn-' + fmt);
  btn.disabled = true;
  setStatus('Generating ' + fmt.toUpperCase() + ' file\u2026', 'loading');
  try {
    const params = { ...currentParams };
    if (fmt === 'stl') {
      params.mesh_min_size = parseFloat(document.getElementById('p-meshmin').value);
      params.mesh_max_size = parseFloat(document.getElementById('p-meshmax').value);
    }
    const resp = await fetch('/api/export/' + fmt, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(params) });
    if (!resp.ok) { const e = await resp.json(); throw new Error(e.error); }
    const blob = await resp.blob();
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'waverider.' + fmt;
    document.body.appendChild(a); a.click();
    URL.revokeObjectURL(a.href); a.remove();
    setStatus(fmt.toUpperCase() + ' downloaded', 'success');
  } catch (e) {
    setStatus('Export error: ' + e.message, 'error');
  } finally { btn.disabled = false; }
}

async function fetchBetaHint() {
  const M = parseFloat(document.getElementById('p-mach').value);
  if (isNaN(M) || M < 1.1) return;
  try {
    const d = await (await fetch('/api/beta-hint?mach=' + M)).json();
    document.getElementById('beta-hint').textContent =
      '\u03B2 range: ' + d.beta_min + '\u00B0 (Mach angle)  \u00B7  Recommended: ' +
      d.recommended.low + '\u00B0 \u2013 ' + d.recommended.high + '\u00B0 (mid: ' + d.recommended.mid + '\u00B0)';
  } catch (_) {}
}

async function autoBeta() {
  const M = parseFloat(document.getElementById('p-mach').value);
  if (isNaN(M) || M < 1.1) return;
  try {
    const d = await (await fetch('/api/beta-hint?mach=' + M)).json();
    document.getElementById('p-beta').value = d.recommended.mid;
    validateConstraints();
  } catch (_) {}
}

async function validateConstraints() {
  if (currentMethod !== 'osc') return;
  const X1 = parseFloat(document.getElementById('p-x1').value);
  const X2 = parseFloat(document.getElementById('p-x2').value);
  const w = parseFloat(document.getElementById('p-width').value);
  const h = parseFloat(document.getElementById('p-height').value);
  try {
    const d = await (await fetch('/api/validate', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({X1, X2, width: w, height: h}) })).json();
    const el = document.getElementById('constraint-badge');
    if (d.valid) {
      el.className = 'constraint-badge valid';
      el.innerHTML = '\u2713 Valid \u2014 Max X2 \u2248 ' + d.max_X2.toFixed(3) + ' (current: ' + d.current_X2.toFixed(3) + ')';
    } else {
      el.className = 'constraint-badge invalid';
      el.innerHTML = '\u26A0 X2 too large! Max X2 \u2248 ' + d.max_X2.toFixed(3) + ' for X1=' + X1.toFixed(3);
    }
  } catch (_) {}
}

/* ─── UI Helpers ─────────────────────────────────────────────────────────── */

function setStatus(msg, type) {
  const el = document.getElementById('status-bar');
  el.className = 'status-bar ' + (type || '');
  el.innerHTML = (type === 'loading' ? '<span class="spinner"></span>' : '') + msg;
}

function updateInfo(info) {
  document.getElementById('info-panel').classList.add('visible');
  document.getElementById('info-method').textContent = info.method || '-';
  document.getElementById('info-mach').textContent = info.mach ? info.mach.toFixed(1) : '-';
  document.getElementById('info-beta').textContent = info.beta ? info.beta.toFixed(1) + '\u00B0' : '-';
  document.getElementById('info-cone').textContent = info.cone_angle ? info.cone_angle.toFixed(2) + '\u00B0' : '-';
  document.getElementById('info-psm').textContent = info.post_shock_mach ? info.post_shock_mach.toFixed(2) : '-';
  document.getElementById('info-length').textContent = info.length ? info.length.toFixed(4) + ' m' : '-';
  document.getElementById('info-area').textContent = info.planform_area ? info.planform_area.toFixed(4) + ' m\u00B2' : '-';
  document.getElementById('info-volume').textContent = info.volume ? info.volume.toFixed(6) + ' m\u00B3' : '-';
  document.getElementById('info-cg').textContent = info.cg ? '[' + info.cg.map(v => v.toFixed(4)).join(', ') + ']' : '-';
}

function syncSlider(id, displayId) {
  const s = document.getElementById(id);
  const d = document.getElementById(displayId);
  d.textContent = parseFloat(s.value).toFixed(2);
  s.addEventListener('input', () => { d.textContent = parseFloat(s.value).toFixed(2); validateConstraints(); });
}

/* ─── Init ───────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  initViewer();

  // Sliders
  syncSlider('p-x1', 'v-x1');
  syncSlider('p-x2', 'v-x2');
  syncSlider('p-x3', 'v-x3');
  syncSlider('p-x4', 'v-x4');

  // Method tabs
  document.getElementById('method-osc').addEventListener('click', () => switchMethod('osc'));
  document.getElementById('method-shadow').addEventListener('click', () => switchMethod('shadow'));

  // Polynomial order toggle
  document.getElementById('p-poly-order').addEventListener('change', function() {
    document.getElementById('field-a3').style.display = this.value === '3' ? '' : 'none';
  });

  // Generate
  document.getElementById('btn-generate').addEventListener('click', generate);
  document.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.target.matches('input,select')) generate(); });

  // Export
  document.getElementById('btn-step').addEventListener('click', () => exportFile('step'));
  document.getElementById('btn-stl').addEventListener('click', () => exportFile('stl'));

  // Beta
  document.getElementById('btn-auto-beta').addEventListener('click', autoBeta);
  document.getElementById('p-mach').addEventListener('change', fetchBetaHint);
  document.getElementById('p-mach').addEventListener('input', fetchBetaHint);

  // Constraint validation
  ['p-x1','p-x2','p-width','p-height'].forEach(id => {
    document.getElementById(id).addEventListener('change', validateConstraints);
    document.getElementById(id).addEventListener('input', validateConstraints);
  });

  // Viewer toggles
  document.getElementById('t-upper').addEventListener('click', function() { showUpper = !showUpper; this.classList.toggle('active', showUpper); updateVisibility(); });
  document.getElementById('t-lower').addEventListener('click', function() { showLower = !showLower; this.classList.toggle('active', showLower); updateVisibility(); });
  document.getElementById('t-le').addEventListener('click', function() { showLE = !showLE; this.classList.toggle('active', showLE); updateVisibility(); });
  document.getElementById('t-cg').addEventListener('click', function() { showCG = !showCG; this.classList.toggle('active', showCG); updateVisibility(); });
  document.getElementById('t-wire').addEventListener('click', function() { showWire = !showWire; this.classList.toggle('active', showWire); updateVisibility(); });
  document.getElementById('t-axes').addEventListener('click', function() { showAxes = !showAxes; this.classList.toggle('active', showAxes); updateVisibility(); });
  document.getElementById('t-reset').addEventListener('click', fitCamera);

  // Mesh presets
  document.querySelectorAll('.mesh-preset').forEach(b => {
    b.addEventListener('click', function() {
      document.querySelectorAll('.mesh-preset').forEach(x => x.classList.remove('active'));
      this.classList.add('active');
      document.getElementById('p-meshmin').value = this.dataset.min;
      document.getElementById('p-meshmax').value = this.dataset.max;
    });
  });

  // Collapsible sections
  document.querySelectorAll('.section-toggle').forEach(t => {
    t.addEventListener('click', () => t.closest('.section').classList.toggle('collapsed'));
  });

  // Init
  fetchBetaHint();
  validateConstraints();
  setStatus('Ready \u2014 configure parameters and click Generate', '');
});
