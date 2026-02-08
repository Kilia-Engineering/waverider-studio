/* ═══════════════════════════════════════════════════════════════════════════
   Waverider Web Designer — Frontend
   ═══════════════════════════════════════════════════════════════════════════ */

let scene, camera, renderer, controls;
let waveriderGroup = null;
let isGenerating = false;
let currentParams = null;

let showUpper = true, showLower = true, showLE = true, showWire = false;

/* ─── Three.js ───────────────────────────────────────────────────────────── */

function initViewer() {
  const el = document.getElementById('viewer-canvas');

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0a0a);

  camera = new THREE.PerspectiveCamera(45, el.clientWidth / el.clientHeight, 0.01, 1000);
  camera.position.set(8, 4, 6);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(el.clientWidth, el.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.8;

  // Lighting
  scene.add(new THREE.AmbientLight(0x404060, 0.6));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.8);
  d1.position.set(5, 8, 5);
  scene.add(d1);
  const d2 = new THREE.DirectionalLight(0x4488ff, 0.3);
  d2.position.set(-5, 2, -3);
  scene.add(d2);

  // Grid
  const grid = new THREE.GridHelper(20, 20, 0x1a1a1a, 0x111111);
  grid.position.y = -2;
  scene.add(grid);

  // Axes
  const axes = new THREE.AxesHelper(1.5);
  axes.position.set(-1, -2, -1);
  scene.add(axes);

  window.addEventListener('resize', onResize);
  animate();
}

function onResize() {
  const el = document.getElementById('viewer-canvas');
  camera.aspect = el.clientWidth / el.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(el.clientWidth, el.clientHeight);
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
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

  // Build full geometry
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(faces.length * 9);
  for (let i = 0; i < faces.length; i++) {
    const [a, b, c] = faces[i];
    for (let k = 0; k < 3; k++) { pos[i*9+k] = vertices[a][k]; pos[i*9+3+k] = vertices[b][k]; pos[i*9+6+k] = vertices[c][k]; }
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.computeVertexNormals();

  const norms = geo.getAttribute('normal');
  const upperPos = [], lowerPos = [];
  for (let i = 0; i < faces.length; i++) {
    const ny = (norms.getY(i*3) + norms.getY(i*3+1) + norms.getY(i*3+2)) / 3;
    const arr = ny > 0 ? upperPos : lowerPos;
    for (let j = 0; j < 9; j++) arr.push(pos[i*9+j]);
  }

  // Upper surface — amber-tinted
  if (upperPos.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(upperPos), 3));
    g.computeVertexNormals();
    waveriderGroup.add(new THREE.Mesh(g, new THREE.MeshPhongMaterial({
      color: 0xf59e0b, transparent: true, opacity: 0.75, side: THREE.DoubleSide, shininess: 80,
    })));
    waveriderGroup.children[waveriderGroup.children.length-1].name = 'upper';
    const gw = g.clone();
    waveriderGroup.add(new THREE.Mesh(gw, new THREE.MeshPhongMaterial({
      color: 0xf59e0b, wireframe: true, transparent: true, opacity: 0.5,
    })));
    waveriderGroup.children[waveriderGroup.children.length-1].name = 'upper_wire';
  }

  // Lower surface — blue-tinted
  if (lowerPos.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lowerPos), 3));
    g.computeVertexNormals();
    waveriderGroup.add(new THREE.Mesh(g, new THREE.MeshPhongMaterial({
      color: 0x3b82f6, transparent: true, opacity: 0.75, side: THREE.DoubleSide, shininess: 80,
    })));
    waveriderGroup.children[waveriderGroup.children.length-1].name = 'lower';
    const gw = g.clone();
    waveriderGroup.add(new THREE.Mesh(gw, new THREE.MeshPhongMaterial({
      color: 0x3b82f6, wireframe: true, transparent: true, opacity: 0.5,
    })));
    waveriderGroup.children[waveriderGroup.children.length-1].name = 'lower_wire';
  }

  // Leading edge — green
  if (data.leading_edge && data.leading_edge.length > 1) {
    const pts = [];
    data.leading_edge.forEach(p => pts.push(new THREE.Vector3(p[0], p[1], p[2])));
    if (data.leading_edge_mirrored) data.leading_edge_mirrored.forEach(p => pts.push(new THREE.Vector3(p[0], p[1], p[2])));
    const g = new THREE.BufferGeometry().setFromPoints(pts);
    const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x10b981, linewidth: 2 }));
    line.name = 'leading_edge';
    waveriderGroup.add(line);
  }

  scene.add(waveriderGroup);
  updateVisibility();
  fitCamera();
  document.querySelector('.empty-state').style.display = 'none';
}

function fitCamera() {
  if (!waveriderGroup) return;
  const box = new THREE.Box3().setFromObject(waveriderGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  controls.target.copy(center);
  camera.position.set(center.x + maxDim * 1.5, center.y + maxDim * 0.8, center.z + maxDim * 1.2);
  camera.lookAt(center);
  controls.update();
}

function updateVisibility() {
  if (!waveriderGroup) return;
  waveriderGroup.traverse(c => {
    if (c.name === 'upper')       c.visible = showUpper && !showWire;
    if (c.name === 'upper_wire')  c.visible = showUpper && showWire;
    if (c.name === 'lower')       c.visible = showLower && !showWire;
    if (c.name === 'lower_wire')  c.visible = showLower && showWire;
    if (c.name === 'leading_edge') c.visible = showLE;
  });
}

/* ─── API ────────────────────────────────────────────────────────────────── */

function getParams() {
  return {
    mach:            parseFloat(document.getElementById('p-mach').value),
    beta:            parseFloat(document.getElementById('p-beta').value),
    height:          parseFloat(document.getElementById('p-height').value),
    width:           parseFloat(document.getElementById('p-width').value),
    X1:              parseFloat(document.getElementById('p-x1').value),
    X2:              parseFloat(document.getElementById('p-x2').value),
    X3:              parseFloat(document.getElementById('p-x3').value),
    X4:              parseFloat(document.getElementById('p-x4').value),
    n_planes:        parseInt(document.getElementById('p-nplanes').value),
    n_streamwise:    parseInt(document.getElementById('p-nstream').value),
    delta_streamwise:parseFloat(document.getElementById('p-delta').value),
    n_upper_surface: parseInt(document.getElementById('p-nupper').value),
    n_shockwave:     parseInt(document.getElementById('p-nshock').value),
    match_shockwave: document.getElementById('method-shadow').classList.contains('active'),
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
    document.body.appendChild(a);
    a.click();
    URL.revokeObjectURL(a.href);
    a.remove();
    setStatus(fmt.toUpperCase() + ' downloaded', 'success');
  } catch (e) {
    setStatus('Export error: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
  }
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
  document.getElementById('info-method').textContent = info.method;
  document.getElementById('info-length').textContent = info.length.toFixed(3) + ' m';
  document.getElementById('info-volume').textContent = info.volume.toFixed(4) + ' m\u00B3';
  document.getElementById('info-deflection').textContent = info.deflection_angle.toFixed(2) + '\u00B0';
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
  document.getElementById('method-osc').addEventListener('click', function() {
    this.classList.add('active'); document.getElementById('method-shadow').classList.remove('active');
  });
  document.getElementById('method-shadow').addEventListener('click', function() {
    this.classList.add('active'); document.getElementById('method-osc').classList.remove('active');
  });

  // Generate
  document.getElementById('btn-generate').addEventListener('click', generate);

  // Keyboard shortcut: Enter to generate
  document.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.target.matches('input')) generate(); });

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
  document.getElementById('t-wire').addEventListener('click', function() { showWire = !showWire; this.classList.toggle('active', showWire); updateVisibility(); });
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
