<script>
const M = __MODEL__;
const LBL = __LABELS__;
const STATE = ['clinical only', 'CMP', 'CMP+CBC'], NEXT = ['the CMP', 'the CBC', 'a urine albumin/creatinine ratio'];
const STEP_NEW = [['age', 'female', 'bmi', 'sbp', 'smoker', 'diabetes'], __CMP__, __CBC__, ['acr']];
const Q80 = 1.2815515655446004;
function val(f) { const el = document.querySelector('[data-f="' + f + '"]'); if (!el || el.value === '') return null; const v = parseFloat(el.value); return isNaN(v) ? null : v; }
function feat(c, v) { if (v === null || v === undefined) return null; if (c in M.log_offset) { const raw = (c === 'acr') ? v / 100 : v; return Math.log(Math.max(raw, 0) + M.log_offset[c]); } return v; }
function risk(m, get) { let s = 0; for (let i = 0; i < m.columns.length; i++) { const z = feat(m.columns[i], get(m.columns[i])); if (z === null) return null; s += m.coef[i] * (z - m.mean[i]) / m.sd[i]; } return 1 - Math.exp(-m.H0 * Math.exp(s)); }
function solve(A, B) {            // A x = b for each column b of B; A symmetric positive definite (Cholesky)
  const n = A.length, L = A.map(r => r.map(() => 0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
    let s = A[i][j]; for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
    L[i][j] = (i === j) ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
  }
  return B.map(b => { const y = []; for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y.push(s / L[i][i]); }
    const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; } return x; });
}
function predict(m, get) {        // missing inputs replaced by their conditional mean in training data; 80% range from the conditional variance
  const cols = m.columns, z = cols.map(c => feat(c, get(c)));
  if (z[cols.indexOf('age')] === null || z[cols.indexOf('female')] === null) return null;
  const pos = {}; M.joint.names.forEach((n, i) => pos[n] = i);
  const mu = M.joint.mean, S = M.joint.cov, w = m.coef.map((b, i) => b / m.sd[i]);
  const obs = [], mis = []; cols.forEach((c, i) => (z[i] === null ? mis : obs).push(i));
  let v = 0;
  if (mis.length) {
    const jo = obs.map(i => pos[cols[i]]), jm = mis.map(i => pos[cols[i]]);
    const Soo = jo.map(a => jo.map(b => S[a][b]));
    const X = solve(Soo, jm.map(b => jo.map(a => S[a][b])));      // row r: S_oo^-1 S_o,m[r]
    const d = obs.map(i => z[i] - mu[pos[cols[i]]]);
    mis.forEach((i, r) => { let s = mu[jm[r]]; for (let q = 0; q < obs.length; q++) s += X[r][q] * d[q]; z[i] = s; });
    for (let r = 0; r < jm.length; r++) for (let q = 0; q < jm.length; q++) {
      let c = S[jm[r]][jm[q]]; for (let o = 0; o < jo.length; o++) c -= X[r][o] * S[jo[o]][jm[q]];
      v += w[mis[r]] * c * w[mis[q]];
    }
  }
  let s = 0; cols.forEach((c, i) => s += w[i] * (z[i] - m.mean[i]));
  const sd = Math.sqrt(Math.max(v, 0)), f = x => 1 - Math.exp(-m.H0 * Math.exp(x));
  return { r: f(s), lo: f(s - Q80 * sd), hi: f(s + Q80 * sd), missing: mis.map(i => cols[i]) };
}
const pct = r => (100 * r).toFixed(1) + '%';
const names = cs => cs.map(c => LBL[c] || c).join(', ');
const verdict = r => r >= M.tau ? '<b class="hi">at or above ' + (100 * M.tau) + '%: flag for a structured preventive assessment</b>' : '<b class="stop">below ' + (100 * M.tau) + '%</b>';
function riskText(p) { return 'Estimated 9-year risk <b>' + pct(p.r) + '</b>' + (p.missing.length ? ' (80% range ' + pct(p.lo) + '–' + pct(p.hi) + '; ' + p.missing.length + ' missing value' + (p.missing.length > 1 ? 's' : '') + ' estimated from the other results)' : ''); }
function rangeNotes(prefix, cols) {
  const out = []; cols.forEach(c => { const v = val(prefix + c); if (v === null || !(c in M.range)) return; const [a, b] = M.range[c];
    if (v < a || v > b) out.push((LBL[c] || c) + ' ' + v + ' is outside the range seen in training (' + (+a.toPrecision(3)) + '–' + (+b.toPrecision(3)) + '); check the unit.'); });
  return out.length ? '<br><small>' + out.join('<br>') + '</small>' : '';
}
function updateSeq() {
  const iv = M.chart[document.getElementById('V').value]; let open = true; const get = c => val('s_' + c);
  for (let k = 0; k < 4; k++) {
    const sec = document.getElementById('s' + k), out = document.getElementById('o' + k); sec.classList.toggle('off', !open);
    if (!open) { out.textContent = ''; continue; }
    const entered = k === 0 ? (get('age') !== null && get('female') !== null) : STEP_NEW[k].some(c => get(c) !== null);
    if (!entered) { out.textContent = k === 0 ? 'Enter age and sex, and the other items if known.' : 'Enter the results of this step; leave any missing result empty.'; open = false; continue; }
    const p = predict(M.models['step' + k], get), notes = rangeNotes('s_', STEP_NEW[k]);
    if (k === 3) { const u = p.missing.length && (p.lo < M.tau) !== (p.hi < M.tau);
      out.innerHTML = riskText(p) + ', ' + verdict(p.r) + '.' + (u ? ' <b class="go">The missing values could change this result: measure ' + names(p.missing) + '.</b>' : '') + ' All tests done (US $' + M.step_costs_US[3].toFixed(2) + ').' + notes; open = false; continue; }
    const band = iv[STATE[k]], inside = x => band.some(([a, b]) => x >= a && x <= b), rng = band.map(([a, b]) => pct(a) + '–' + pct(b)).join(', ');
    const unsure = p.missing.length && (inside(p.lo) !== inside(p.r) || inside(p.hi) !== inside(p.r));
    const warn = unsure ? ' <b class="go">The missing values could change this recommendation: measure ' + names(p.missing) + '.</b>' : '';
    if (inside(p.r)) out.innerHTML = riskText(p) + ', inside ' + rng + ': <b class="go">order ' + NEXT[k] + '.</b>' + warn + notes;
    else { out.innerHTML = riskText(p) + ', outside ' + rng + ': <b class="stop">stop here</b>; ' + verdict(p.r) + '. Laboratory cost so far US $' + M.step_costs_US[k].toFixed(2) + '.' + warn + notes; open = false; }
  }
}
function updateFix() {
  const m = M.models[document.getElementById('panel').value], get = c => val('f_' + c);
  document.querySelectorAll('#tab-fix label[data-a]').forEach(l => l.hidden = !m.columns.includes(l.dataset.a));
  const out = document.getElementById('of'), tests = m.columns.filter(c => c !== 'age' && c !== 'female');
  if (get('age') === null || get('female') === null || !tests.some(c => get(c) !== null)) { out.innerHTML = 'Enter age, sex and the test results; leave any missing result empty.'; return; }
  const p = predict(m, get), unsure = p.missing.length && (p.lo < M.tau) !== (p.hi < M.tau);
  out.innerHTML = riskText(p) + ', ' + verdict(p.r) + '.' + (unsure ? ' <b class="go">The missing values could change this decision: measure ' + names(p.missing) + '.</b>' : (p.missing.length ? ' The missing values would not change the decision.' : '')) +
    ' Panel price US $' + m.price_US.toFixed(2) + '.' + rangeNotes('f_', tests);
}
document.querySelectorAll('nav button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('nav button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
  document.getElementById('tab-seq').hidden = b.dataset.tab !== 'seq'; document.getElementById('tab-fix').hidden = b.dataset.tab !== 'fix';
}));
document.querySelectorAll('input, select').forEach(el => el.addEventListener('input', () => { updateSeq(); updateFix(); }));
function selfTest() {
  let n = 0, ok = 0;
  for (const [name, tests] of Object.entries(M.tests)) for (const t of tests) { n++; const r = risk(M.models[name], c => t.inputs[c] ?? null); if (r !== null && Math.abs(r - t.risk) < 1e-9) ok++; }
  for (const t of M.tests_missing) { n++; const p = predict(M.models[t.model], c => t.inputs[c] ?? null);
    if (p && Math.abs(p.r - t.risk) < 1e-7 && Math.abs(p.lo - t.low) < 1e-7 && Math.abs(p.hi - t.high) < 1e-7) ok++; }
  return [ok, n];
}
(function () { const [ok, n] = selfTest(); document.getElementById('selftest').textContent = 'Arithmetic self-check against the fitted models (complete and missing inputs): ' + ok + ' of ' + n + ' test cases match.'; })();
updateSeq(); updateFix();
</script></body></html>
