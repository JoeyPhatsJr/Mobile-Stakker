// Quality gate for the enhancement pipeline (AI upscale + denoise + deblur), scored against the
// ground-truth Moon that tests/make_test_data.py rendered the test photos from.
//
// Fails if the app's default output, compared with the plain stack, has less real fine detail,
// a softer limb, a halo around the limb, more sky noise, distorted broad brightness, or fine
// patterning (hatching, checkerboard) in areas where the real Moon is smooth.
//
// AI upscaling needs the model packages locally (they're fetched from jsdelivr in production):
//   npm i @tensorflow/tfjs@4.22.0 @upscalerjs/esrgan-slim@1.0.0
// Without them the app falls back to its classic upscaler and the same gates apply.
const { chromium } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..'), G = path.join(__dirname, 'gen');
const server = http.createServer((q, r) => {
  const f = q.url === '/truth.png' ? path.join(G, 'truth.png') : path.join(ROOT, q.url === '/' ? 'index.html' : q.url);
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200); r.end(d); } });
}).listen(8767);
const resolvePkg = (name) => [...module.paths, ...(process.env.NODE_PATH || '').split(path.delimiter)].map((d) => path.join(d, name)).find((d) => d && fs.existsSync(path.join(d, 'package.json'))) || null;
const PKGS = { '@tensorflow/tfjs': resolvePkg('@tensorflow/tfjs'), '@upscalerjs/esrgan-slim': resolvePkg('@upscalerjs/esrgan-slim') };
let fails = 0;
const check = (ok, msg) => { console.log((ok ? 'PASS ' : 'FAIL ') + msg); if (!ok) fails++; };

const SETS = [
  { name: '5x phone zoom (Moon ~130 px, undersampled)', dir: 'photos_small', truth: 'truth_small.json', minFineGain: 0.03 },
  { name: 'binoculars/telescope (Moon ~400 px, blur-limited)', dir: 'photos', truth: 'truth.json', minFineGain: 0.04 },
];

async function scorePage(p, truth) {
  return p.evaluate(async (truth) => {
    const img = new Image(); img.src = '/truth.png'; await img.decode();
    const O = state.OUT, W = state.enh.W, f = W / O, ref = state.frames[state.refIdx], t = truth[ref.label];
    let cx = f * (t.cx - (ref.cx - state.side / 2)) * state.cropScale, cy = f * (t.cy - (ref.cy - state.side / 2)) * state.cropScale, R = f * ((t.diam || 400) * state.cropScale) / 2;
    const tc = makeCanvas(W, W), tg = tc.getContext('2d', { willReadFrequently: true });
    const drawT = (cx, cy, R) => { tg.fillStyle = '#000'; tg.fillRect(0, 0, W, W); tg.imageSmoothingQuality = 'high'; tg.drawImage(img, cx - R, cy - R, 2 * R, 2 * R); return lumOf(tg.getImageData(0, 0, W, W).data, W * W); };
    const plain = f === 2 ? lumRGB(cubic2(state.stack, O, 3), W * W) : lumRGB(state.stack, O * O);
    // register the truth to the plain stack (position and size), least squares with gain/offset
    const Tb = new Float32Array(W * W);
    const cost = (cx, cy, R) => {
      gaussBlur(drawT(cx, cy, R), Tb, W, W, 1.5 * f);
      let n = 0, st = 0, sl = 0, stt = 0, stl = 0, sll = 0;
      for (let y = 0; y < W; y += 2) for (let x = 0; x < W; x += 2) { if (Math.hypot(x - cx, y - cy) > R + 10 * f) continue; const p = y * W + x, a = Tb[p], l = plain[p]; n++; st += a; sl += l; stt += a * a; stl += a * l; sll += l * l; }
      const a = (stl - st * sl / n) / (stt - st * st / n), b = (sl - a * st) / n;
      return (sll - 2 * a * stl - 2 * b * sl + a * a * stt + 2 * a * b * st + n * b * b) / n;
    };
    let best = cost(cx, cy, R);
    for (const step of [2, 1, 0.5, 0.25]) for (let moved = true; moved;) {
      moved = false;
      for (const [dx, dy, dr] of [[step, 0, 0], [-step, 0, 0], [0, step, 0], [0, -step, 0], [0, 0, step], [0, 0, -step]]) {
        const c = cost(cx + dx, cy + dy, R + dr); if (c < best) { best = c; cx += dx; cy += dy; R += dr; moved = true; }
      }
    }
    const T = drawT(cx, cy, R);
    const band = (A, s1, s2) => { const a = new Float32Array(W * W), b = new Float32Array(W * W); gaussBlur(A, a, W, W, s1); gaussBlur(A, b, W, W, s2); for (let i = 0; i < a.length; i++) a[i] -= b[i]; return a; };
    const inner = [], insideFit = [], limbBand = [], ring = [], skyP = [];
    for (let y = 0; y < W; y++) for (let x = 0; x < W; x++) {
      const r = Math.hypot(x + 0.5 - cx, y + 0.5 - cy), p = y * W + x;
      if (r < R - 12) inner.push(p);
      if (r < R - 6) insideFit.push(p);
      else if (r < R + 8) limbBand.push(p);
      if (r > R + 2 && r < R + 14) { const q = Math.round(cy + (y - cy) * (R - 10) / r) * W + Math.round(cx + (x - cx) * (R - 10) / r); if (T[q] > 60) ring.push(p); }
      if (r > R + 14 && x > 2 && y > 2 && x < W - 3 && y < W - 3 && plain[p] > 0.5) skyP.push(p);
    }
    const corr = (A, B) => { let sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0; const n = inner.length; for (const p of inner) { const a = A[p], b = B[p]; sa += a; sb += b; sab += a * b; saa += a * a; sbb += b * b; } return (sab - sa * sb / n) / Math.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n)); };
    const Tfine = band(T, 0.6, 2), Tmid = band(T, 2, 5);
    // smooth areas of the real Moon: anything 2-pixel-scale there is noise or an artifact
    const flat = [];
    { const m = new Float32Array(W * W), m2 = new Float32Array(W * W), sq = T.map((v) => v * v); boxBlur(T, m, W, W, 3); boxBlur(sq, m2, W, W, 3); for (const p of inner) if (m2[p] - m[p] * m[p] < 4) flat.push(p); }
    const score = (L) => {
      let n = 0, st = 0, sl = 0, stt = 0, stl = 0;
      for (const p of insideFit) if (T[p] > 30) { n++; st += T[p]; sl += L[p]; stt += T[p] * T[p]; stl += T[p] * L[p]; }
      const a = (stl - st * sl / n) / (stt - st * st / n), b = (sl - a * st) / n;
      let skyM = 0; skyP.forEach((p) => (skyM += L[p])); skyM /= skyP.length;
      let tm = 0; insideFit.forEach((p) => (tm += T[p])); tm /= insideFit.length;
      const tsd = Math.sqrt(insideFit.reduce((s, p) => s + (T[p] - tm) ** 2, 0) / insideFit.length);
      const rms = (P, fn) => Math.sqrt(P.reduce((s, p) => s + fn(p) ** 2, 0) / P.length);
      let halo = 0; ring.forEach((p) => (halo += L[p] - skyM)); halo /= ring.length * a;
      return {
        fine: corr(band(L, 0.6, 2), Tfine), mid: corr(band(L, 2, 5), Tmid),
        tex: rms(insideFit, (p) => L[p] - (a * T[p] + b)) / (a * tsd),
        edge: rms(limbBand, (p) => L[p] - (T[p] > 1 ? a * T[p] + b : skyM)) / (a * tsd),
        pattern: Math.sqrt(flat.reduce((e, p) => e + (L[p] - 0.25 * (L[p - 1] + L[p + 1] + L[p - W] + L[p + W])) ** 2, 0) / flat.length) / a,
        halo, sky: Math.sqrt(skyP.reduce((s, p) => s + (L[p] - skyM) ** 2, 0) / skyP.length) / a,
      };
    };
    return { plain: score(plain), enhanced: score(state.enh.L), upLabel: state.upLabel, upNote: document.getElementById('upNote').textContent, psf: state.limb && state.limb.sigma, guard: state.guard, W };
  }, truth);
}

(async () => {
  const b = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
  const p = await b.newPage({ viewport: { width: 1000, height: 900 } });
  p.on('pageerror', (e) => { console.log('pageerror', e.message); fails++; });
  if (PKGS['@tensorflow/tfjs'] && PKGS['@upscalerjs/esrgan-slim']) {
    await p.route('https://cdn.jsdelivr.net/npm/**', (route) => {
      const u = new URL(route.request().url()).pathname.replace(/^\/npm\//, '');
      const m = u.match(/^(@[^/]+\/[^@/]+|[^@/]+)@[^/]+\/(.*)$/);
      route.fulfill({ path: path.join(PKGS[m[1]], m[2]) });
    });
  } else console.log('NOTE  AI model packages not installed; testing the classic fallback');
  await p.goto('http://localhost:8767/');
  await p.waitForFunction(() => state.enh && !state.enhancing && !state.busy, null, { timeout: 180000 });
  for (const set of SETS) {
    const truth = JSON.parse(fs.readFileSync(path.join(G, set.truth)));
    await p.evaluate(() => { state.doneMsg = ''; });
    const t0 = Date.now();
    await p.setInputFiles('#files', fs.readdirSync(path.join(G, set.dir)).map((f) => path.join(G, set.dir, f)));
    await p.waitForFunction(() => /^Done/.test(state.doneMsg || '') && state.enh && !state.enhancing && !state.busy, null, { timeout: 600000 });
    const r = await scorePage(p, truth);
    const P = r.plain, E = r.enhanced, f3 = (v) => v.toFixed(3);
    console.log(`\n${set.name}: ${((Date.now() - t0) / 1000).toFixed(1)}s, upscale "${r.upLabel}" (${r.upNote}), output ${r.W}px, blur measured ${r.psf.toFixed(2)}px ${r.guard}`);
    console.log(`      plain stack: fine ${f3(P.fine)} mid ${f3(P.mid)} limb ${f3(P.edge)} halo ${P.halo.toFixed(2)} sky noise ${f3(P.sky)} texture ${f3(P.tex)} flat-area pattern ${f3(P.pattern)}`);
    console.log(`      enhanced:    fine ${f3(E.fine)} mid ${f3(E.mid)} limb ${f3(E.edge)} halo ${E.halo.toFixed(2)} sky noise ${f3(E.sky)} texture ${f3(E.tex)} flat-area pattern ${f3(E.pattern)}`);
    check(E.fine >= P.fine + set.minFineGain, `more real fine detail (correlation with the true Moon ${f3(P.fine)} -> ${f3(E.fine)})`);
    check(E.mid >= P.mid, `mid-scale detail not worse (${f3(P.mid)} -> ${f3(E.mid)})`);
    check(E.edge <= P.edge * 0.8, `limb at least 20% closer to the truth (${f3(P.edge)} -> ${f3(E.edge)})`);
    check(Math.abs(E.halo) <= 0.5, `no halo or dark ring at the limb (${E.halo.toFixed(2)} grey levels)`);
    check(E.sky <= P.sky, `sky noise not amplified (${f3(P.sky)} -> ${f3(E.sky)})`);
    check(E.pattern <= P.pattern * 1.2, `no hatching/checkerboard in smooth areas (${f3(P.pattern)} -> ${f3(E.pattern)})`);
    check(E.tex <= P.tex * 1.1, `broad brightness not distorted (texture error ${f3(P.tex)} -> ${f3(E.tex)})`);
  }
  console.log(fails ? `\n${fails} FAILED` : '\nALL PASSED');
  await b.close(); server.close();
  process.exit(fails ? 1 : 0);
})();
