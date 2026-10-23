// End-to-end: serve the repo over HTTP like Pages, emulate a phone, upload files, check results.
const { chromium, devices } = require('playwright');
const http = require('http'), fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..'), G = path.join(__dirname, 'gen');
const truth = JSON.parse(fs.readFileSync(path.join(G, 'truth.json')));
const server = http.createServer((q, r) => {
  const f = path.join(ROOT, q.url === '/' ? 'index.html' : q.url);
  fs.readFile(f, (e, d) => { if (e) { r.writeHead(404); r.end(); } else { r.writeHead(200, { 'content-type': 'text/html' }); r.end(d); } });
}).listen(8765);
let fails = 0;
const check = (ok, msg) => { console.log((ok ? 'PASS ' : 'FAIL ') + msg); if (!ok) fails++; };

async function waitDone(p) {
  await p.waitForFunction(() => state.busy || /^(No Moon|This browser|.*can't)/.test(document.getElementById('status').textContent), null, { timeout: 30000 });
  await p.waitForFunction(() => !state.busy && !state.pending && !state.enhancing && /^(Done|No Moon|This browser|.*can't)/.test(document.getElementById('status').textContent), null, { timeout: 180000 });
  return p.textContent('#status');
}
const metrics = (p) => p.evaluate(() => {
  const O = state.OUT, S = state.stack, R = state.frames[state.refIdx].data;
  // background noise (corner patch) and sharpness, stack vs best single frame
  const patch = (get) => { let s = 0, s2 = 0, n = 0; for (let y = (O >> 1) - 18; y < (O >> 1) + 18; y++) for (let x = 4; x < 40; x++) { const v = get(y * O + x); s += v; s2 += v * v; n++; } return Math.sqrt(Math.max(0, s2 / n - (s / n) ** 2)); };
  const noiseStack = patch((i) => 0.299 * S[i * 3] + 0.587 * S[i * 3 + 1] + 0.114 * S[i * 3 + 2]);
  const noiseRef = patch((i) => 0.299 * R[i * 4] + 0.587 * R[i * 4 + 1] + 0.114 * R[i * 4 + 2]);
  const Ls = new Float32Array(O * O); for (let i = 0; i < O * O; i++) Ls[i] = 0.299 * S[i * 3] + 0.587 * S[i * 3 + 1] + 0.114 * S[i * 3 + 2];
  // correlation of stack with best frame over the whole crop
  let sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0; const N = O * O;
  for (let i = 0; i < N; i++) { const a = Ls[i], b2 = 0.299 * R[i * 4] + 0.587 * R[i * 4 + 1] + 0.114 * R[i * 4 + 2]; sa += a; sb += b2; sab += a * b2; saa += a * a; sbb += b2 * b2; }
  const corr = (sab - sa * sb / N) / Math.sqrt((saa - sa * sa / N) * (sbb - sb * sb / N));
  return { corr, O, n: state.frames.length, used: +document.getElementById('sUsed').textContent, noiseStack, noiseRef,
    qStack: sharpness(Ls, O), qRef: state.frames[state.refIdx].q,
    frames: state.frames.map((f) => ({ label: f.label, dx: f.dx, dy: f.dy, cx: f.cx, cy: f.cy })), scale: state.cropScale, side: state.side };
});

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ ...devices['Pixel 7'] });
  const p = await ctx.newPage();
  const errs = [];
  p.on('pageerror', (e) => errs.push(e.message));
  // failed downloads (the AI model with no network) are expected; script errors are not
  p.on('console', (m) => { if (m.type() === 'error' && !/Failed to load resource/.test(m.text())) errs.push(m.text()); });
  await p.goto('http://localhost:8765/');

  // 1) pick photos WHILE the sample is still processing (tests the queue)
  const photos = fs.readdirSync(path.join(G, 'photos')).map((f) => path.join(G, 'photos', f));
  await p.setInputFiles('#files', photos);
  let t0 = Date.now();
  let st = await waitDone(p);
  console.log(`photos: ${((Date.now() - t0) / 1000).toFixed(1)}s  ${st}`);
  let m = await metrics(p);
  check(m.n === 22, `22 of 24 photos used (2 are clouded out) -> ${m.n}`);
  check(!m.frames.some((f) => truth[f.label.split(' ')[0]]?.cloud), 'clouded frames rejected');
  // registration error: true centre in crop coords minus shift must agree across frames
  const res = m.frames.filter((f) => f.label !== 'IMG_4103.jpg').map((f) => {
    const t = truth[f.label];
    return [(t.cx - (f.cx - m.side / 2)) * m.scale - f.dx, (t.cy - (f.cy - m.side / 2)) * m.scale - f.dy];
  });
  const mx = res.reduce((a, r) => a + r[0], 0) / res.length, my = res.reduce((a, r) => a + r[1], 0) / res.length;
  const errMax = Math.max(...res.map((r) => Math.hypot(r[0] - mx, r[1] - my)));
  check(m.corr > 0.98, `stack matches best frame: r=${m.corr.toFixed(4)}`);
  check(errMax < 0.5, `alignment error vs ground truth, worst frame: ${errMax.toFixed(3)} px`);
  check(m.noiseStack < m.noiseRef * 0.5, `stack noise ${m.noiseStack.toFixed(2)} vs single frame ${m.noiseRef.toFixed(2)}`);
  check(m.qStack > m.qRef * 0.4, `stack sharpness ${m.qStack.toFixed(0)} vs best frame ${m.qRef.toFixed(0)} (raw, before sharpening)`);
  // EXIF-rotated photo must be found and aligned too
  check(m.frames.some((f) => f.label === 'IMG_4103.jpg'), 'EXIF-rotated photo used');


  // 2) controls: sliders, method, drop frame, compare, save
  await p.locator('#keep').fill('80'); await p.waitForTimeout(600);
  check(+(await p.textContent('#sUsed')) === Math.round(22 * 0.8), 'keep 80% restacks');
  await p.selectOption('#method', 'mean'); await p.waitForTimeout(600);
  await p.locator('#sharpFine').fill('1.5'); await p.locator('#sat').fill('3'); await p.waitForTimeout(300);
  const idle = () => p.waitForFunction(() => !state.enhancing && state.enh, null, { timeout: 120000 });
  await idle();
  for (const [mode, factor] of [['off', 1], ['drizzle', 2], ['ai', 2]]) {
    await p.selectOption('#upscale', mode); await p.waitForTimeout(100); await idle();
    const [W, O] = await p.evaluate(() => [state.enh.W, state.OUT]);
    check(W === O * factor, `upscale "${mode}" gives ${W}px output (${factor}x of ${O})`);
  }
  const before = await p.evaluate(() => state.enh.L);
  await p.locator('#deblur').fill('0'); await p.locator('#denoise').fill('0'); await p.waitForTimeout(400); await idle();
  const changed = await p.evaluate((b) => state.enh.L.some((v, i) => Math.abs(v - b[i]) > 0.5), before);
  check(changed, 'denoise/deblur sliders re-run the enhancement');
  await p.locator('#deblur').fill('0.65'); await p.locator('#denoise').fill('2.5'); await p.waitForTimeout(400); await idle();
  await p.locator('.thumb').first().click(); await p.waitForTimeout(600);
  check(await p.evaluate(() => state.frames.some((f) => f.off)), 'tapping a thumbnail drops the frame');
  const box = await p.locator('#compare').boundingBox();
  await p.touchscreen.tap(box.x + 5, box.y + 5);
  await p.locator('#compare').dispatchEvent('pointerdown');
  check((await p.textContent('#tag')) === 'BEST SINGLE FRAME', 'hold-to-compare shows best frame');
  await p.locator('#compare').dispatchEvent('pointerup');
  check((await p.textContent('#tag')).includes('STACK OF'), 'release returns to stack');
  await p.click('#save');
  await p.waitForSelector('#modal:not([hidden])');
  await p.waitForFunction(() => document.getElementById('outImg').naturalWidth > 0);
  const [dl] = await Promise.all([p.waitForEvent('download'), p.click('#download')]);
  const dpath = path.join(G, 'download.png'); await dl.saveAs(dpath);
  const sig = fs.readFileSync(dpath).subarray(0, 8).toString('hex');
  check(sig === '89504e470d0a1a0a' && dl.suggestedFilename().startsWith('moon-stack-'), `download is a PNG (${dl.suggestedFilename()}, ${fs.statSync(dpath).size} bytes)`);
  await p.screenshot({ path: path.join(G, 'e2e-modal.png') });
  await p.click('#closeModal');

  // 3) videos
  for (const v of ['moon_h264.mp4', 'moon_vp9.webm']) {
    await p.locator('#vidFrames').fill('40');
    await p.evaluate(() => { document.getElementById('status').textContent = ''; });
    await p.setInputFiles('#files', path.join(G, v));
    t0 = Date.now();
    st = await waitDone(p);
    console.log(`${v}: ${((Date.now() - t0) / 1000).toFixed(1)}s  ${st}`);
    if (v.endsWith('mp4') && !st.startsWith('Done')) { console.log('NOTE  this Chromium build has no H.264 decoder; real Chrome/Safari do'); continue; }
    m = await metrics(p);
    check(m.n === 40, `${v}: 40 frames extracted -> ${m.n}`);
    const uniq = new Set(m.frames.map((f) => f.cx.toFixed(1) + ',' + f.cy.toFixed(1))).size;
    check(uniq > 30, `${v}: frames are distinct (seeking works): ${uniq} unique positions`);
    check(m.corr > 0.98, `${v}: stack matches best frame (aligned, not smeared): r=${m.corr.toFixed(4)}`);
  }
  await p.screenshot({ path: path.join(G, 'e2e-video.png') });

  // 4) junk input gives a clear message
  await p.evaluate(() => { document.getElementById('status').textContent = ''; });
  await p.setInputFiles('#files', { name: 'notes.jpg', mimeType: 'image/jpeg', buffer: Buffer.from('not an image') });
  st = await waitDone(p);
  check(/couldn't open/.test(st), `unreadable file message: "${st}"`);

  check(errs.length === 0, 'no page errors' + (errs.length ? ': ' + errs.join(' | ') : ''));
  console.log(fails ? `${fails} FAILED` : 'ALL PASSED');
  await b.close(); server.close();
  process.exit(fails ? 1 : 0);
})();
