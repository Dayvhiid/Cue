const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');

// screen.js requires electron at load time; stub it the way llm.test.js stubs
// the openai SDK so the encoder can be exercised without a display.
const originalModuleLoad = Module._load;
Module._load = function loadWithElectronStub(request, parent, isMain) {
  if (request === 'electron') return { desktopCapturer: {}, screen: {} };
  return originalModuleLoad.call(this, request, parent, isMain);
};
const { encodeScreenshot, MAX_LONG_EDGE_PX, JPEG_QUALITY, MAX_DATA_URL_BYTES } = require('../src/screen');
Module._load = originalModuleLoad;

// A stand-in for Electron's NativeImage. The JPEG size is modelled as a fixed
// number of bytes per pixel so that the *only* thing that can bring a large
// capture under the cap is the resize step this test is guarding.
const FAKE_JPEG_BYTES_PER_PIXEL = 0.5;
function fakeImage(width, height, log = []) {
  return {
    getSize: () => ({ width, height }),
    isEmpty: () => false,
    resize(opts) {
      log.push({ resize: opts });
      const aspect = width / height;
      const w = opts.width != null ? opts.width : Math.round(opts.height * aspect);
      const h = opts.height != null ? opts.height : Math.round(opts.width / aspect);
      return fakeImage(w, h, log);
    },
    toJPEG(quality) {
      log.push({ toJPEG: quality });
      return Buffer.alloc(Math.round(width * height * FAKE_JPEG_BYTES_PER_PIXEL), 1);
    }
  };
}

test('a 4K capture is resized to the 1568 px long edge and encoded as JPEG q70 under 3 MB', () => {
  const log = [];
  const url = encodeScreenshot(fakeImage(3840, 2160, log));

  assert.match(url, /^data:image\/jpeg;base64,/);
  assert.deepEqual(log[0], { resize: { width: MAX_LONG_EDGE_PX } });
  assert.deepEqual(log[1], { toJPEG: JPEG_QUALITY });
  assert.ok(Buffer.byteLength(url) < MAX_DATA_URL_BYTES, `data URL is ${Buffer.byteLength(url)} bytes`);
  assert.equal(MAX_LONG_EDGE_PX, 1568);
  assert.equal(JPEG_QUALITY, 70);
});

test('the unresized 4K capture would NOT fit — proves the resize is load-bearing', () => {
  const raw = fakeImage(3840, 2160);
  const url = `data:image/jpeg;base64,${raw.toJPEG(70).toString('base64')}`;
  assert.ok(Buffer.byteLength(url) > MAX_DATA_URL_BYTES);
});

test('a portrait capture is resized by its height', () => {
  const log = [];
  encodeScreenshot(fakeImage(1200, 2600, log));
  assert.deepEqual(log[0], { resize: { height: MAX_LONG_EDGE_PX } });
});

test('a small capture is never upscaled', () => {
  const log = [];
  encodeScreenshot(fakeImage(1280, 800, log));
  assert.deepEqual(log, [{ toJPEG: JPEG_QUALITY }]);
});

// Regression: a capture that never settles used to hold runFeature's lock
// forever, so every later Smart assist click was silently dropped.
test('withTimeout rejects a capture that never settles', async () => {
  const { withTimeout } = requireScreen();
  const never = new Promise(() => {});
  await assert.rejects(withTimeout(never, 20, 'Screen capture timed out.'), /Screen capture timed out/);
});

test('withTimeout passes through a capture that settles in time', async () => {
  const { withTimeout } = requireScreen();
  assert.equal(await withTimeout(Promise.resolve('data:ok'), 1000), 'data:ok');
  await assert.rejects(withTimeout(Promise.reject(new Error('boom')), 1000), /boom/);
});

test('captureScreenshot is bounded by the capture timeout', async () => {
  const { captureScreenshot } = requireScreen({
    desktopCapturer: { getSources: () => new Promise(() => {}) },
    screen: { getPrimaryDisplay: () => ({ id: 1, size: { width: 100, height: 100 }, scaleFactor: 1 }) }
  });
  if (process.platform === 'darwin') return; // mac path shells out to screencapture first
  await assert.rejects(captureScreenshot({ timeoutMs: 20 }), /timed out/);
});

function requireScreen(electronStub = { desktopCapturer: {}, screen: {} }) {
  const p = require.resolve('../src/screen');
  delete require.cache[p];
  const load = Module._load;
  Module._load = function (request, parent, isMain) {
    if (request === 'electron') return electronStub;
    return load.call(this, request, parent, isMain);
  };
  try { return require('../src/screen'); } finally { Module._load = load; }
}
