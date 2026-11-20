// Regression: Cerebras models are text-only and answer any request carrying a
// screenshot with "400 status code (no body)" — so Smart assist and every other
// screen feature failed outright on that provider (reproduced live against
// api.cerebras.ai with gpt-oss-120b and qwen-3.8-27b).
const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');

const requests = [];
const originalModuleLoad = Module._load;
Module._load = function loadWithOpenAIStub(request, parent, isMain) {
  if (request === 'openai') {
    return class FakeOpenAI {
      constructor() {
        this.chat = { completions: { create: (req) => {
          requests.push(req);
          const hasImage = req.messages.some((m) => Array.isArray(m.content) && m.content.some((p) => p.type === 'image_url'));
          if (hasImage) {
            const e = new Error('400 status code (no body)');
            e.status = 400;
            return Promise.reject(e);
          }
          return Promise.resolve([
            { choices: [{ delta: { content: 'answer' } }] },
            { choices: [{ delta: {}, finish_reason: 'stop' }] }
          ]);
        } } };
      }
    };
  }
  return originalModuleLoad.call(this, request, parent, isMain);
};
const { createLLM } = require('../src/llm');
test.after(() => { Module._load = originalModuleLoad; });

const settings = { provider: 'cerebras', apiKeys: { cerebras: 'csk-test' }, models: { cerebras: { fast: 'gpt-oss-120b', smart: 'gpt-oss-120b' } }, smart: false };
// Groq is not pre-marked text-only, so it exercises discovery via the 400.
const discover = { provider: 'groq', apiKeys: { groq: 'gsk-test' }, models: { groq: { fast: 'text-model', smart: 'text-model' } }, smart: false };
const ask = (onImageUnsupported) => createLLM(discover).stream({
  system: 's', turns: [{ role: 'user', text: 'what is on my screen?' }],
  imageDataUrl: 'data:image/jpeg;base64,AAAA', onToken: () => {}, onImageUnsupported
});

test('a text-only model still answers when a screenshot is attached', async () => {
  let notified = 0;
  requests.length = 0;
  const out = await ask(() => { notified++; });
  assert.equal(out, 'answer');
  assert.equal(notified, 1, 'the user is told the screen was not used');
  assert.equal(requests.length, 2, 'one rejected attempt with the image, one without');
});

test('once known text-only, the screenshot is not sent again', async () => {
  let notified = 0;
  requests.length = 0;
  const out = await ask(() => { notified++; });
  assert.equal(out, 'answer');
  assert.equal(requests.length, 1, 'no wasted rejected request');
  assert.equal(notified, 0);
});

// Regression: "when I ask it to check my screen it answers a totally different
// question" — the screenshot was silently dropped and the model answered from
// the transcript. A screen request must never be answered blind.


test('a screen request to a text-only model throws instead of answering blind', async () => {
  requests.length = 0;
  await assert.rejects(
    createLLM(settings).stream({ system: 's', turns: [{ role: 'user', text: 'what is on my screen?' }], imageDataUrl: 'data:image/jpeg;base64,AAAA', requireImage: true, onToken: () => {} }),
    (e) => e.code === 'IMAGE_UNSUPPORTED' && /can't read screenshots/.test(e.message)
  );
  assert.equal(requests.length, 0, 'Cerebras is known text-only, so nothing is sent');
});

test('Cerebras reports that it cannot read images', () => {
  assert.equal(createLLM(settings).supportsImages(), false);
  assert.equal(createLLM({ ...settings, provider: 'gemini', apiKeys: { gemini: 'g' } }).supportsImages(), true);
});
