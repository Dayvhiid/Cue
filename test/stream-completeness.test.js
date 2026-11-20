// Regression tests for "cue sometimes gives an incomplete answer, or none":
//  - a stream that stopped at the output cap was presented as complete
//  - a stream with no visible text resolved as success (blank bubble)
//  - reasoning-only chunks did not count as liveness, so the 25s inactivity
//    watchdog in main.js killed a model that was still working
const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');

let chunks = [];
let capturedRequest = null;
let requests = [];
const originalModuleLoad = Module._load;
Module._load = function loadWithOpenAIStub(request, parent, isMain) {
  if (request === 'openai') {
    return class FakeOpenAI {
      constructor() {
        this.chat = { completions: { create: (req) => {
          capturedRequest = req;
          requests.push(req);
          return Promise.resolve(Array.isArray(chunks[0]) ? chunks.shift() : chunks);
        } } };
      }
    };
  }
  return originalModuleLoad.call(this, request, parent, isMain);
};
const { createLLM, normalizeFinishReason, ANSWER_TOKEN_CAP, CEREBRAS_REASONING_HEADROOM, REASONING_RETRY_HEADROOM } = require('../src/llm');
test.after(() => { Module._load = originalModuleLoad; });
test.beforeEach(() => { requests = []; });

const content = (t) => ({ choices: [{ delta: { content: t } }] });
const reasoning = (t) => ({ choices: [{ delta: { reasoning_content: t } }] });
const finish = (r) => ({ choices: [{ delta: {}, finish_reason: r }] });
const settings = (over = {}) => ({ provider: 'openai', apiKeys: { openai: 'sk-test' }, models: { openai: { fast: 'gpt-x', smart: 'gpt-y' } }, smart: false, ...over });
const ask = (llm, extra = {}) => llm.stream({ system: 's', turns: [{ role: 'user', text: 'q' }], onToken: () => {}, ...extra });

test('an answer cut off at the cap is reported as truncated, and its text kept', async () => {
  chunks = [content('def solve():'), content('\n    return'), finish('length')];
  let truncated = 0;
  const out = await ask(createLLM(settings()), { onTruncated: () => { truncated++; } });
  assert.equal(out, 'def solve():\n    return');
  assert.equal(truncated, 1);
});

test('a normally finished answer is not reported as truncated', async () => {
  chunks = [content('Done.'), finish('stop')];
  let truncated = 0;
  await ask(createLLM(settings()), { onTruncated: () => { truncated++; } });
  assert.equal(truncated, 0);
});

test('an empty answer is an error, not a blank success', async () => {
  chunks = [finish('stop')];
  await assert.rejects(ask(createLLM(settings())), /empty answer/);
});

test('reasoning that eats the whole budget with no text says so', async () => {
  chunks = [reasoning('thinking…'), reasoning('more…'), finish('length')];
  await assert.rejects(ask(createLLM(settings())), /ran out of room/);
});

test('a reasoning-only cap hit is retried once with more room, then answers', async () => {
  // Two streams in sequence: the first thinks until the cap, the retry answers.
  chunks = [[reasoning('thinking…'), finish('length')], [content('Answer'), finish('stop')]];
  let truncated = 0;
  const out = await ask(createLLM(settings()), { onTruncated: () => { truncated++; } });
  assert.equal(out, 'Answer');
  assert.equal(truncated, 0);
  assert.equal(requests.length, 2);
  assert.equal(requests[1].max_tokens, ANSWER_TOKEN_CAP.fast + REASONING_RETRY_HEADROOM);
});

test('Cerebras requests leave headroom for its reasoning model', async () => {
  const cerebras = (smart) => ({ provider: 'cerebras', apiKeys: { cerebras: 'csk-test' }, models: { cerebras: { fast: 'gpt-oss-120b', smart: 'gpt-oss-120b' } }, smart });
  chunks = [content('ok'), finish('stop')];
  await ask(createLLM(cerebras(false)));
  assert.equal(capturedRequest.max_tokens, ANSWER_TOKEN_CAP.fast + CEREBRAS_REASONING_HEADROOM.fast);
  await ask(createLLM(cerebras(true)));
  assert.equal(capturedRequest.max_tokens, ANSWER_TOKEN_CAP.smart + CEREBRAS_REASONING_HEADROOM.smart);
});

test('a content-filtered empty answer says it was filtered', async () => {
  chunks = [finish('content_filter')];
  await assert.rejects(ask(createLLM(settings())), /content filter/);
});

test('reasoning-only chunks count as liveness for the inactivity watchdog', async () => {
  chunks = [reasoning('a'), reasoning('b'), reasoning('c'), content('Answer'), finish('stop')];
  let activity = 0;
  await ask(createLLM(settings()), { onActivity: () => { activity++; } });
  assert.equal(activity, chunks.length);
});

test('the output cap leaves room for full coding answers', async () => {
  chunks = [content('ok'), finish('stop')];
  await ask(createLLM(settings()));
  assert.equal(capturedRequest.max_tokens, ANSWER_TOKEN_CAP.fast);
  assert.ok(ANSWER_TOKEN_CAP.fast >= 2000 && ANSWER_TOKEN_CAP.smart >= 4000);
  await ask(createLLM(settings({ smart: true })));
  assert.equal(capturedRequest.max_tokens, ANSWER_TOKEN_CAP.smart);
});

test('finish reasons from every provider normalize to one vocabulary', () => {
  assert.equal(normalizeFinishReason('length'), 'length');      // OpenAI-compatible
  assert.equal(normalizeFinishReason('max_tokens'), 'length');  // Anthropic
  assert.equal(normalizeFinishReason('MAX_TOKENS'), 'length');  // Gemini
  assert.equal(normalizeFinishReason('SAFETY'), 'filtered');
  assert.equal(normalizeFinishReason('content_filter'), 'filtered');
  assert.equal(normalizeFinishReason('stop'), 'stop');
  assert.equal(normalizeFinishReason(undefined), null);
});
