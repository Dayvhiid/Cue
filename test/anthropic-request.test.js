const assert = require('node:assert/strict');
const test = require('node:test');
const Module = require('node:module');

// Stub @anthropic-ai/sdk so the real streamAnthropic path runs without a network.
let captured = null;      // { body, options }
let fakeEvents = [];
const originalModuleLoad = Module._load;
Module._load = function loadWithAnthropicStub(request, parent, isMain) {
  if (request === '@anthropic-ai/sdk') {
    return class FakeAnthropic {
      constructor() {
        this.messages = {
          create: async (body, options) => {
            captured = { body, options };
            const events = fakeEvents;
            return { async *[Symbol.asyncIterator]() { for (const ev of events) yield ev; } };
          }
        };
      }
    };
  }
  return originalModuleLoad.call(this, request, parent, isMain);
};
const {
  createLLM, anthropicRequestOptions,
  CURRENT_ANTHROPIC_DEFAULT_FAST, CURRENT_ANTHROPIC_DEFAULT_SMART, PREVIOUS_ANTHROPIC_DEFAULTS
} = require('../src/llm');
test.after(() => { Module._load = originalModuleLoad; });

const text = (t) => ({ type: 'content_block_delta', delta: { type: 'text_delta', text: t } });
const stop = (reason) => ({ type: 'message_delta', delta: { stop_reason: reason } });
const thinkingStart = { type: 'content_block_start', content_block: { type: 'thinking', thinking: '' } };

function anthropicSettings(over = {}) {
  return {
    provider: 'anthropic',
    apiKeys: { anthropic: 'sk-ant-test' },
    models: { anthropic: { fast: CURRENT_ANTHROPIC_DEFAULT_FAST, smart: CURRENT_ANTHROPIC_DEFAULT_SMART } },
    smart: false,
    ...over
  };
}

test('defaults are the current Haiku and Opus ids, with no date suffix', () => {
  assert.equal(CURRENT_ANTHROPIC_DEFAULT_FAST, 'claude-haiku-5-5');
  assert.equal(CURRENT_ANTHROPIC_DEFAULT_SMART, 'claude-opus-5-5');
});

test('a settings file still carrying the previously shipped defaults is upgraded', () => {
  const s = anthropicSettings({
    models: { anthropic: { fast: PREVIOUS_ANTHROPIC_DEFAULTS.fast[0], smart: PREVIOUS_ANTHROPIC_DEFAULTS.smart[0] } }
  });
  assert.equal(createLLM(s).model, 'claude-haiku-5-5');
  assert.equal(createLLM({ ...s, smart: true }).model, 'claude-opus-5-5');
});

test('a model the user chose themselves is left alone', () => {
  const s = anthropicSettings({ models: { anthropic: { fast: 'claude-sonnet-5-5', smart: 'claude-opus-5-5' } } });
  assert.equal(createLLM(s).model, 'claude-sonnet-5-5');
});

test('fast tier on Haiku 5.5: low effort, thinking headroom, no fallback beta', () => {
  const r = anthropicRequestOptions({ model: 'claude-haiku-5-5', maxTokens: 700, thinking: false });
  assert.deepEqual(r.body.output_config, { effort: 'low' });
  assert.equal(r.body.thinking.type, 'adaptive');
  assert.ok(r.maxTokens > 700, 'thinking counts toward max_tokens, so the cap needs headroom');
  assert.equal(r.headers, null);
});

test('smart tier on Opus 5.5: medium effort plus server-side refusal fallbacks', () => {
  const r = anthropicRequestOptions({ model: 'claude-opus-5-5', maxTokens: 1400, thinking: true });
  assert.equal(r.body.output_config.effort, 'medium');
  assert.equal(r.body.fallbacks, 'default');
  assert.equal(r.headers['anthropic-beta'], 'server-side-fallback-2026-07-01');
  // Opus 5.5 rejects disabled thinking; adaptive + summarized streams events while it thinks.
  assert.deepEqual(r.body.thinking, { type: 'adaptive', display: 'summarized' });
  assert.ok(!('temperature' in r.body));
});

test('older 4.x ids get the request unchanged (they reject effort)', () => {
  const r = anthropicRequestOptions({ model: 'claude-haiku-4-5', maxTokens: 700, thinking: false });
  assert.deepEqual(r, { maxTokens: 700, body: {}, headers: null });
});

test('streams text, sends the shaped body, and reports liveness on non-text events', async () => {
  fakeEvents = [thinkingStart, text('Hello'), text(' there'), stop('end_turn')];
  let activity = 0;
  const tokens = [];
  const llm = createLLM(anthropicSettings({ smart: true }));
  const out = await llm.stream({
    system: 'sys', turns: [{ role: 'user', text: 'hi' }],
    imageDataUrl: 'data:image/jpeg;base64,AAAA',
    onToken: (t) => tokens.push(t), onActivity: () => { activity++; }
  });
  assert.equal(out, 'Hello there');
  assert.deepEqual(tokens, ['Hello', ' there']);
  assert.equal(activity, fakeEvents.length);
  assert.equal(captured.body.model, 'claude-opus-5-5');
  assert.equal(captured.body.fallbacks, 'default');
  assert.equal(captured.options.headers['anthropic-beta'], 'server-side-fallback-2026-07-01');
  const content = captured.body.messages[0].content;
  assert.equal(content[0].type, 'image');
  assert.equal(content[0].source.media_type, 'image/jpeg');
});

test('a refusal surfaces as an error instead of a truncated answer', async () => {
  fakeEvents = [text('Partial'), stop('refusal')];
  const llm = createLLM(anthropicSettings());
  await assert.rejects(llm.stream({ system: 's', turns: [{ role: 'user', text: 'q' }], onToken: () => {} }), /declined/);
});

test('thinking that exhausts max_tokens with no text surfaces as an error', async () => {
  fakeEvents = [thinkingStart, stop('max_tokens')];
  const llm = createLLM(anthropicSettings());
  await assert.rejects(llm.stream({ system: 's', turns: [{ role: 'user', text: 'q' }], onToken: () => {} }), /ran out of room/);
});
