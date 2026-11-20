// LLM factory — OpenAI, Anthropic, Gemini, and OpenAI-compatible APIs behind one streaming interface.
// stream({ system, turns:[{role,text}], imageDataUrl, maxTokens, onToken }) -> Promise<fullText>

const { createCompatibleClientOptions } = require('./openai-compatible');
const publik = require('./publik');

const CUSTOM_PROVIDER = 'custom';
const PUBLIK_PROVIDER = publik.PUBLIK_PROVIDER;
// gemini-2.0-flash was Google's default here until it was deprecated (Feb 2026)
// and fully retired (Mar 3 2026) — every request against it now 404s with a
// generic "exception parsing response" body. gemini-3.8-flash is the current
// Flash release (Aug 2026), so it is the single default used everywhere in
// this file and for Gemini transcription in stt.js / stt-streaming.js.
const CURRENT_GEMINI_DEFAULT = 'gemini-3.8-flash';
// Purpose-built speech-to-text model: no thinking tokens, returns nothing on
// silence, and answers with an `audioTranscription` part instead of `text`
// (see extractGeminiTranscript in stt.js). Falls back to
// CURRENT_GEMINI_DEFAULT if Google ever retires it.
const GEMINI_TRANSCRIBE_MODEL = 'gemini-3.5-transcribe';
// Streaming counterpart over the Live API (bidiGenerateContent): word-by-word
// interim hypotheses plus a final on each pause. Used by GeminiLiveSTT in
// stt-streaming.js; the batch model above is the fallback when it fails.
const GEMINI_TRANSCRIBE_LIVE_MODEL = 'gemini-3.5-transcribe-live';
// claude-3-5-haiku-latest / claude-3-5-sonnet-latest were retired by Anthropic
// (confirmed absent from GET https://api.anthropic.com/v1/models as of Sep 19
// 2026 — every claude-2.x and claude-3.x id 404s with not_found_error).
// Fast tier: Claude Haiku 5.5 (latency-sensitive overlay answers). Smart tier:
// Claude Opus 5.5. Bare ids — current Claude models take no date suffix.
const CURRENT_ANTHROPIC_DEFAULT_FAST = 'claude-haiku-5-5';
const CURRENT_ANTHROPIC_DEFAULT_SMART = 'claude-opus-5-5';
// Exact ids earlier releases shipped as the defaults. Store defaults are
// persisted into settings.json, so an install that never touched the model
// fields still carries these; treat them as "the default" and upgrade.
const PREVIOUS_ANTHROPIC_DEFAULTS = {
  fast: ['claude-haiku-4-5-20251001'],
  smart: ['claude-sonnet-4-5-20250929']
};
// Cerebras's shared tier only offers reasoning models. qwen-3.8-27b (the old
// default) has no way to cap its thinking and could spend the whole output
// budget before answering; gpt-oss-120b takes reasoning_effort, so cue keeps
// its thinking short. The old default is persisted in settings.json, so it is
// migrated at read time like PREVIOUS_ANTHROPIC_DEFAULTS.
const CURRENT_CEREBRAS_DEFAULT = 'gpt-oss-120b';
const PREVIOUS_CEREBRAS_DEFAULTS = ['qwen-3.8-27b'];
const DEFAULT_MODELS = {
  cerebras: CURRENT_CEREBRAS_DEFAULT,
  openai: 'gpt-4o-mini',
  anthropic: CURRENT_ANTHROPIC_DEFAULT_FAST,
  gemini: CURRENT_GEMINI_DEFAULT,
  ollama: 'llama3.2',
  groq: 'llama-3.1-8b-instant',
  minimax: 'MiniMax-M2.7',
  // deepseek-chat/deepseek-reasoner were retired 2026-07-24 and now 404; the
  // replacement aliases are deepseek-flash (non-thinking) and deepseek-v4-pro
  // (thinking), so those are the defaults used everywhere in this file.
  deepseek: 'deepseek-flash',
  azure: 'gpt-4o-mini',
  publik: publik.DEFAULT_MODELS.fast
};
const CEREBRAS_BASE_URL = 'https://api.cerebras.ai/v1';

// Gemini model ids that Google has since deprecated/retired (the 2.5 family
// went "no longer available to new users" in Sep 2026). A settings file saved
// before this fix can still have one of these persisted on disk, so
// resolveGeminiModel migrates them at read time rather than only fixing the
// default — otherwise an existing user would keep re-hitting the same 404
// forever.
const DEAD_GEMINI_MODEL_RE = /^gemini-(1\.0|1\.5|2\.0|2\.5)(?:-|$)/i;

// Single place that answers "which Gemini model should this request use?".
// Both the chat path (createLLM) and the transcription paths (src/stt.js,
// src/stt-streaming.js) go through here. STT used to skip this and hardcode
// the default instead, so a user who picked a working model in Settings still
// got 404s from a model they had never selected — the app reported a failure
// against a model id that appeared nowhere in their config.
function resolveGeminiModel(settings) {
  const s = settings || {};
  const tier = s.smart ? 'smart' : 'fast';
  const configured = ((s.models || {}).gemini || {})[tier];
  if (!configured || DEAD_GEMINI_MODEL_RE.test(configured)) return CURRENT_GEMINI_DEFAULT;
  return configured;
}

// Same self-heal, for Anthropic: matches every retired claude-2.x/claude-3.x
// id (including the "-latest" aliases), so a settings file saved back when
// claude-3-5-haiku-latest/claude-3-5-sonnet-latest were the shipped defaults
// (store.js's DEFAULTS.models.anthropic, before this fix) gets migrated to a
// live model on next read instead of permanently re-hitting the same 404.
const DEAD_ANTHROPIC_MODEL_RE = /^claude-(2(?:\.\d+)?(?:-|$)|3-)/i;

// Same story for DeepSeek's retired chat/reasoner aliases — map each to its
// closest current replacement rather than collapsing both to one default.
const DEAD_DEEPSEEK_MODEL_RE = /^deepseek-(chat|reasoner)$/i;
const CURRENT_DEEPSEEK_FAST_DEFAULT = 'deepseek-flash';
const CURRENT_DEEPSEEK_SMART_DEFAULT = 'deepseek-v4-pro';

const PROVIDER_LABELS = { azure: 'Azure AI Foundry', cerebras: 'Cerebras', openai: 'OpenAI', minimax: 'MiniMax', publik: publik.PROVIDER_LABEL, deepseek: 'DeepSeek' };

// DeepSeek is OpenAI-compatible and reuses the OpenAI screenshot/streaming path via baseURL.
const DEEPSEEK_BASE_URL = 'https://api.deepseek.com';

function normalizeProviderName(provider) {
  if (!provider) return 'provider';
  if (PROVIDER_LABELS[provider]) return PROVIDER_LABELS[provider];
  return provider.charAt(0).toUpperCase() + provider.slice(1);
}

// A 429 on its own says only "slow down" — it never says the account is out of
// credit. So quota exhaustion is recognised ONLY from an upstream body that
// actually claims it (OpenAI `insufficient_quota`, Gemini `RESOURCE_EXHAUSTED`,
// or plain "exceeded your current quota"/quota/billing wording); every other
// 429 is reported as the rate limit it is. Anthropic's API has no quota concept
// at all — every Anthropic 429 is a per-minute rate_limit_error whose envelope
// is {type:'error', error:{type:'rate_limit_error', message}} (SDKs expose the
// whole envelope as error.error, so the real type sits at error.error.error.type)
// — so under the old `status === 429` rule an Anthropic user could never avoid
// the false "free-tier quota exhausted" message.
function errorSignals(error) {
  const status = error && (error.status || error.statusCode || error.response?.status);
  const code = error && (error.code || error.error?.code);
  const upstreamType = error && (error.error?.error?.type || error.error?.type);
  const rawMessage = (error && (error.message || String(error))) || '';
  return { status, code, upstreamType, text: `${rawMessage} ${code || ''} ${upstreamType || ''}`.toLowerCase() };
}

// The two upstream codes that mean "this is a burst/per-minute limit", never
// "your account is empty".
function hasRateLimitSignal({ code, upstreamType }) {
  return code === 'rate_limit_exceeded' || upstreamType === 'rate_limit_error';
}

// Pulled out so both the LLM and STT error paths (llm.js and stt.js) agree on
// what counts as a quota failure instead of drifting independently.
function isQuotaError(error) {
  const signals = errorSignals(error);
  const { code, upstreamType, text } = signals;
  if (code === 'insufficient_quota' || code === 'RESOURCE_EXHAUSTED' ||
      upstreamType === 'insufficient_quota' || upstreamType === 'RESOURCE_EXHAUSTED') return true;
  if (hasRateLimitSignal(signals)) return false;
  return /insufficient_quota|resource_exhausted|exceeded your current quota|\bquota\b|\bbilling\b/i.test(text);
}

// Everything else that 429s (including a bare 429 with no body at all, which is
// what most providers send under load) is a rate limit, not exhaustion.
function isRateLimitError(error) {
  if (isQuotaError(error)) return false;
  const signals = errorSignals(error);
  const { status, code, text } = signals;
  return status === 429 || code === 429 || hasRateLimitSignal(signals) ||
    /\b429\b|too many requests|rate limit/i.test(text);
}

function isNotFoundError(error) {
  const status = error && (error.status || error.statusCode || error.response?.status);
  const code = error && (error.code || error.error?.code);
  const rawMessage = (error && (error.message || String(error))) || '';
  const text = `${rawMessage} ${status || ''} ${code || ''}`.toLowerCase();
  return status === 404 || code === 404 || /\b404\b|is not found for api version|model not found/i.test(text);
}

// Gemini 429 bodies often carry a google.rpc.RetryInfo detail like
// {"retryDelay":"38s"} inside the JSON error text. Not every quota error has
// one (OpenAI/Anthropic don't), so this is best-effort and returns null when
// absent instead of guessing a wait time.
function extractRetryDelaySeconds(rawMessage) {
  const match = /retryDelay"?\s*:\s*"?(\d+(?:\.\d+)?)\s*s/i.exec(String(rawMessage || ''));
  if (!match) return null;
  const seconds = Number(match[1]);
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

function formatRetryWait(seconds) {
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  const minutes = Math.round(seconds / 60);
  return `${minutes} minute${minutes === 1 ? '' : 's'}`;
}

// Structured errors for publik so main.js can offer a button, not just a sentence.
function publikError(message, action) {
  const e = new Error(message);
  e.action = action || null;
  return e;
}

function isConnectionError(error) {
  const name = error && error.name;
  const text = (error && (error.message || String(error))) || '';
  return name === 'APIConnectionError' || name === 'APIConnectionTimeoutError' ||
    /fetch failed|ECONNREFUSED|ENOTFOUND|ETIMEDOUT|ECONNRESET|network|connection error|aborted/i.test(text);
}

// publik gateway failures, mapped BEFORE isQuotaError: its regex matches
// "billing"/"quota" in message text and the generic 429 copy tells the user to
// "add billing to your publik API account", which is wrong for publik. Reads
// the openai SDK's APIError shape (status, error = the envelope's inner
// object, headers) and falls through (null) for anything not publik-specific.
function describePublikError(error, model) {
  const status = error && (error.status || error.statusCode || error.response?.status);
  const body = error && (error.error || error.response?.data?.error || null);
  const headers = error && error.headers;
  const retryAfter = headers ? (typeof headers.get === 'function' ? headers.get('retry-after') : (headers['retry-after'] || headers['Retry-After'])) : null;
  if (!status && !isConnectionError(error)) return null;
  const described = publik.describeGatewayError({ status, body, model, retryAfter });
  return described ? publikError(described.message, described.action) : null;
}

function formatProviderErrorMessage(error, provider, model) {
  const label = normalizeProviderName(provider);
  const rawMessage = (error && (error.message || String(error))) || '';

  if (provider === PUBLIK_PROVIDER) {
    const described = describePublikError(error, model);
    if (described) return described;
  }

  if (isQuotaError(error)) {
    const retrySeconds = extractRetryDelaySeconds(rawMessage);
    const waitHint = retrySeconds ? ` Wait about ${formatRetryWait(retrySeconds)}` : ' Wait a moment';
    return `${label} free-tier quota exhausted (429 Too Many Requests).${waitHint} and try again, or add billing to your ${label} account. You can also switch providers or models in Settings.`;
  }

  if (isRateLimitError(error)) {
    const retrySeconds = extractRetryDelaySeconds(rawMessage);
    const waitHint = retrySeconds ? ` Wait about ${formatRetryWait(retrySeconds)}` : ' Wait a moment';
    return `${label} is rate-limiting requests right now (429 Too Many Requests).${waitHint} and try again — this is a temporary per-minute/request limit, not your account running out of credit.`;
  }

  if (isNotFoundError(error)) {
    const modelHint = model ? ` "${model}"` : '';
    return `${label} model${modelHint} is unavailable (404) — it may have been renamed, retired by the provider, or misspelled. Open Settings and pick a current model for ${label} (or clear the field to use cue's default), then try again.`;
  }

  return rawMessage || 'Unknown LLM error.';
}

function sanitizeTurns(turns) {
  const valid = new Set(['user', 'assistant']);
  return (turns || []).filter(t => valid.has(t.role)).map(t => ({ role: t.role, text: String(t.text || '') }));
}

// MiniMax is OpenAI-compatible and exposes two regional gateways. MiniMax-M3
// accepts image input, so it reuses the OpenAI screenshot path via baseURL.
const MINIMAX_BASE_URLS = {
  global_en: 'https://api.minimax.io/v1',
  cn_zh: 'https://api.minimaxi.com/v1'
};

function stripDataUrl(dataUrl) {
  const m = /^data:(.+?);base64,(.*)$/s.exec(dataUrl || '');
  return m ? { mime: m[1], b64: m[2] } : null;
}

async function streamOpenAI({ apiKey, baseURL, model, system, turns, imageDataUrl, maxTokens, extraBody, onToken, onResponse, onActivity, onFinish }) {
  const OpenAI = require('openai');
  const client = new OpenAI(baseURL ? { apiKey, baseURL } : { apiKey });
  const messages = [{ role: 'system', content: system }];
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      messages.push({
        role: 'user', content: [
          { type: 'text', text: t.text },
          { type: 'image_url', image_url: { url: imageDataUrl } }
        ]
      });
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  });
  const pending = client.chat.completions.create({ model, messages, stream: true, max_tokens: maxTokens, ...(extraBody || {}) });
  let stream;
  if (typeof onResponse === 'function' && pending && typeof pending.withResponse === 'function') {
    // The gateway stamps x-publik-* headers at admission; hand the raw
    // Response to the caller so the balance line can move before settlement.
    const { data, response } = await pending.withResponse();
    try { onResponse(response); } catch { /* a display hook must never break the answer */ }
    stream = data;
  } else {
    stream = await pending;
  }
  return consumeOpenAIStream(stream, { onToken, onActivity, onFinish });
}

// Shared by every OpenAI-compatible stream. Reasoning models (DeepSeek's
// v4-pro, o-series/gpt-5 behind Custom, many gateway models) stream their
// chain of thought in delta.reasoning_content / delta.reasoning and no
// `content` for a long stretch: that is still liveness for the caller's
// inactivity watchdog, never visible text. finish_reason is reported so a
// cap-truncated answer is not passed off as a complete one.
async function consumeOpenAIStream(stream, { onToken, onActivity, onFinish }) {
  let full = '';
  for await (const part of stream) {
    if (onActivity) onActivity();
    const choice = part && part.choices && part.choices[0];
    if (!choice) continue;
    const d = choice.delta && choice.delta.content;
    if (d) { full += d; onToken(d); }
    if (choice.finish_reason && onFinish) onFinish(choice.finish_reason);
  }
  return full;
}

// Azure AI Foundry Models API (cognitiveservices.azure.com hosts) lives under
// {endpoint}/openai/v1 and authenticates with the `api-key` header.
function normalizeAzureBaseURL(raw) {
  let u = String(raw || '').trim().replace(/\/+$/, '').replace(/\/chat\/completions$/i, '');
  if (!u) return '';
  if (/cognitiveservices\.azure\.com/i.test(u) && !/\/openai\/v1$/i.test(u)) {
    u += '/openai/v1';
  }
  return u;
}

async function streamAzure({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken, onActivity, onFinish, endpoint }) {
  const url = normalizeAzureBaseURL(endpoint);
  if (!url) throw new Error('Missing Azure endpoint. Add your Azure AI Foundry or Azure OpenAI endpoint in Settings.');
  const messages = [{ role: 'system', content: system }];
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      messages.push({ role: 'user', content: [
        { type: 'text', text: t.text },
        { type: 'image_url', image_url: { url: imageDataUrl } }
      ] });
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  });
  const OpenAI = require('openai');
  let client;
  if (/openai\.azure\.com/i.test(url)) {
    client = new OpenAI.AzureOpenAI({ endpoint: url.replace(/\/openai$/i, ''), apiKey, apiVersion: '2024-10-21' });
  } else {
    // Foundry / OpenAI-compatible base: force the `api-key` header and drop the
    // Authorization header the SDK adds by default (those hosts don't take a Bearer key).
    const azureFetch = async (input, init) => {
      const headers = new Headers(init && init.headers);
      headers.set('api-key', apiKey);
      headers.delete('authorization');
      return fetch(input, { ...init, headers });
    };
    client = new OpenAI({ baseURL: url, apiKey, fetch: azureFetch });
  }
  const stream = await client.chat.completions.create({ model, messages, stream: true, max_completion_tokens: maxTokens });
  return consumeOpenAIStream(stream, { onToken, onActivity, onFinish });
}

// Current Claude families think by default (adaptive thinking), and thinking
// tokens count toward max_tokens: a 700-token cap sized for a short visible
// answer can be eaten by thinking, ending with stop_reason "max_tokens" and no
// text. These models get headroom plus an explicit effort level (the lever
// Anthropic recommends over disabling thinking; Opus 5.5 rejects "disabled").
// Older ids (4.x) get the request unchanged — they reject `effort`.
const ANTHROPIC_ADAPTIVE_MODEL_RE = /^claude-(haiku-5-5|sonnet-5|sonnet-5-5|opus-5|opus-5-5|fable-5|fable-5-1)$/;
// Models that accept server-side refusal fallbacks in the "default" form.
const ANTHROPIC_FALLBACK_MODEL_RE = /^claude-(opus-5|opus-5-5|sonnet-5-5|fable-5-1)$/;
const ANTHROPIC_FALLBACK_BETA = 'server-side-fallback-2026-07-01';
const ANTHROPIC_THINKING_HEADROOM = { fast: 2048, smart: 8192 };

/**
 * Model-specific request fields for one Anthropic call. Pure, so the shape is
 * unit-testable without the SDK. `thinking` is cue's Smart toggle.
 */
function anthropicRequestOptions({ model, maxTokens, thinking }) {
  if (!ANTHROPIC_ADAPTIVE_MODEL_RE.test(model || '')) return { maxTokens, body: {}, headers: null };
  const tier = thinking ? 'smart' : 'fast';
  const body = {
    // "summarized" rather than the default "omitted": with omitted, nothing at
    // all is streamed while the model thinks, and a long think trips the
    // caller's inactivity watchdog. The summaries are never shown.
    thinking: { type: 'adaptive', display: 'summarized' },
    output_config: { effort: thinking ? 'medium' : 'low' }
  };
  let headers = null;
  if (ANTHROPIC_FALLBACK_MODEL_RE.test(model)) {
    // On a safety-classifier decline the API re-runs the request on a
    // fallback model inside the same stream instead of returning nothing.
    body.fallbacks = 'default';
    headers = { 'anthropic-beta': ANTHROPIC_FALLBACK_BETA };
  }
  return { maxTokens: maxTokens + ANTHROPIC_THINKING_HEADROOM[tier], body, headers };
}

async function streamAnthropic({ apiKey, model, system, turns, imageDataUrl, maxTokens, thinking, onToken, onActivity, onFinish }) {
  const Anthropic = require('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey });
  const shaped = anthropicRequestOptions({ model, maxTokens, thinking });
  const messages = turns.map((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      const content = [];
      if (img) content.push({ type: 'image', source: { type: 'base64', media_type: img.mime, data: img.b64 } });
      content.push({ type: 'text', text: t.text });
      return { role: 'user', content };
    }
    return { role: t.role, content: t.text };
  });
  const stream = await client.messages.create(
    { model, max_tokens: shaped.maxTokens, system, messages, stream: true, ...shaped.body },
    shaped.headers ? { headers: shaped.headers } : undefined
  );
  let full = '';
  let stopReason = null;
  for await (const ev of stream) {
    // Thinking streams no visible text (its display is "omitted" by default),
    // so report every event as liveness — otherwise the caller's inactivity
    // watchdog can fire while the model is still working.
    if (onActivity) onActivity();
    if (ev.type === 'content_block_delta' && ev.delta && ev.delta.type === 'text_delta') { full += ev.delta.text; onToken(ev.delta.text); }
    if (ev.type === 'message_delta' && ev.delta && ev.delta.stop_reason) stopReason = ev.delta.stop_reason;
  }
  // A refusal is an HTTP 200, not an exception; any partial text is not a
  // complete answer and must not be presented as one.
  if (stopReason === 'refusal') throw new Error('Claude declined to answer this request. Try rephrasing it.');
  if (stopReason && onFinish) onFinish(stopReason);
  return full;
}

// Gemini 3.x counts its internal "thinking" tokens against maxOutputTokens, so
// a 700-token cap was mostly eaten by reasoning and the visible answer came
// back truncated mid-sentence (measured: 3.8-flash spends ~390 thinking tokens
// on a short notes prompt by default). Fast tier: thinkingLevel "low", which
// on flash means no thinking at all (0 thought tokens, ~3x faster). Smart tier:
// leave the model's default reasoning alone. Either way, give the cap headroom
// for thoughts so the visible budget is what maxTokens says.
// (thinkingBudget: 0 is rejected by pro models and "minimal" by flash, so
// "low" is the one setting that works across the family.)
const GEMINI_THINKING_HEADROOM = { fast: 1024, smart: 4096 };
function geminiGenerationConfig({ system, maxTokens, thinking }) {
  const config = {
    systemInstruction: system,
    maxOutputTokens: maxTokens + (thinking ? GEMINI_THINKING_HEADROOM.smart : GEMINI_THINKING_HEADROOM.fast)
  };
  if (!thinking) config.thinkingConfig = { thinkingLevel: 'low' };
  return config;
}

async function streamGemini({ apiKey, model, system, turns, imageDataUrl, maxTokens, thinking, onToken, onActivity, onFinish }) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  const contents = turns.map((t, i) => {
    const last = i === turns.length - 1;
    const parts = [{ text: t.text }];
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      if (img) parts.push({ inlineData: { mimeType: img.mime, data: img.b64 } });
    }
    return { role: t.role === 'assistant' ? 'model' : 'user', parts };
  });
  const config = geminiGenerationConfig({ system, maxTokens, thinking });
  let stream;
  try {
    stream = await ai.models.generateContentStream({ model, contents, config });
  } catch (e) {
    // A model that predates thinkingLevel (or a custom id that rejects it)
    // should still answer: retry once without the thinking setting.
    if (!config.thinkingConfig || !/thinking/i.test((e && e.message) || '')) throw e;
    delete config.thinkingConfig;
    stream = await ai.models.generateContentStream({ model, contents, config });
  }
  let full = '';
  for await (const chunk of stream) {
    if (onActivity) onActivity(); // thought-only chunks carry no text but are liveness
    const t = chunk && chunk.text;
    if (t) { full += t; onToken(t); }
    const reason = chunk && chunk.candidates && chunk.candidates[0] && chunk.candidates[0].finishReason;
    if (reason && onFinish) onFinish(reason);
  }
  return full;
}

async function streamOllama({ apiKey, model, system, turns, imageDataUrl, maxTokens, onToken, onActivity, onFinish }) {
  const baseUrl = apiKey || 'http://localhost:11434';
  const url = `${baseUrl.replace(/\/$/, '')}/api/chat`;

  const messages = [{ role: 'system', content: system }];
  turns.forEach((t, i) => {
    const last = i === turns.length - 1;
    if (last && imageDataUrl && t.role === 'user') {
      const img = stripDataUrl(imageDataUrl);
      if (img) {
        messages.push({ role: 'user', content: t.text, images: [img.b64] });
      } else {
        messages.push({ role: 'user', content: t.text });
      }
    } else {
      messages.push({ role: t.role, content: t.text });
    }
  });

  let response;
  try {
    response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, messages, stream: true })
    });
  } catch (err) {
    throw new Error(`Ollama fetch failed: ${err.message}. Is Ollama running at ${baseUrl}?`);
  }

  if (!response.ok) {
    throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
  }

  const decoder = new TextDecoder();
  let full = '';
  let buffer = '';
  const note = (data) => { if (data && data.done && data.done_reason && onFinish) onFinish(data.done_reason); };
  for await (const chunk of response.body) {
    if (onActivity) onActivity();
    buffer += decoder.decode(chunk, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep incomplete line
    for (const line of lines) {
      if (!line.trim()) continue;
      try {
        const data = JSON.parse(line);
        if (data.message && data.message.content) {
          full += data.message.content;
          onToken(data.message.content);
        }
        note(data);
      } catch (e) {
        // ignore
      }
    }
  }
  if (buffer.trim()) {
    try {
      const data = JSON.parse(buffer);
      if (data.message && data.message.content) {
        full += data.message.content;
        onToken(data.message.content);
      }
      note(data);
    } catch (e) { }
  }
  return full;
}

function createLLM(settings) {
  const provider = settings.provider;
  const keys = settings.apiKeys || {};
  let apiKey = keys[provider];
  let baseURL = '';
  let configurationError = '';
  const tier = settings.smart ? 'smart' : 'fast';
  const models = settings.models || {};
  let model = (models[provider] || {})[tier];
  if (provider === 'gemini') {
    model = resolveGeminiModel(settings);
  }
  if (provider === PUBLIK_PROVIDER && !model) model = publik.DEFAULT_MODELS[tier];
  if (provider === 'anthropic' && (DEAD_ANTHROPIC_MODEL_RE.test(model || '') || PREVIOUS_ANTHROPIC_DEFAULTS[tier].includes(model))) {
    model = tier === 'smart' ? CURRENT_ANTHROPIC_DEFAULT_SMART : CURRENT_ANTHROPIC_DEFAULT_FAST;
  }
  if (provider === 'deepseek' && DEAD_DEEPSEEK_MODEL_RE.test(model || '')) {
    model = /reasoner/i.test(model) ? CURRENT_DEEPSEEK_SMART_DEFAULT : CURRENT_DEEPSEEK_FAST_DEFAULT;
  }
  if (provider === 'cerebras' && PREVIOUS_CEREBRAS_DEFAULTS.includes(model)) model = CURRENT_CEREBRAS_DEFAULT;
  if (!model) model = DEFAULT_MODELS[provider] || '';
  const minimaxRegion = settings.minimaxRegion || 'global_en';
  const endpoint = settings.azureEndpoint || '';

  if (provider === PUBLIK_PROVIDER) {
    // The packaged-build default: an OpenAI-compatible endpoint with a fixed
    // base URL (the provisioning response's, else the build's) and the key the
    // install minted. Never settings.baseUrl — that is the user's Custom slot.
    const base = (settings.publik && settings.publik.baseUrl) || publik.DEFAULT_BASE_URL;
    try {
      baseURL = createCompatibleClientOptions(apiKey, base).baseURL;
    } catch (error) {
      configurationError = error.message;
    }
    if (!apiKey) configurationError = `${publik.PROVIDER_LABEL} is not set up on this computer yet.`;
  } else if (provider === CUSTOM_PROVIDER) {
    try {
      const clientOptions = createCompatibleClientOptions(apiKey, settings.baseUrl);
      apiKey = clientOptions.apiKey;
      baseURL = clientOptions.baseURL;
    } catch (error) {
      configurationError = error.message;
    }
    if (!model && !configurationError) {
      configurationError = 'Set a Fast or Smart model for the Custom provider.';
    }
  } else if (provider !== 'ollama' && !apiKey) {
    // Ollama is a local server: the field holds a URL, and no key is required.
    configurationError = `Add your ${provider} API key in Settings.`;
  }

  // Azure needs a second credential: the resource endpoint.
  if (!configurationError && provider === 'azure' && !endpoint) {
    configurationError = 'Add your Azure AI Foundry endpoint in Settings.';
  }

  const ready = !configurationError && !!model;
  // The visible-answer cap. Prompts already ask for concise output, so this is
  // a ceiling, not a target, but at the old 700/1400 a LeetCode answer
  // (approach + full solution + complexity) routinely hit it mid-code-block.
  const maxTokens = settings.smart ? ANSWER_TOKEN_CAP.smart : ANSWER_TOKEN_CAP.fast;

  function supportsImages() {
    return !TEXT_ONLY_PROVIDERS.has(provider) && !TEXT_ONLY_MODELS.has(provider + ':' + model);
  }

  return {
    provider, model, apiKey, baseURL,
    ready,
    configurationError,
    supportsImages,
    async stream(params) {
      if (!ready) throw new Error(configurationError || `Complete the ${provider} provider settings.`);
      let finish = null;
      const args = {
        apiKey, baseURL, endpoint, model, maxTokens, thinking: !!settings.smart, ...params,
        turns: sanitizeTurns(params.turns),
        onFinish: (reason) => { finish = normalizeFinishReason(reason) || finish; }
      };
      // Text-only models (e.g. every Cerebras model) answer a request that
      // carries a screenshot with a bare "400 status code (no body)". Once a
      // model is known to be text-only, don't attach the image at all.
      // With `requireImage` (a "look at my screen" request) the screenshot is
      // the question: answering without it produces a confident answer to
      // something else, so throw IMAGE_UNSUPPORTED and let the caller pick a
      // model that can see, instead of silently dropping the image.
      const visionKey = provider + ':' + model;
      if (args.imageDataUrl && !supportsImages()) {
        if (params.requireImage) throw imageUnsupportedError(provider, model);
        args.imageDataUrl = null;
      }
      let text;
      try {
        text = await dispatch(args);
      } catch (error) {
        if (!args.imageDataUrl || error.status !== 400) throw error;
        // The screenshot is the most likely cause; ask once more without it.
        // If this also fails, the error is unrelated and surfaces as usual.
        if (params.requireImage) {
          TEXT_ONLY_MODELS.add(visionKey);
          throw imageUnsupportedError(provider, model);
        }
        text = await dispatch({ ...args, imageDataUrl: null });
        TEXT_ONLY_MODELS.add(visionKey);
        if (typeof params.onImageUnsupported === 'function') params.onImageUnsupported(normalizeProviderName(provider), model);
      }
      // A reasoning model that spent the whole cap thinking streamed no
      // visible text (so nothing reached onToken): retry once with room for
      // the reasoning instead of failing an answer the model was writing.
      if (!String(text || '').trim() && finish === 'length') {
        finish = null;
        text = await dispatch({ ...args, maxTokens: maxTokens + REASONING_RETRY_HEADROOM });
      }
      const label = normalizeProviderName(provider);
      // An empty answer used to resolve as success and leave a blank bubble.
      if (!String(text || '').trim()) {
        if (finish === 'length') throw new Error(`${label} ran out of room before writing an answer (model "${model}" spent its output budget reasoning). Try again, or pick a non-reasoning model in Settings.`);
        if (finish === 'filtered') throw new Error(`${label} blocked this response with its content filter. Try rephrasing the request.`);
        throw new Error(`${label} returned an empty answer. Please try again.`);
      }
      if (finish === 'length' && typeof params.onTruncated === 'function') params.onTruncated();
      return text;
    }
  };

  async function dispatch(args) {
      try {
        if (provider === 'openai') return await streamOpenAI(args);
        if (provider === CUSTOM_PROVIDER) return await streamOpenAI(args);
        if (provider === PUBLIK_PROVIDER) return await streamOpenAI(args);
        if (provider === 'ollama') return await streamOllama(args);
        if (provider === 'groq') return await streamOpenAI({ ...args, baseURL: 'https://api.groq.com/openai/v1' });
        // Every Cerebras shared model reasons before answering, and reasoning
        // tokens count against max_tokens: give the cap headroom so the
        // visible budget is still what maxTokens says, and keep gpt-oss's
        // thinking short on the Fast tier.
        if (provider === 'cerebras') return await streamOpenAI({
          ...args,
          maxTokens: args.maxTokens + CEREBRAS_REASONING_HEADROOM[tier],
          baseURL: CEREBRAS_BASE_URL,
          extraBody: /^gpt-oss/i.test(model) ? { reasoning_effort: tier === 'smart' ? 'medium' : 'low' } : undefined
        });
        if (provider === 'minimax') return await streamOpenAI({ ...args, baseURL: MINIMAX_BASE_URLS[minimaxRegion] || MINIMAX_BASE_URLS.global_en });
        if (provider === 'deepseek') return await streamOpenAI({ ...args, baseURL: DEEPSEEK_BASE_URL });
        if (provider === 'anthropic') return await streamAnthropic(args);
        if (provider === 'gemini') return await streamGemini(args);
        if (provider === 'azure') return await streamAzure(args);
        throw new Error('unknown provider: ' + provider);
      } catch (error) {
        // publik branches return an Error carrying `.action`; the string
        // branches keep working unchanged.
        const wrapped = formatProviderErrorMessage(error, provider, model);
        const out = wrapped instanceof Error ? wrapped : new Error(wrapped);
        // Keep the HTTP status so stream() can tell a rejected request apart.
        const status = error && (error.status || error.statusCode);
        if (status && out.status === undefined) out.status = status;
        throw out;
      }
  }
}

// provider:model pairs that rejected an attached screenshot this session.
const TEXT_ONLY_MODELS = new Set();
// Providers with no image input on any model (Cerebras: verified live with
// gpt-oss-120b and qwen-3.8-27b, both 400 on an image part).
const TEXT_ONLY_PROVIDERS = new Set(['cerebras']);

function imageUnsupportedError(provider, model) {
  const e = new Error(`${normalizeProviderName(provider)} model "${model}" can't read screenshots.`);
  e.code = 'IMAGE_UNSUPPORTED';
  return e;
}

const ANSWER_TOKEN_CAP = { fast: 2000, smart: 4000 };
const CEREBRAS_REASONING_HEADROOM = { fast: 4096, smart: 8192 };
// Extra room for the one retry after a reasoning-only, cap-truncated stream.
const REASONING_RETRY_HEADROOM = 12000;

// Each provider names "stopped at the cap" / "blocked" differently.
function normalizeFinishReason(reason) {
  const r = String(reason || '').toLowerCase();
  if (!r) return null;
  if (r === 'length' || r === 'max_tokens') return 'length';
  if (['content_filter', 'safety', 'prohibited_content', 'blocklist', 'spii', 'recitation'].includes(r)) return 'filtered';
  return 'stop';
}

module.exports = {
  createLLM,
  formatProviderErrorMessage,
  isQuotaError,
  isNotFoundError,
  geminiGenerationConfig,
  resolveGeminiModel,
  CURRENT_GEMINI_DEFAULT,
  GEMINI_TRANSCRIBE_MODEL,
  GEMINI_TRANSCRIBE_LIVE_MODEL,
  CURRENT_ANTHROPIC_DEFAULT_FAST,
  CURRENT_ANTHROPIC_DEFAULT_SMART,
  PREVIOUS_ANTHROPIC_DEFAULTS,
  anthropicRequestOptions,
  normalizeFinishReason,
  ANSWER_TOKEN_CAP,
  CEREBRAS_REASONING_HEADROOM,
  REASONING_RETRY_HEADROOM,
  PUBLIK_PROVIDER,
  isRateLimitError
};
