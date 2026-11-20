// Screen features on text-only models (every Cerebras model) read the screen
// with local OCR and send the text, instead of answering without the screen.
const assert = require('node:assert/strict');
const test = require('node:test');
const { cleanOcrText, withScreenText, isAvailable, recognizeText, MAX_OCR_CHARS, WINDOWS_OCR_SCRIPT } = require('../src/ocr');

test('OCR output is trimmed, blank lines dropped, and capped', () => {
  assert.equal(cleanOcrText('  Two Sum \r\n\r\n  def f():  \n'), 'Two Sum\ndef f():');
  const long = cleanOcrText('x'.repeat(MAX_OCR_CHARS + 500));
  assert.ok(long.length <= MAX_OCR_CHARS + 2 && long.endsWith('…'));
});

test('screen text is appended to the question and the system prompt explains it', () => {
  const { system, userText } = withScreenText('Look at the screenshot.', 'Question: what is this?', 'Two Sum\ndef twoSum(nums, target):');
  assert.match(system, /^Look at the screenshot\./);
  assert.match(system, /provided as text read from a screenshot by OCR/);
  assert.match(userText, /^Question: what is this\?/);
  assert.match(userText, /<<<\nTwo Sum\ndef twoSum\(nums, target\):\n>>>/);
});

test('an empty OCR result says so rather than looking like no screen was sent', () => {
  assert.match(withScreenText('s', 'q', '').userText, /OCR found no readable text/);
});

test('OCR is Windows-only and the image path never goes into the script', () => {
  assert.equal(isAvailable('win32'), true);
  assert.equal(isAvailable('darwin'), false);
  assert.match(WINDOWS_OCR_SCRIPT, /\$env:CUE_OCR_IMAGE/);
  assert.match(WINDOWS_OCR_SCRIPT, /IAsyncOperation`1/);
});

test('recognizeText rejects off Windows', { skip: process.platform === 'win32' }, async () => {
  await assert.rejects(recognizeText(Buffer.alloc(1)), /only available on Windows/);
});
