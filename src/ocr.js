// Local OCR for providers that can't read images (every Cerebras model). The
// screenshot is turned into text on this machine and that text goes to the
// model with the question, so a "look at my screen" request is answered about
// the screen instead of being answered blind. Windows only (Windows.Media.Ocr:
// built in, offline, nothing to install); elsewhere isAvailable() is false.
const { execFile } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const OCR_TIMEOUT_MS = 15000;
// Keeps the prompt bounded on a screen full of text.
const MAX_OCR_CHARS = 12000;

// Runs in Windows PowerShell 5.1 (powershell.exe), whose WinRT interop loads
// these types; PowerShell 7 can't. The image path comes in via an environment
// variable so it is never spliced into the script text.
const WINDOWS_OCR_SCRIPT = String.raw`
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null = [Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime]
$null = [Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime]
$null = [Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime]
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and
    $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation${'`'}1'
  })[0]
function Await($op, [Type]$type) {
  $task = $asTask.MakeGenericMethod($type).Invoke($null, @($op))
  $null = $task.Wait(-1)
  $task.Result
}
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages()
if ($null -eq $engine) { throw 'No Windows OCR language is installed.' }
$file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($env:CUE_OCR_IMAGE)) ([Windows.Storage.StorageFile])
$stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
try {
  $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
  $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
  $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
  foreach ($line in $result.Lines) { $line.Text }
} finally {
  $stream.Dispose()
}
`;

function isAvailable(platform = process.platform) {
  return platform === 'win32';
}

/** Trim, drop blank lines, and cap the length of OCR output. */
function cleanOcrText(raw) {
  const text = String(raw || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean).join('\n');
  return text.length > MAX_OCR_CHARS ? text.slice(0, MAX_OCR_CHARS) + '\n…' : text;
}

/** OCR an image Buffer (PNG or JPEG). Resolves to the recognized text ('' if none). */
function recognizeText(imageBuffer, { timeoutMs = OCR_TIMEOUT_MS } = {}) {
  if (!isAvailable()) return Promise.reject(new Error('Screen text reading is only available on Windows.'));
  const tmp = path.join(os.tmpdir(), `cue_ocr_${Date.now()}_${Math.random().toString(36).slice(2)}.png`);
  return new Promise((resolve, reject) => {
    fs.writeFile(tmp, imageBuffer, (writeErr) => {
      if (writeErr) return reject(writeErr);
      const encoded = Buffer.from(WINDOWS_OCR_SCRIPT, 'utf16le').toString('base64');
      execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded], {
        env: { ...process.env, CUE_OCR_IMAGE: tmp },
        timeout: timeoutMs,
        windowsHide: true,
        maxBuffer: 8 * 1024 * 1024
      }, (err, stdout, stderr) => {
        // A picture of the user's screen: never leave it behind.
        fs.unlink(tmp, () => {});
        if (err) return reject(new Error('Reading the text on your screen failed: ' + (String(stderr || '').trim().split(/\r?\n/)[0] || err.message)));
        resolve(cleanOcrText(stdout));
      });
    });
  });
}

/**
 * Put OCR'd screen text into a prompt written for a screenshot. The system
 * prompts say "look at the screenshot"; this tells the model the screenshot
 * arrives as text, and appends that text to the user message.
 */
function withScreenText(system, userText, screenText) {
  const note = '\n\nThe user\'s screen is provided as text read from a screenshot by OCR (layout, colors and images are lost; code indentation may be off). ' +
    'Wherever these instructions refer to the screenshot or the screen, use that text.';
  const block = screenText
    ? '\n\nText on my screen right now (OCR):\n<<<\n' + screenText + '\n>>>'
    : '\n\n(OCR found no readable text on my screen.)';
  return { system: String(system || '') + note, userText: String(userText || '') + block };
}

module.exports = { recognizeText, isAvailable, cleanOcrText, withScreenText, MAX_OCR_CHARS, WINDOWS_OCR_SCRIPT };
