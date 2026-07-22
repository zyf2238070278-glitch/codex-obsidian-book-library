import { lstat, readFile } from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { pathToFileURL } from 'node:url';


const MAX_IMAGE_BYTES = 256 * 1024 * 1024;
const EXECUTION_PROVIDERS = new Set(['auto', 'cpu', 'apple', 'webgpu']);


export function executionOptions(value) {
  const provider = value || 'cpu';
  if (!EXECUTION_PROVIDERS.has(provider)) {
    throw new TypeError('LIGHT_OCR_EXECUTION is not a supported provider');
  }
  return provider === 'cpu'
    ? { provider }
    : { provider, sessionFallback: 'cpu' };
}


function rounded(value) {
  return Math.round(value * 1e12) / 1e12;
}


function pointsFromQuadrilateral(value) {
  if (!Array.isArray(value)) throw new TypeError('quadrilateral must be an array');
  if (value.length === 8 && value.every(Number.isFinite)) {
    return [
      [value[0], value[1]],
      [value[2], value[3]],
      [value[4], value[5]],
      [value[6], value[7]],
    ];
  }
  if (value.length === 4) {
    return value.map((point) => {
      const x = Array.isArray(point) ? point[0] : point?.x;
      const y = Array.isArray(point) ? point[1] : point?.y;
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        throw new TypeError('quadrilateral points must be finite');
      }
      return [x, y];
    });
  }
  throw new TypeError('quadrilateral must contain four points');
}


export function normalizeLine(line, imageWidth, imageHeight) {
  if (!line || typeof line !== 'object') throw new TypeError('line must be an object');
  if (typeof line.text !== 'string' || line.text.trim() === '') {
    throw new TypeError('line text must be nonblank');
  }
  if (!Number.isFinite(line.confidence) || line.confidence < 0 || line.confidence > 1) {
    throw new TypeError('line confidence must be between 0 and 1');
  }
  if (!Number.isInteger(imageWidth) || imageWidth <= 0 || !Number.isInteger(imageHeight) || imageHeight <= 0) {
    throw new TypeError('image dimensions must be positive integers');
  }
  const points = pointsFromQuadrilateral(line.box ?? line.quadrilateral);
  const alreadyNormalized = points.every(([x, y]) => x >= 0 && y >= 0 && x <= 1 && y <= 1);
  const normalized = points.map(([x, y]) => (
    alreadyNormalized ? [x, y] : [x / imageWidth, y / imageHeight]
  ));
  const xs = normalized.map(([x]) => x);
  const ys = normalized.map(([, y]) => y);
  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);
  if (left < 0 || top < 0 || right > 1.000001 || bottom > 1.000001 || right <= left || bottom <= top) {
    throw new RangeError('quadrilateral is outside normalized image bounds');
  }
  return {
    text: line.text.trim(),
    confidence: line.confidence,
    box: {
      x: rounded(left),
      y: rounded(top),
      width: rounded(Math.min(1, right) - left),
      height: rounded(Math.min(1, bottom) - top),
    },
  };
}


export function imageDimensions(bytes) {
  if (!Buffer.isBuffer(bytes)) throw new TypeError('encoded image must be a Buffer');
  if (bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex'))) {
    const width = bytes.readUInt32BE(16);
    const height = bytes.readUInt32BE(20);
    if (width > 0 && height > 0) return { width, height };
  }
  if (bytes.length >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    const sof = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
    while (offset + 4 <= bytes.length) {
      while (offset < bytes.length && bytes[offset] === 0xff) offset += 1;
      if (offset >= bytes.length) break;
      const marker = bytes[offset];
      offset += 1;
      if (marker === 0xd8 || marker === 0xd9 || marker === 0x01) continue;
      if (offset + 2 > bytes.length) break;
      const length = bytes.readUInt16BE(offset);
      if (length < 2 || offset + length > bytes.length) break;
      if (sof.has(marker) && length >= 7) {
        const height = bytes.readUInt16BE(offset + 3);
        const width = bytes.readUInt16BE(offset + 5);
        if (width > 0 && height > 0) return { width, height };
      }
      offset += length;
    }
  }
  throw new TypeError('Light OCR accepts valid PNG or JPEG images');
}


async function recognizeRequest(engine, request) {
  if (!request || typeof request !== 'object') throw new TypeError('request must be an object');
  if (typeof request.id !== 'string' || request.id.trim() === '') throw new TypeError('request id is required');
  if (request.op !== 'recognize') throw new TypeError('unsupported operation');
  if (typeof request.image !== 'string' || !path.isAbsolute(request.image)) {
    throw new TypeError('image must be an absolute path');
  }
  const extension = path.extname(request.image).toLowerCase();
  if (!['.png', '.jpg', '.jpeg'].includes(extension)) throw new TypeError('image must be PNG or JPEG');
  const info = await lstat(request.image);
  if (!info.isFile() || info.isSymbolicLink()) throw new TypeError('image must be a regular file');
  if (info.size <= 0 || info.size > MAX_IMAGE_BYTES) throw new RangeError('image exceeds byte limit');
  const bytes = await readFile(request.image);
  const { width, height } = imageDimensions(bytes);
  const result = await engine.recognizeEncoded(bytes);
  if (!result || !Array.isArray(result.lines)) throw new TypeError('Light OCR returned invalid lines');
  const lines = [];
  for (const line of result.lines) {
    try {
      lines.push(normalizeLine(line, width, height));
    } catch {
      // A malformed observation is discarded; Python applies the final page gate.
    }
  }
  return { id: request.id, ok: true, lines };
}


async function run() {
  const { createEngine } = await import('@arcships/light-ocr');
  const engine = await createEngine({
    queueCapacity: 1,
    maxPendingInputBytes: MAX_IMAGE_BYTES,
    execution: executionOptions(process.env.LIGHT_OCR_EXECUTION),
  });
  const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
  try {
    for await (const raw of input) {
      if (raw.length > 1024 * 1024) {
        process.stdout.write(`${JSON.stringify({ id: null, ok: false, error: 'request exceeds protocol limit' })}\n`);
        continue;
      }
      let request;
      try {
        request = JSON.parse(raw);
      } catch {
        process.stdout.write(`${JSON.stringify({ id: null, ok: false, error: 'request must be valid JSON' })}\n`);
        continue;
      }
      if (request?.op === 'close') break;
      try {
        process.stdout.write(`${JSON.stringify(await recognizeRequest(engine, request))}\n`);
      } catch (error) {
        const message = error instanceof Error ? `${error.name}: ${error.message}` : 'recognition failed';
        process.stdout.write(`${JSON.stringify({ id: request?.id ?? null, ok: false, error: message.slice(0, 500) })}\n`);
      }
    }
  } finally {
    input.close();
    await engine.close();
  }
}


const isMain = process.argv[1]
  && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
  run().catch((error) => {
    const message = error instanceof Error ? `${error.name}: ${error.message}` : 'startup failed';
    process.stderr.write(`light-ocr worker failed: ${message.slice(0, 500)}\n`);
    process.exitCode = 1;
  });
}
