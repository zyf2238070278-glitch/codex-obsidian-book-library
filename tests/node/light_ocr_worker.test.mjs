import assert from 'node:assert/strict';
import test from 'node:test';

import {
  executionOptions,
  imageDimensions,
  normalizeLine,
} from '../../scripts/light_ocr_worker.mjs';


test('executionOptions defaults to the qualified CPU compatibility path', () => {
  assert.deepEqual(executionOptions(undefined), { provider: 'cpu' });
});


test('normalizeLine converts pixel quadrilateral to normalized top-left box', () => {
  const line = normalizeLine(
    {
      text: '测试文字',
      confidence: 0.9,
      quadrilateral: [10, 20, 40, 20, 40, 30, 10, 30],
    },
    100,
    100,
  );

  assert.deepEqual(line, {
    text: '测试文字',
    confidence: 0.9,
    box: { x: 0.1, y: 0.2, width: 0.3, height: 0.1 },
  });
});


test('normalizeLine accepts normalized point objects', () => {
  const line = normalizeLine(
    {
      text: 'ABC',
      confidence: 0.75,
      quadrilateral: [
        { x: 0.2, y: 0.3 },
        { x: 0.5, y: 0.3 },
        { x: 0.5, y: 0.4 },
        { x: 0.2, y: 0.4 },
      ],
    },
    1000,
    800,
  );

  assert.deepEqual(line.box, { x: 0.2, y: 0.3, width: 0.3, height: 0.1 });
});


test('normalizeLine accepts the official OcrLine box property', () => {
  const line = normalizeLine(
    {
      text: 'Official',
      confidence: 0.88,
      box: [
        { x: 20, y: 10 },
        { x: 60, y: 10 },
        { x: 60, y: 30 },
        { x: 20, y: 30 },
      ],
    },
    100,
    50,
  );

  assert.deepEqual(line.box, { x: 0.2, y: 0.2, width: 0.4, height: 0.4 });
});


test('imageDimensions reads PNG IHDR dimensions', () => {
  const bytes = Buffer.alloc(24);
  Buffer.from('89504e470d0a1a0a', 'hex').copy(bytes);
  bytes.writeUInt32BE(1234, 16);
  bytes.writeUInt32BE(567, 20);

  assert.deepEqual(imageDimensions(bytes), { width: 1234, height: 567 });
});


test('normalizeLine rejects malformed geometry', () => {
  assert.throws(
    () => normalizeLine({ text: 'bad', confidence: 1, quadrilateral: [1, 2] }, 10, 10),
    /quadrilateral/,
  );
});
