import assert from 'node:assert/strict';
import { test } from 'node:test';
import { sharedAmplitude } from '../web/charts.mjs';

test('one amplitude scale includes outliers in every EEG-like channel', () => {
  assert.equal(sharedAmplitude({ a: [1, 2, 3], b: [-100, 0], c: Array(200).fill(1) }), 100);
  assert.equal(sharedAmplitude({ a: [0, 0], b: [] }), 1);
});
