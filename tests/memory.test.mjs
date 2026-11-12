import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  createMemoryStore,
  LEGACY_STORAGE_KEY,
  STORAGE_KEY,
  validateMemory,
} from '../web/memory.mjs';

function storage(entries = []) {
  const rows = new Map(entries);
  return {
    rows,
    getItem: (key) => rows.get(key) ?? null,
    setItem: (key, value) => rows.set(key, value),
    removeItem: (key) => rows.delete(key),
  };
}
const saved = {
  people: [{ id: 'sam', name: 'Sam', photoDataUrl: 'data:image/jpeg;base64,AQ==' }],
  episodes: [{ id: 'e1', personId: 'sam', seed: 17 }],
};
test('migrates legacy records only after durable write', () => {
  const disk = storage([[LEGACY_STORAGE_KEY, JSON.stringify(saved)]]);
  const store = createMemoryStore(disk);
  assert.deepEqual(store.value, saved);
  assert.equal(disk.rows.has(LEGACY_STORAGE_KEY), false);
  assert.deepEqual(JSON.parse(disk.rows.get(STORAGE_KEY)), saved);
});
test('quota failure preserves live state and old records', () => {
  const disk = storage([[LEGACY_STORAGE_KEY, JSON.stringify(saved)]]);
  disk.setItem = () => {
    throw new Error('quota');
  };
  const store = createMemoryStore(disk);
  assert.throws(() => store.commit({ people: [], episodes: [] }), /quota/);
  assert.deepEqual(store.value, saved);
  assert.equal(disk.rows.has(LEGACY_STORAGE_KEY), true);
});
test('a cleared current store never revives a legacy copy', () => {
  const empty = { people: [], episodes: [] };
  assert.deepEqual(
    createMemoryStore(
      storage([
        [STORAGE_KEY, JSON.stringify(empty)],
        [LEGACY_STORAGE_KEY, JSON.stringify(saved)],
      ]),
    ).value,
    empty,
  );
});
test('rejects whole malformed imports without altering memory', () => {
  const store = createMemoryStore(storage([[STORAGE_KEY, JSON.stringify(saved)]]));
  for (const bad of [
    null,
    { people: [null], episodes: [] },
    { people: [{ id: 'x', name: 'x', trust: 2 }], episodes: [] },
    {
      people: [{ id: 'x', name: 'x', photoDataUrl: 'javascript:alert(1)' }],
      episodes: [],
    },
    { people: [], episodes: [{ id: 'x', dimensions: { valence: NaN } }] },
    { people: [saved.people[0], saved.people[0]], episodes: [] },
  ]) {
    assert.throws(() => store.commit(bad));
    assert.deepEqual(store.value, saved);
  }
});
test('accepts current and legacy backup envelopes, preserves optional fields', () => {
  for (const format of ['human-emotion-memory', 'human-emotion-lab-memory'])
    assert.deepEqual(validateMemory({ format, version: 1, ...saved }), saved);
  assert.throws(() => validateMemory({ format: 'another-app', version: 1, ...saved }));
});
