export const STORAGE_KEY = 'human-emotion-memory-v1';
export const LEGACY_STORAGE_KEY = 'human-emotion-lab-memory-v1';
const empty = () => ({ people: [], episodes: [] });
const record = (value) => value !== null && typeof value === 'object' && !Array.isArray(value);
function requireValue(condition, message) {
  if (!condition) throw new Error(`Invalid memory: ${message}.`);
}
function text(row, key, maximum, required = false) {
  if (!(key in row) && !required) return;
  requireValue(
    typeof row[key] === 'string' &&
      row[key].length <= maximum &&
      (!required || row[key].trim().length > 0),
    key,
  );
}
function number(row, key, minimum, maximum, integer = false) {
  if (!(key in row)) return;
  requireValue(
    typeof row[key] === 'number' &&
      Number.isFinite(row[key]) &&
      row[key] >= minimum &&
      row[key] <= maximum &&
      (!integer || Number.isInteger(row[key])),
    key,
  );
}
function date(row, key) {
  if (key in row)
    requireValue(typeof row[key] === 'string' && Number.isFinite(Date.parse(row[key])), key);
}

// Older backups omit some fields. Validate every present field; never truncate an import.
export function validateMemory(value) {
  requireValue(record(value), 'expected a backup object');
  if ('format' in value)
    requireValue(
      ['human-emotion-memory', 'human-emotion-lab-memory'].includes(value.format) &&
        value.version === 1,
      'unsupported backup format',
    );
  for (const key of ['people', 'episodes']) {
    requireValue(
      Array.isArray(value[key]) && value[key].length <= 250,
      `${key} must contain at most 250 records`,
    );
    const ids = new Set();
    for (const row of value[key]) {
      requireValue(record(row), `${key} record`);
      text(row, 'id', 200, true);
      requireValue(!ids.has(row.id), `duplicate ${key} id`);
      ids.add(row.id);
    }
  }
  for (const person of value.people) {
    text(person, 'name', 80, true);
    text(person, 'relationship', 60);
    text(person, 'meaning', 500);
    text(person, 'notes', 5000);
    for (const key of ['trust', 'closeness', 'valence']) number(person, key, 0, 1);
    date(person, 'createdAt');
    if (person.photoDataUrl != null)
      requireValue(
        typeof person.photoDataUrl === 'string' &&
          person.photoDataUrl.length <= 5_000_000 &&
          /^data:image\/(?:jpeg|png|webp|gif);base64,[A-Za-z0-9+/=]+$/.test(person.photoDataUrl),
        'photo must be a local raster image',
      );
  }
  for (const episode of value.episodes) {
    text(episode, 'event', 500);
    text(episode, 'personId', 200);
    text(episode, 'personName', 80);
    text(episode, 'topLabel', 200);
    date(episode, 'savedAt');
    number(episode, 'seed', 0, 2147483647, true);
    number(episode, 'neurons', 1, Number.MAX_SAFE_INTEGER, true);
    number(episode, 'modelVersion', 1, 1000, true);
    for (const key of ['dimensions', 'inputs']) {
      if (!(key in episode)) continue;
      requireValue(record(episode[key]), key);
      for (const name of Object.keys(episode[key]))
        number(episode[key], name, key === 'dimensions' ? -1 : 0, 1);
    }
  }
  return structuredClone({ people: value.people, episodes: value.episodes });
}

export function createMemoryStore(storage) {
  let value = empty();
  let warning = '';
  try {
    const current = storage.getItem(STORAGE_KEY);
    const legacy = current === null ? storage.getItem(LEGACY_STORAGE_KEY) : null;
    if (current !== null || legacy !== null) value = validateMemory(JSON.parse(current ?? legacy));
    if (current === null && legacy !== null) {
      try {
        storage.setItem(STORAGE_KEY, JSON.stringify(value));
        storage.removeItem(LEGACY_STORAGE_KEY);
      } catch {
        warning = 'Legacy memory remains available; migration could not be saved.';
      }
    }
  } catch {
    warning = 'Saved memory could not be read. The original browser data has been retained.';
  }
  return {
    get value() {
      return structuredClone(value);
    },
    get warning() {
      return warning;
    },
    commit(candidate) {
      const checked = validateMemory(candidate);
      storage.setItem(STORAGE_KEY, JSON.stringify(checked));
      value = checked;
      warning = '';
      try {
        storage.removeItem(LEGACY_STORAGE_KEY);
      } catch {
        warning = 'Memory saved; an older browser copy could not be removed.';
      }
      return structuredClone(value);
    },
  };
}
