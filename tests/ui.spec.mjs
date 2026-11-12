import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { fileURLToPath } from 'node:url';
import AxeBuilder from '@axe-core/playwright';
import { expect, test } from '@playwright/test';

let server;
let baseURL;
test.use({ channel: 'chrome' });
test.beforeAll(async () => {
  server = createServer(async (request, response) => {
    const path = request.url.split('?')[0];
    if (
      !/^\/(?:index\.html|app\.js|workbench\.mjs|memory\.mjs|charts\.mjs|styles\.css|tokens\.css)?$/.test(
        path,
      )
    ) {
      response.writeHead(404).end();
      return;
    }
    const file = fileURLToPath(
      new URL(`../web/${path === '/' ? 'index.html' : path.slice(1)}`, import.meta.url),
    );
    const type = path.endsWith('css')
      ? 'text/css'
      : /\.(?:m?js)$/.test(path)
        ? 'text/javascript'
        : 'text/html';
    response.writeHead(200, { 'Content-Type': type });
    response.end(await readFile(file));
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseURL = `http://127.0.0.1:${server.address().port}`;
});
test.afterAll(async () => {
  await new Promise((resolve) => server.close(resolve));
});

// Explicit contract fixtures exercise UI states; these are not scientific validation data.
const info = {
  available: true,
  inputs: [{ id: 'threat_cue', label: 'Threat cue', group: 'senses', default: 0 }],
  timeline: { stimulus_seconds: [2.5, 5] },
  network_sizes: [16384],
  default_neurons: 16384,
  hardware: { available: false, name: 'test CPU' },
};
const result = {
  simulation: {
    neuron_count: 16384,
    requested_neuron_count: 16384,
    synapse_count: 100,
    duration_seconds: 7,
    device: 'cpu',
    random_seed: 17,
    elapsed_seconds: 1,
  },
  timeline: { duration_seconds: 7, stimulus_seconds: [2.5, 5] },
  spikes: {
    total_count: 2,
    layout: [{ id: 'insula', cells: 2, projection: 1 }],
    events: [
      [3000, 0],
      [4000, 1],
    ],
  },
  emotions: [{ label: 'Test pattern', score: 20 }],
  dimensions: { valence: 0, arousal: 0.1, agency: 0, social_safety: 0 },
  chemistry: {
    neuromodulators: [],
    synaptic_release_per_cell_hz: { glutamate: [1, 2], gaba: [1, 1] },
    timecourse: { t: [0, 3, 7], series: { dopamine: [0.2, 0.3, 0.2] } },
  },
  eeg: {
    channels: { frontal: [0, 1, 0], central: [0, 10, 0], posterior: [0, -2, 0] },
    spectrum: { frequencies_hz: [1, 20, 45], power: [0, 1, 0], bands: { alpha: 10 }, peak_hz: 20 },
  },
  interpretation: 'Test fixture only.',
  caveat: 'Not a clinical assessment.',
  regions: Array.from({ length: 8 }, (_, index) => ({
    id: `r${index}`,
    label: `Region ${index}`,
    cells: 2,
    role: 'fixture',
    transmitter: 'glutamate',
    baseline_hz: 1,
    response_hz: 2,
    change_log2: index,
  })),
};
test.beforeEach(async ({ page }) => {
  await page.route('**/api/**', async (route) => {
    const path = new URL(route.request().url()).pathname;
    const payload = path === '/api/info' ? info : path === '/api/simulate' ? result : [];
    await route.fulfill({ json: payload });
  });
  await page.goto(baseURL);
});

test('plots retain scales, region toggle expands and results survive resize', async ({ page }) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.getByRole('button', { name: 'Run simulation' }).click();
  await expect(page.locator('#top-emotion')).toHaveText('Test pattern');
  await expect(page.locator('#eeg-scale')).toHaveText('Shared scale: ±10.0 a.u.');
  await expect(page.locator('.region-row:visible')).toHaveCount(6);
  await page.locator('#region-toggle').click();
  await expect(page.locator('.region-row:visible')).toHaveCount(8);
  await expect(page.locator('#region-toggle')).toHaveAttribute('aria-expanded', 'true');
  await page.locator('#region-toggle').click();
  await expect(page.locator('.region-row:visible')).toHaveCount(6);
  for (const width of [320, 375, 768, 1280, 1920]) {
    await page.setViewportSize({ width, height: 900 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(
      true,
    );
    const box = await page.locator('#spike-raster').boundingBox();
    expect(box.width).toBeGreaterThan(100);
  }
  expect(errors).toEqual([]);
});

test('failed storage leaves the visible record and durable memory unchanged', async ({ page }) => {
  await page.getByRole('link', { name: 'Memory', exact: true }).click();
  await page.locator('#person-name').fill('Example');
  await page.locator('#person-meaning').fill('Test record');
  await page.locator('#save-person').click();
  await expect(page.locator('.person-card')).toHaveCount(1);
  await page.evaluate(() => {
    Storage.prototype.setItem = () => {
      throw new DOMException('Quota exceeded', 'QuotaExceededError');
    };
  });
  page.on('dialog', (dialog) => dialog.accept());
  await page.locator('.delete-person').click();
  await expect(page.locator('#memory-status')).toContainText('Memory was not changed');
  await expect(page.locator('.person-card')).toHaveCount(1);
  expect(
    await page.evaluate(
      () => JSON.parse(localStorage.getItem('human-emotion-memory-v1')).people.length,
    ),
  ).toBe(1);
});

test('navigation, visible focus and empty destinations are accessible', async ({ page }) => {
  for (const name of ['Simulation', 'Sessions', 'Experiments', 'Memory', 'Model']) {
    await page.getByRole('link', { name, exact: true }).click();
    await expect(page.locator('#page-title')).toHaveText(name);
    const scan = await new AxeBuilder({ page })
      .withTags(['wcag2a', 'wcag2aa', 'wcag21aa'])
      .analyze();
    expect(scan.violations, JSON.stringify(scan.violations, null, 2)).toEqual([]);
  }
  await page.keyboard.press('Tab');
  await page.getByRole('link', { name: 'Simulation', exact: true }).focus();
  expect(
    await page
      .getByRole('link', { name: 'Simulation', exact: true })
      .evaluate((node) => getComputedStyle(node).outlineStyle),
  ).toBe('solid');
});

test('a lost session response retries the same run without transmitting browser notes', async ({
  page,
}) => {
  const session = {
    id: 'session-test',
    head_revision: 0,
    neurons: 16384,
    learning: { rule: 'pair_stdp' },
    revisions: [],
  };
  await page.route('**/api/sessions', (route) => route.fulfill({ json: [session] }));
  const requests = [];
  await page.route('**/api/simulate', async (route) => {
    requests.push(route.request().postDataJSON());
    if (requests.length === 1) await route.abort('failed');
    else
      await route.fulfill({ json: { ...result, session: { id: session.id, head_revision: 1 } } });
  });
  await page.reload();
  await page.locator('.context-controls summary').click();
  await page.locator('#event-text').fill('Private browser note');
  await page.locator('.compute-row summary').click();
  await page.locator('#session-select').selectOption(session.id);
  await page.locator('#run-button').click();
  await expect(page.locator('#run-status')).toContainText('Could not run:');
  await page.locator('#run-button').click();
  await expect(page.locator('#top-emotion')).toHaveText('Test pattern');
  expect(requests).toHaveLength(2);
  expect(requests[0].session.run_id).toEqual(requests[1].session.run_id);
  expect(requests[0]).not.toHaveProperty('event');
  await expect(page.locator('#session-revision')).toHaveValue('1');
});

test('invalid backup import preserves the entire existing browser memory', async ({ page }) => {
  await page.getByRole('link', { name: 'Memory', exact: true }).click();
  await page.locator('#person-name').fill('Example');
  await page.locator('#person-meaning').fill('Test record');
  await page.locator('#save-person').click();
  page.on('dialog', (dialog) => dialog.accept());
  await page.locator('#import-memory').setInputFiles({
    name: 'invalid.json',
    mimeType: 'application/json',
    buffer: Buffer.from(
      JSON.stringify({ people: [{ id: 'bad', name: 'Invalid', trust: 7 }], episodes: [] }),
    ),
  });
  await expect(page.locator('.person-card h4')).toHaveText('Example');
  expect(
    await page.evaluate(
      () => JSON.parse(localStorage.getItem('human-emotion-memory-v1')).people[0].name,
    ),
  ).toBe('Example');
});
