import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createInterface } from 'node:readline';
import { expect, test } from '@playwright/test';

let child;
let baseURL;
test.use({ channel: 'chrome' });
test.beforeAll(async () => {
  const python = process.platform === 'win32' ? '.venv/Scripts/python.exe' : '.venv/bin/python';
  child = spawn(python, ['tests/serve_ui.py'], {
    env: {
      ...process.env,
      HUMAN_EMOTION_DEVICE: 'cpu',
      PYTHONUTF8: '1',
      OMP_NUM_THREADS: '1',
      MKL_NUM_THREADS: '1',
    },
    stdio: ['ignore', 'pipe', 'inherit'],
    windowsHide: true,
  });
  const lines = createInterface({ input: child.stdout });
  const [line] = await Promise.race([
    once(lines, 'line'),
    once(child, 'exit').then(([code]) => {
      throw new Error(`Python test service exited during startup (${code}).`);
    }),
  ]);
  baseURL = `http://127.0.0.1:${JSON.parse(line).port}`;
});
test.afterAll(() => child?.kill());

test('real Python model produces browser plots and committed session revisions', async ({
  page,
}) => {
  const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(baseURL);
  await expect(page.locator('#run-button')).toBeEnabled();
  await page.getByText('Population and learning', { exact: true }).click();
  await page.locator('#seed-input').fill('23');
  await page.locator('#learning-rule').selectOption('pair_stdp');
  await page.getByRole('link', { name: 'Sessions', exact: true }).click();
  await page.locator('#create-session').click();
  await expect(page.locator('#session-list tr')).toHaveCount(1);
  await page.getByRole('button', { name: 'Use', exact: true }).click();
  const response = page.waitForResponse((item) => item.url().endsWith('/api/simulate'));
  await page.locator('#run-button').click();
  const payload = await (await response).json();
  expect(payload.simulation.neuron_count).toBe(192);
  expect(payload.session.committed).toBe(true);
  expect(payload.session.revision).toBe(1);
  await expect(page.locator('#run-button')).toBeEnabled();
  await page.getByRole('link', { name: 'Sessions', exact: true }).click();
  await page.locator('#refresh-sessions').click();
  await expect(page.locator('#session-list tr td').nth(1)).toHaveText('1');
  expect(errors).toEqual([]);
});

test('real experiment queue completes a numeric job through browser controls', async ({ page }) => {
  await page.goto(`${baseURL}/#experiments`);
  await page.locator('#experiment-manifest').fill(
    JSON.stringify({
      protocol: 'mechanics',
      duration_seconds: 0.02,
      chunk_seconds: 0.01,
      neurons: 256,
    }),
  );
  await page.getByRole('button', { name: 'Start experiment', exact: true }).click();
  await expect(async () => {
    const jobs = await (await page.request.get(`${baseURL}/api/experiments`)).json();
    expect(jobs[0].status).toBe('completed');
    expect(jobs[0].chunks).toHaveLength(2);
  }).toPass({ timeout: 15000 });
  await page.locator('#refresh-experiments').click();
  await expect(page.locator('#experiment-list')).toContainText('completed: 100%');
});
