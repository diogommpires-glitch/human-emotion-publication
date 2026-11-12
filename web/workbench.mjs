const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `Request failed (${response.status}).`);
  return payload;
}
const post = (path, value = {}) =>
  api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(value),
  });
function button(label, action) {
  const element = document.createElement('button');
  element.type = 'button';
  element.textContent = label;
  element.addEventListener('click', async () => {
    element.disabled = true;
    try {
      await action();
    } finally {
      element.disabled = false;
    }
  });
  return element;
}
function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function mountWorkbench(redraw) {
  let sessions = [];
  let pendingRun = null;
  const titles = {
    simulation: 'Simulation',
    sessions: 'Sessions',
    experiments: 'Experiments',
    memory: 'Memory',
    grounding: 'Model',
  };
  function navigate() {
    const destination = location.hash.slice(1);
    const active = destination in titles ? destination : 'simulation';
    $$('.destination').forEach((panel) => {
      panel.hidden = panel.id !== active;
    });
    $$('.page-nav a').forEach((link) => {
      if (link.hash === `#${active}`) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    });
    $('#page-title').textContent = titles[active];
    if (active === 'simulation') requestAnimationFrame(redraw);
  }
  window.addEventListener('hashchange', navigate);
  navigate();

  function selectSession(id) {
    $('#session-select').value = id;
    const session = sessions.find((item) => item.id === id);
    $('#session-run-controls').hidden = !session;
    if (session) {
      $('#session-revision').value = session.head_revision;
      $('#model-size').value = session.neurons;
      $('#learning-rule').value = session.learning.rule;
      $('#session-history').textContent = JSON.stringify(session.revisions, null, 2);
    }
    $('#model-size').disabled = Boolean(session);
    $('#learning-rule').disabled = Boolean(session);
    pendingRun = null;
  }
  $('#session-select').addEventListener('change', () => selectSession($('#session-select').value));
  async function refreshSessions() {
    $('#session-status').textContent = 'Loading sessions…';
    $('#session-list').setAttribute('aria-busy', 'true');
    try {
      sessions = await api('/api/sessions');
      const selected = $('#session-select').value;
      $('#session-select').replaceChildren(
        new Option('Independent run', ''),
        ...sessions.map(
          (session) => new Option(`${session.id} (r${session.head_revision})`, session.id),
        ),
      );
      $('#session-select').value = sessions.some((item) => item.id === selected) ? selected : '';
      $('#session-list').replaceChildren();
      for (const session of sessions) {
        const row = document.createElement('tr');
        for (const value of [
          session.id,
          session.head_revision,
          session.neurons.toLocaleString(),
          session.learning.rule,
        ]) {
          const cell = document.createElement('td');
          cell.textContent = value;
          row.append(cell);
        }
        const actions = document.createElement('td');
        actions.className = 'toolbar';
        const guarded = (action) => async () => {
          try {
            await action();
          } catch (error) {
            $('#session-status').textContent = error.message;
          }
        };
        actions.append(
          button('Use', () => {
            selectSession(session.id);
            location.hash = 'simulation';
          }),
          button(
            'History',
            guarded(async () => {
              const detail = await api(`/api/sessions/${session.id}`);
              $('#session-history').textContent = JSON.stringify(detail.revisions, null, 2);
              $('#session-detail').open = true;
            }),
          ),
          button(
            'Export',
            guarded(async () => {
              const response = await fetch(`/api/sessions/${session.id}/export`);
              if (!response.ok) throw new Error((await response.json()).error || 'Export failed.');
              download(await response.blob(), `${session.id}.zip`);
            }),
          ),
          button(
            'Reset',
            guarded(async () => {
              if (
                !confirm(
                  'Create a new revision with the initial weights? Existing revisions remain available.',
                )
              )
                return;
              await post(`/api/sessions/${session.id}/reset`, {
                from_revision: session.head_revision,
                run_id: crypto.randomUUID(),
              });
              await refreshSessions();
              if ($('#session-select').value === session.id) selectSession(session.id);
            }),
          ),
        );
        row.append(actions);
        $('#session-list').append(row);
      }
      $('#session-status').textContent = sessions.length
        ? `${sessions.length} local sessions.`
        : 'No sessions yet. Create one using the Simulation population, seed and learning rule.';
    } catch (error) {
      $('#session-status').textContent =
        `Could not load sessions: ${error.message} Use Refresh to retry.`;
    } finally {
      $('#session-list').removeAttribute('aria-busy');
    }
  }
  $('#refresh-sessions').addEventListener('click', refreshSessions);
  $('#create-session').addEventListener('click', async () => {
    const control = $('#create-session');
    control.disabled = true;
    $('#session-status').textContent = 'Creating initial weights…';
    try {
      const seed = $('#seed-input').value;
      const payload = {
        neurons: Number($('#model-size').value),
        learning: { rule: $('#learning-rule').value },
      };
      if (seed !== '') payload.seed = Number(seed);
      const session = await post('/api/sessions', payload);
      await refreshSessions();
      selectSession(session.id);
    } catch (error) {
      $('#session-status').textContent = `Could not create session: ${error.message}`;
    } finally {
      control.disabled = false;
    }
  });
  $('#import-session').addEventListener('change', async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 8 * 1024 ** 3) {
      $('#session-status').textContent = 'Checkpoint exceeds the default 8 GiB storage budget.';
      event.target.value = '';
      return;
    }
    $('#session-status').textContent = 'Importing checkpoint…';
    try {
      await api('/api/sessions/import', {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: file,
      });
      await refreshSessions();
    } catch (error) {
      $('#session-status').textContent = `Import failed: ${error.message}`;
    } finally {
      event.target.value = '';
    }
  });

  async function refreshExperiments() {
    $('#experiment-status').textContent = 'Loading experiments…';
    try {
      const experiments = await api('/api/experiments');
      $('#experiment-list').replaceChildren();
      for (const experiment of experiments) {
        const row = document.createElement('article');
        row.className = 'dataset-row';
        const content = document.createElement('div');
        const title = document.createElement('h3');
        title.textContent = experiment.id;
        const status = document.createElement('p');
        status.textContent = `${experiment.status}: ${Math.round(experiment.progress * 100)}%${experiment.error ? ` / ${experiment.error}` : ''}`;
        const progress = document.createElement('progress');
        progress.max = 1;
        progress.value = experiment.progress;
        progress.setAttribute('aria-label', `${experiment.id} progress`);
        content.append(title, status, progress);
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Manifest and results';
        const output = document.createElement('pre');
        output.textContent = JSON.stringify(experiment, null, 2);
        details.append(summary, output);
        content.append(details);
        const actions = document.createElement('div');
        actions.className = 'toolbar';
        const action = (verb) => async () => {
          try {
            await post(`/api/experiments/${experiment.id}/${verb}`);
            await refreshExperiments();
          } catch (error) {
            $('#experiment-status').textContent = `${verb} failed: ${error.message}`;
          }
        };
        if (['queued', 'running'].includes(experiment.status))
          actions.append(button('Cancel', action('cancel')));
        if (['cancelled', 'interrupted', 'failed'].includes(experiment.status))
          actions.append(button('Resume', action('resume')));
        row.append(content, actions);
        $('#experiment-list').append(row);
      }
      $('#experiment-status').textContent = experiments.length
        ? `${experiments.length} experiments. Refresh to read current progress.`
        : 'No experiments yet. Submit a versioned manifest to begin.';
    } catch (error) {
      $('#experiment-status').textContent =
        `Could not load experiments: ${error.message} Use Refresh to retry.`;
    }
  }
  $('#refresh-experiments').addEventListener('click', refreshExperiments);
  $('#experiment-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    const control = event.currentTarget.querySelector('button');
    control.disabled = true;
    try {
      await post('/api/experiments', JSON.parse($('#experiment-manifest').value));
      await refreshExperiments();
    } catch (error) {
      $('#experiment-status').textContent = `Could not start experiment: ${error.message}`;
    } finally {
      control.disabled = false;
    }
  });
  refreshSessions();
  refreshExperiments();
  return {
    request(payload) {
      payload.learning = { rule: $('#learning-rule').value };
      const id = $('#session-select').value;
      if (!id) return payload;
      const session = {
        id,
        from_revision: Number($('#session-revision').value),
        action: $('#session-action').value,
      };
      const signature = JSON.stringify({ ...payload, session });
      // A lost response must retry the same committed run, not learn twice.
      if (pendingRun?.signature !== signature)
        pendingRun = { signature, run_id: crypto.randomUUID() };
      return { ...payload, session: { ...session, run_id: pendingRun.run_id } };
    },
    async completed(result) {
      pendingRun = null;
      if (result.session) {
        await refreshSessions();
        $('#session-revision').value = result.session.head_revision;
      }
    },
  };
}
