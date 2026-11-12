# Learning and session persistence

Learning is off by default. A request selects exactly one version-1 rule using
`learning.rule`: `none`, `pair_stdp` or `dopamine_eligibility`. The effective
parameters and rule version are included in simulation metadata. Unsupported old
parameters fail with a migration error rather than acquiring a new meaning.

Only contacts marked plastic in a glutamate transmission group can change.
GABAergic contacts and other fixed contacts keep their initial values. For an
initial contact weight `w0`, both rules clamp updated weights to
`w_min = 0.1 * w0` and `w_max = 2 * w0`. Learning uses private weight arrays and
does not mutate the topology's initial weights.

## Pair STDP

For source `i` and target `j`, `s_i` and `s_j` are the current step's binary
spikes. First decay the pre/post traces, then compute the weight change, then add
the current spikes to the traces:

```text
p_i <- p_i * exp(-dt_ms / tau_plus_ms)
q_j <- q_j * exp(-dt_ms / tau_minus_ms)

delta_w = g * (
    A_plus  * p_i * s_j * (w_max - w) / (w_max - w_min)
  - A_minus * q_j * s_i * (w - w_min) / (w_max - w_min)
)
w <- clip(w + delta_w, w_min, w_max)

p_i <- p_i + s_i
q_j <- q_j + s_j
```

The default amplitudes are `A_plus = 0.001`, `A_minus = 0.0012` in normalised
weight units, `tau_plus_ms = tau_minus_ms = 20 ms` and `g = 1`. The gate `g` is
dimensionless in [0,1]. This is a per-spike pairing update; it has no extra
`dt_seconds` multiplier. Current-step spikes are inserted after the update, so
simultaneous spikes alone do not form a causal pair. A zero initial weight remains
zero under the bounds.

## Experimental dopamine eligibility

This rule shares the decayed pre/post traces above. It accumulates only the
causal term `p_i * s_j` into contact eligibility `e_ij`; it does not use the
pair rule's anti-causal term:

```text
e_ij <- e_ij * exp(-dt_seconds / trace_tau_s) + p_i * s_j
D_j = dopamine_occupancy(region_j) - baseline_occupancy(region_j)
D_gated = D_j if abs(D_j) >= da_threshold else 0
delta_w = eta * dt_seconds * e_ij * D_gated
w <- clip(w + delta_w, w_min, w_max)
```

Defaults are `trace_tau_s = 1 s`, `eta = 0.0001` normalised weight units per second
and `da_threshold = 0.05` occupancy units. Eligibility and occupancy differences
are dimensionless. The regional baseline is the mean dopamine occupancy over the
pre-stimulus baseline window. The simulator supplies zero occupancy difference
before stimulus onset. Positive differences potentiate eligible contacts and
negative differences depress them. The parameter `g` belongs to pair STDP and
does not scale this rule.

There is no supplied reward, reward prediction error, value function or
temporal-difference target. A weight change demonstrates the specified mechanism;
it does not establish learned human meaning or a successful behavioural model.

## Episode modes

| Mode or action | Starting weights | Dynamic state | Persistence |
| --- | --- | --- | --- |
| Independent simulation | Initial topology weights | Fresh episode | None |
| Session `learn` | Selected current HEAD revision | Fresh episode | Commits learned weights as the next revision |
| Session `evaluate` | Selected saved revision | Fresh episode, learning disabled | Saves the response; no weight revision |
| Session `replay` | Selected saved revision | Fresh episode with requested learning | Saves the response; no weight revision |
| Continuous research chunk | Previous continuous state | Restored neurons, chemicals, delays, traces, RNG and accumulators | Numeric chunk and continuation state |

Session episodes retain weights only. They do not carry neuronal activity,
chemical state, short-term plasticity or eligibility into the next episode.
Continuous chunks preserve those quantities to reproduce an uninterrupted
episode. Evaluation and replay can inspect an older revision. Learning requires
the current HEAD to prevent two runs from silently replacing each other's work.

Each session response includes a no-learning reference computed with the starting
weights, plus the frozen training reference from session creation. A reset appends
a revision containing initial weights; it does not delete revision history.

## Publication, retries and checkpoints

A session request includes an explicit seed and this identity:

```json
{
  "session": {
    "id": "SESSION_ID",
    "from_revision": 0,
    "run_id": "UNIQUE_RUN_ID",
    "action": "learn"
  }
}
```

This fragment accompanies the numeric simulation request. The idempotency
fingerprint binds the canonical numerical inputs, effective learning parameters,
seed, action, session and starting revision. Reusing a `run_id` with a different
identity fails. Retrying the same completed run returns its saved response before
the stale-HEAD check, so a lost HTTP response cannot train the weights twice.

One execution lock spans private computation, finite-JSON response construction
and persistence. Publication writes and flushes staged files, renames the complete
revision directory, then atomically replaces `HEAD.json`. HEAD identifies visible
state. An error before publication preserves the previous visible revision. If a
complete revision exists after an interrupted HEAD write, the same retry validates
it and publishes it when its parent is still HEAD.

The JSON manifest stores configuration, version and topology identity. Dense
safetensors hold the realised topology once and learned weights per revision.
Import checks hashes, schemas, shapes, dtypes, finite values, bounds, topology
references and revision consistency. It does not load pickle or executable model
objects. The default session storage budget is 8 GiB; exceeding it fails without
automatic deletion of old sessions or outputs.

Names, photographs and browser event notes are excluded from scientific request
records. Keep identifying annotations in browser memory and its explicit backups.
Research checkpoints use a separate numeric recording format described in
[the research guide](RESEARCH.md).
