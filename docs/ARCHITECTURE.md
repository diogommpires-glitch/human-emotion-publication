# Architecture and numerical units

The project uses Python/PyTorch, a standard-library localhost HTTP server and
native JavaScript/canvas. `network.json` supplies the model configuration. There
is no remote inference service. Explicit research acquisition commands are the
network boundary for downloading datasets and atlases.

## Module boundaries

| Module | Responsibility |
| --- | --- |
| `brain_model/config.py` | Configuration constants, finite/range checks and versioned learning parameters |
| `brain_model/topology.py` | Cell layout, sampled contacts, assemblies and canonical CSR transmission |
| `brain_model/simulator.py` | Integration, delays, short-term plasticity, chemicals, recording and continuous state |
| `brain_model/learning.py` | Contact-level weight updates and eligibility/pre/post traces |
| `brain_model/readouts.py` | Pure activity comparisons, descriptive labels, dimensions and EEG-like display traces |
| `brain_model/sessions.py` | Weight revisions, checkpoint validation, atomic publication and retry identity |
| `brain_model/model.py` | Public request validation, device/size selection, reference runs and JSON responses |
| `server.py` | Local HTTP boundaries, stable browser origin, request admission and service lifecycle |
| `research/` | Versioned job store, numeric recording, data adapters, observations and evaluation |
| `web/` | Controls, charts, sessions, experiments and transactional browser memory |

Existing Python entry points include `model.simulate`, `simulator.run` and
`simulator.build_topology`. `run_episode` also returns a `RunState` for continuous
research chunks. HTTP responses contain JSON data rather than simulator objects.

## Cells, contacts and transmission

Cells are contiguous by region and then by projection, pooled SST/PV and VIP
population. These are three population classes. Glutamate and GABA are two
transmitter channels, determined by the source cell. A projection cell can be
GABAergic, so projection membership must not be used as an excitatory mask.
Empty populations report zero rate rather than dividing by zero.

Each contact retains its source, target, initial weight and plasticity flag.
Contacts that share a source/target slot are summed into a canonical CSR matrix.
Learning changes eligible contacts independently and re-aggregates their values
into transmission slots. Stored checkpoints include the realised contact arrays,
cell parameters and stimulus assemblies. A seed alone does not identify them.

The default time step is 1 ms. Local release is delivered one step later, and
long-range release is delivered after 5 ms. The delay ring, pending recording
bins and random-generator state continue across research chunks.

## Integration and units

The membrane update is an explicit discrete AdEx model. With voltages in mV and
time constants in ms, the implemented subthreshold step is:

```text
V_next = V + (dt_ms / tau_m_ms) *
    (V_rest - V + Delta_T * exp(min((V - V_threshold) / Delta_T, 10))
     - w + I_glu + I_gaba + I_mod)
w_next = w + (dt_ms / tau_w_ms) * (a * (V - V_rest) - w)
```

Here `w` is a voltage-equivalent adaptation term. Conductances and connection
weights are normalised model quantities; `I_glu`, `I_gaba` and `I_mod` are
voltage-equivalent contributions, not currents calibrated in amperes. NMDA has
voltage-dependent magnesium block. Conductances decay exponentially before
incoming release is added. A threshold crossing resets voltage, adds the
spike-triggered adaptation increment and starts the refractory counter.

Short-term plasticity acts at each presynaptic cell. Resource availability `x`
recovers towards 1 and utilisation `u` relaxes towards `U`. At a spike the released
fraction is `u_spike * x`; `x` decreases by that fraction. Transmission uses
`release = spike * u_spike * x_before_release / U`. Division by `U` normalises
release at resting utilisation; facilitating cells raise utilisation on a spike.
This fast state resets between session episodes.

Neuromodulators relax towards a regional source-rate target:

```text
target = projection_density * source_rate_hz / resting_rate_hz
chemical_next = target + (chemical - target) * exp(-dt_ms / clearance_ms)
occupancy = chemical_next / (chemical_next + K)
```

`chemical` and `K` are internal dimensionless scales. Occupancy is dimensionless
in [0,1], not a molar concentration. Its change from resting occupancy modulates
excitability, adaptation and NMDA gain. Cortisol, adenosine and ghrelin inputs are
tonic model factors rather than a simulated endocrine system.

| Recorded value | Units or meaning |
| --- | --- |
| Simulation duration and time-course timestamps | Seconds |
| Integration time, spike-raster timestamps and configured delays | Milliseconds |
| Regional firing and normalised release per cell | Hz |
| Membrane voltage and voltage-equivalent adaptation | mV |
| Receptor occupancy and input slider levels | Dimensionless, 0 to 1 |
| EEG-like field potential and research neural drive | Arbitrary model units |
| Emotion-like scores and dimensions | Hand-designed summaries, without empirical calibration |

## Readouts and observations

Default episodes last 6 s. The baseline window is [1.0,2.5) s and the stimulus is
[2.5,5.0) s. Response averages use [2.6,5.0) s, skipping the first 100 ms of
transmission. A regional firing comparison uses
`clip(log2((response_hz + 0.5) / (baseline_hz + 0.5)), -2, 3)`.
Chemical evidence uses eight times the occupancy difference. Circuit summaries
combine stimulus-minus-baseline evidence with half of baseline-minus-reference
evidence. Labels and softmax-like scores describe these summaries and have no
probability calibration against human labels.

The displayed field-potential proxy is
`abs(I_glu)(t - 6 ms) - 1.65 * abs(I_gaba)(t)`, mixed by configured regions.
`I_glu` includes the model's AMPA and NMDA contributions. Display traces remove
slow drift and are sampled every 4 ms at the default step. They are not scalp
electrodes or microvolt EEG. The raster is sampled; its sample count must not be
confused with the full-network spike count.

Research BOLD uses the sum of absolute excitatory and inhibitory synaptic current
within whole regions, followed by an HRF observation model and train-only fits.
The empirical EEG adapter uses acquired EEG, not the display spectrum as measured
EEG. See [the research guide](RESEARCH.md) for preprocessing, timing, cohort
separation, train/test isolation and statistical choices.

## State and I/O boundaries

Independent simulations start with initial weights and reset dynamic state.
Sessions carry weights between fresh episodes. Continuous research jobs instead
retain voltage, adaptation, conductances, short-term plasticity, chemical state,
delay buffers, learning traces, RNG state, absolute time and recording accumulators.
These are distinct contracts. [Learning and persistence](LEARNING.md) documents
weight publication and session retry behaviour.

The server binds to loopback and checks Host/Origin and request bounds. One
execution lock protects shared simulation resources and session publication.
Model records under `.local/` exclude browser names, photographs and event notes.
Browser memory writes a validated candidate before replacing its in-memory state.
Keep browser backups separate from numeric model checkpoints.
