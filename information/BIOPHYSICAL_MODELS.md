# Human Emotion - Biophysical Models

## Complete Mathematical Formulation

---

## 1. Adaptive Exponential Integrate-and-Fire (AdEx) Neuron

### 1.1 Membrane Equation

The AdEx model extends the Leaky Integrate-and-Fire (LIF) with an exponential upswing term:

```
dv/dt = (1/τm) * [v_rest - v + δT * exp((v - v_th) / δT) - w + I_syn + I_mod]
```

Where:
- `v`: Membrane potential (mV)
- `v_rest`: Resting potential (mV)
- `τm`: Membrane time constant (ms)
- `δT`: Slope factor / sharpness of exponential upswing (mV)
- `v_th`: Threshold potential (mV)
- `w`: Adaptation current (mV)
- `I_syn`: Total synaptic current (mV)
- `I_mod`: Neuromodulator-induced current (mV)

### 1.2 Spike Triggered Adaptation

```
On spike (v ≥ v_peak):
    v ← v_reset
    w ← w + b * adapt_gain
```

Where:
- `b`: Spike-triggered adaptation increment (mV)
- `adapt_gain`: Neuromodulator-modulated adaptation scaling

### 1.3 Adaptation Current Dynamics

```
dw/dt = (1/τw) * [a * (v - v_rest) - w]
```

Where:
- `a`: Subthreshold adaptation coupling
- `τw`: Adaptation time constant (ms)

### 1.4 Four Cell Types

#### Regular Spiking (RS) - Pyramidal Cells
| Parameter | Value | Description |
|-----------|-------|-------------|
| v_rest | -65 mV | Resting potential |
| v_reset | -60 mV | Post-spike reset |
| v_threshold | -50 mV | Spike threshold |
| δT | 2.0 mV | Exponential slope |
| v_peak | -30 mV | Peak for spike detection |
| τm | 20 ms | Membrane time constant |
| a | 0.13 | Adaptation coupling |
| b | 2.5 mV | Adaptation increment |
| τw | 150 ms | Adaptation recovery |
| refractory | 2 ms | Refractory period |
| release_u | 0.35 | Baseline release prob |
| recovery_ms | 450 ms | STD recovery time |
| facilitation_ms | 0 | No facilitation |

#### Fast Spiking (FS) - Interneurons
| Parameter | Value | Description |
|-----------|-------|-------------|
| v_rest | -65 mV | Same as RS |
| v_reset | -65 mV | No post-spike hyperpol |
| v_threshold | -52 mV | Slightly lower |
| δT | 0.5 mV | Sharp upswing |
| v_peak | -30 mV | |
| τm | 10 ms | Fast membrane |
| a | 0.0 | No subthreshold adapt |
| b | 0.0 mV | No spike adapt |
| τw | 100 ms | |
| refractory | 1 ms | Very short |
| release_u | 0.25 | Lower release |
| recovery_ms | 150 ms | Fast STD |
| facilitation_ms | 0 | No facilitation |

#### Medium Spiny (MS) - Striatal Neurons
| Parameter | Value | Description |
|-----------|-------|-------------|
| v_rest | -72 mV | Hyperpolarized |
| v_reset | -65 mV | |
| v_threshold | -50 mV | |
| δT | 2.0 mV | |
| v_peak | -30 mV | |
| τm | 15 ms | Medium time constant |
| a | 0.05 | Weak adaptation |
| b | 1.0 mV | Mild adaptation |
| τw | 200 ms | Slow recovery |
| refractory | 2 ms | |
| release_u | 0.2 | Low release |
| recovery_ms | 200 ms | |
| facilitation_ms | 350 ms | **FACILITATING** |

#### Pacemaker - Neuromodulator Sources
| Parameter | Value | Description |
|-----------|-------|-------------|
| v_rest | -52 mV | **DEPOLARIZED** |
| v_reset | -58 mV | Below rest! |
| v_threshold | -50 mV | Near rest |
| δT | 2.0 mV | |
| v_peak | -30 mV | |
| τm | 30 ms | Slow membrane |
| a | 0.0 | No adaptation |
| b | 6.0 mV | Strong adaptation |
| τw | 600 ms | Very slow |
| refractory | 3 ms | |
| release_u | 0.3 | |
| recovery_ms | 300 ms | |
| facilitation_ms | 0 | No facilitation |

**Pacemaker logic**: Near-threshold rest → spontaneous firing → adaptation pulls down → recovery → repeat

---

## 2. Conductance-Based Synapses

### 2.1 Synaptic Current

```
I_syn = g_AMPA * (E_AMPA - v) + g_NMDA * MgBlock(v) * (E_NMDA - v)
      + g_GABA_A * (E_GABA_A - v) + g_GABA_B * (E_GABA_B - v)
```

Where MgBlock(v) is the voltage-dependent magnesium block.

### 2.2 AMPA Conductance

```
dg_AMPA/dt = -g_AMPA / τ_AMPA + Σ w_ij * release_j

τ_AMPA = 5 ms
E_AMPA = 0 mV
```

AMPA receptors are:
- Fast (5 ms decay)
- Linear conductance
- Ionotropic (Na+/K+)

### 2.3 NMDA Conductance

```
dg_NMDA/dt = -g_NMDA / τ_NMDA + g_AMPA * r_NMDA/AMPA * nmda_gain

τ_NMDA = 100 ms
E_NMDA = 0 mV
r_NMDA/AMPA = 0.08 (NMDA/AMPA ratio)
```

**Magnesium Block**:
```
MgBlock(v) = 1 / (1 + (Mg²⁺/3.57) * exp(-0.062 * v))
           = 1 / (1 + 0.28 * exp(-0.062 * v))
```

At v = -65 mV (rest): MgBlock ≈ 0.94 (mostly blocked)
At v = -30 mV (peak): MgBlock ≈ 0.26 (more open)

This creates voltage-dependent plasticity - NMDA only conducts strongly when the cell is depolarized.

### 2.4 GABA-A Conductance

```
dg_GABA_A/dt = -g_GABA_A / τ_GABA_A + Σ w_ij * release_j

τ_GABA_A = 8 ms
E_GABA_A = -75 mV (hyperpolarizing)
```

GABA-A is:
- Fast (8 ms decay)
- Hyperpolarizing (E = -75 mV)
- Ionotropic (Cl⁻ channel)

### 2.5 GABA-B Conductance

```
dg_GABA_B/dt = -g_GABA_B / τ_GABA_B + g_GABA_A * r_GABA_B/A

τ_GABA_B = 150 ms
E_GABA_B = -90 mV (more hyperpolarizing)
r_GABA_B/A = 0.03 (small contribution)
```

GABA-B is:
- Very slow (150 ms decay)
- Metabotropic (G-protein coupled)
- G-protein cascade (slower onset)

---

## 3. Tsodyks-Markram Short-Term Plasticity

### 3.1 Three-State Model

Each synapse has three normalized variables:
- `x ∈ [0,1]`: Fraction of available vesicles
- `u ∈ [0,1]`: Release probability
- `u₀`: Baseline release probability (release_u)

### 3.2 Depression (Vesicle Depletion)

```
dx/dt = (1 - x) / τ_rec

τ_rec = recovery_ms
```

Recovery from depression after a spike:
```
x_new = x_old + (1 - x_old) * (1 - exp(-dt/τ_rec))
```

### 3.3 Facilitation

```
du/dt = (u_target - u) / τ_facil

τ_facil = facilitation_ms
u_target = u₀ + (u + u₀ * (1 - u) - u₀)  [on spike]
```

For facilitating synapses (medium_spiny):
```
u_spike = u + u₀ * (1 - u)  [facilitation update]
```

### 3.4 Synaptic Release

```
On spike:
    released = u_spike * x_old * spike
    
    x_new = x_old - released  [depression]
    u_new = u_spike          [update facilitation]
    
    release = released / u₀  [normalized to [0,1]]
```

The normalized `release` variable means a fully rested synapse (u=u₀, x=1) transmits at nominal weight.

### 3.5 Cell-Type Plasticity Profiles

| Cell Type | depression_ms | facilitation_ms | Dominant |
|-----------|---------------|-----------------|---------|
| regular_spiking | 450 | 0 | STD |
| fast_spiking | 150 | 0 | Fast STD |
| medium_spiny | 200 | 350 | **Facilitation** |
| pacemaker | 300 | 0 | Moderate STD |

---

## 4. Neuromodulator Dynamics

### 4.1 Release Model

Each neuromodulator m has:
- Source region r_source
- Clearance time constant τ_m
- Resting source firing rate Hz_rest[m]

```
Chemical level in region r:
    target[r,m] = K * density[r,m] * (source_firing[m] / Hz_rest[m])
    chem[r,m] += (target[r,m] - chem[r,m]) * (1 - exp(-dt/τ_m))
```

Where:
- `K = 1.0` (half-occupancy constant)
- `density[r,m]` = projection density of modulator m into region r

### 4.2 Receptor Occupancy

```
occupancy[r,m] = chem[r,m] / (chem[r,m] + K)
```

At resting state (source fires at Hz_rest):
```
occupancy_rest = density / (density + K)
```

### 4.3 Effect on Neurons

Three effect channels per modulator:

#### Excitability Shift
```
I_mod = Σ_m excitability[r,m] * Δoccupancy[r,m] * excitability_scale

excitability_scale = 12.0 mV per full occupancy
Δoccupancy = occupancy - occupancy_rest
```

#### Adaptation Scaling
```
adapt_gain = clamp(1 + gain_scale * Σ_m adaptation[r,m] * Δoccupancy[r,m], 0.2, 3.0)

gain_scale = 2.0 per full occupancy
```

When neuromodulators increase adaptation → neurons adapt more → firing rate decreases despite constant input

#### NMDA Gain Scaling
```
nmda_gain = clamp(1 + gain_scale * Σ_m nmda[r,m] * Δoccupancy[r,m], 0.2, 3.0)
```

When neuromodulators increase NMDA gain → more NMDA current → more Ca²⁺ entry → more plasticity

### 4.4 Five Neuromodulators

| Modulator | Source | τ_clearance | Hz_rest | Primary Effect |
|-----------|--------|-------------|---------|---------------|
| Dopamine | VTA | 400 ms | 3.2 Hz | Reward, motivation |
| Noradrenaline | LC | 1000 ms | 2.7 Hz | Arousal, attention |
| Serotonin | Raphe | 2000 ms | 2.1 Hz | Mood, impulse control |
| Acetylcholine | BF | 200 ms | 2.5 Hz | Attention, plasticity |
| Oxytocin | PVN | 8000 ms | 2.4 Hz | Social bonding |

---

## 5. Conductance Delays

### 5.1 Ring Buffer

```
Local connections: 1 ms delay (dt = 1 ms)
Long-range connections: 5 ms delay (5 * dt)

Ring buffer:
    ring[step % delay] = release  [write]
    delayed = ring[(step - delay) % delay]  [read]
```

### 5.2 Synaptic Routing

```
Local synapses:     use 'release' directly
Long-range synapses: use 'delayed' from ring buffer
```

---

## 6. Spike Timing Dependent Plasticity (STDP) - Future

### 6.1 Classical STDP Rule

```
if Δt = t_post - t_pre > 0 (causal):
    Δw = A_plus * exp(-Δt / τ_plus)
    (LTP - strengthen causal connections)
    
else (anti-causal):
    Δw = -A_minus * exp(Δt / τ_minus)
    (LTD - weaken anti-causal connections)
```

Parameters:
- `A_plus = 0.001` (LTP amplitude)
- `A_minus = 0.0012` (LTD amplitude)
- `τ_plus = 20 ms`
- `τ_minus = 20 ms`

### 6.2 Soft Bounds

```
w = clamp(w, 0.1 * w_max, 0.9 * w_max)
```

### 6.3 Neuromodulator Gating

```
DA rise  → LTP enhanced
DA fall  → LTD enhanced
NE rise  → LTP enhanced
ACh rise → plasticity enabled
```

---

## 7. Gap Junctions - Future

### 7.1 Electrical Coupling

```
I_gap = g_gap * (v_neighbor - v_self)
```

Gap junctions:
- PV+ interneurons: electrical coupling for gamma synchrony
- Striatal MSNs: coupling for coordinated output
- Network oscillations emerge from gap junction coupling

---

## 8. Field Potential (LFP) Proxy

### 8.1 LFP Model

Based on Mazzoni et al. (2015):

```
LFP(t) = |I_AMPA|(t - 6ms) - 1.65 * |I_GABA|(t)
```

Where:
- `6 ms` = synaptic and conduction delay for AMPA contribution
- `1.65` = weighting factor for GABA contribution

### 8.2 Regional Weighting

Three LFP channels:

**Frontal**:
```
LFP_frontal = 0.45 * LFP(vmpfc) + 0.35 * LFP(dlpfc) + 0.2 * LFP(dACC)
```

**Central**:
```
LFP_central = 0.4 * LFP(sensory_cortex) + 0.35 * LFP(insula) + 0.25 * LFP(dACC)
```

**Posterior**:
```
LFP_posterior = 0.7 * LFP(sensory_cortex) + 0.3 * LFP(hippocampus)
```

---

## 9. Input Model

### 9.1 Poisson Spike Trains

```
λ_stim(t) = λ_base + stimulus_on * assembly_fraction * level * max_rate * weight
```

Where:
- `λ_base`: Background rate (2000 Hz, scaled by interneuron_scale)
- `stimulus_on`: 1 during 2.5-5s window, 0 otherwise
- `assembly_fraction`: Fraction of cells in target region activated
- `level`: User slider value [0,1]
- `max_rate`: Max spike rate for this input (3000 Hz)
- `weight`: Projection density to this region

### 9.2 Tonic Inputs

Present for entire 6s run:

| Input | Default | Target Regions |
|-------|---------|---------------|
| body_arousal | 0.35 | insula, LC, hypothalamus, amygdala_cea |
| hunger | 0.2 | hypothalamus |
| fatigue | 0.25 | (via adenosine) |
| stress | 0.2 | (via cortisol) |
| control | 0.55 | dLPFC, vmPFC |
| uncertainty | 0.4 | dACC, BNST, LC |

---

## 10. Second Messenger Cascades - Future

### 10.1 cAMP/PKA Pathway (D1, β-AR, 5-HT4/6/7)

```
Receptor → Gs → Adenylyl cyclase → cAMP → PKA
```

Effects:
- PKA phosphorylates ion channels → enhanced excitability
- PKA phosphorylates AMPA receptors → enhanced transmission
- PKA phosphorylates transcription factors → gene expression

### 10.2 PLC/IP3/DAG Pathway (α1-AR, 5-HT2, M1)

```
Receptor → Gq → PLC → IP3 + DAG
IP3 → Ca²⁺ release from ER
DAG → PKC activation
```

Effects:
- PKC modulates K+ channels → reduced adaptation
- PKC modulates NMDA receptors → enhanced transmission
- Intracellular Ca²⁺ → plasticity

---

*Document generated: 2026-09-28*
*Part of Human Emotion - Comprehensive Analysis*
